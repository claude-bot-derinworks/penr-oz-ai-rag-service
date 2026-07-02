//! End-to-end test of `penr-oz-rag serve`: boot the real binary on a random port,
//! then exercise `POST /retrieve` over plain HTTP/1.1 the way `curl` would.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use tempfile::tempdir;

/// Kills the served binary when the test ends, pass or fail.
struct ServerGuard(Child);

impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Spawn `penr-oz-rag serve <corpus> --addr 127.0.0.1:0` and wait for it to report the
/// address it actually bound.
fn start_server(corpus: &std::path::Path) -> (ServerGuard, String) {
    let child = Command::new(env!("CARGO_BIN_EXE_penr-oz-rag"))
        .arg("serve")
        .arg(corpus)
        .args([
            "--addr",
            "127.0.0.1:0",
            "--chunk-size",
            "64",
            "--overlap",
            "8",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("serve binary spawns");
    let mut guard = ServerGuard(child);

    // The binary prints the bound address once the listener is ready; everything
    // before that line is the ingestion report.
    let stdout = guard.0.stdout.take().expect("stdout is piped");
    let mut lines = BufReader::new(stdout).lines();
    let addr = loop {
        let line = lines
            .next()
            .expect("server exited before reporting its address")
            .expect("server stdout is utf-8");
        if let Some(rest) = line.strip_prefix("Serving POST /retrieve on http://") {
            break rest.trim().to_string();
        }
    };
    (guard, addr)
}

/// Send one `POST /retrieve` with a JSON `body` and return `(status, response body)`.
fn post_retrieve(addr: &str, body: &str) -> (u16, String) {
    let mut stream = TcpStream::connect(addr).expect("connect to served address");
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();

    let request = format!(
        "POST /retrieve HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).expect("send request");

    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .expect("read full response");

    let status: u16 = response
        .split_whitespace()
        .nth(1)
        .expect("status line present")
        .parse()
        .expect("numeric status code");
    let payload = match response.find("\r\n\r\n") {
        Some(index) => response[index + 4..].to_string(),
        None => String::new(),
    };
    (status, payload)
}

#[test]
fn serves_retrieval_over_http() {
    // Each file is shorter than the 64-character chunk size, so each becomes exactly
    // one chunk whose content is the whole file — which makes retrieval assertable:
    // the mock provider embeds deterministically, so querying one file's exact text
    // is a (near-)perfect cosine match for its chunk.
    let corpus = tempdir().unwrap();
    std::fs::write(
        corpus.path().join("a.txt"),
        "retrieval augmented generation",
    )
    .unwrap();
    std::fs::write(
        corpus.path().join("b.txt"),
        "cosine similarity vector search",
    )
    .unwrap();

    let (_server, addr) = start_server(corpus.path());

    // Happy path: the exact-text match ranks first with a near-perfect score, and the
    // response carries content, metadata, and scores for each hit.
    let (status, body) = post_retrieve(
        &addr,
        r#"{"query": "retrieval augmented generation", "top_k": 2}"#,
    );
    assert_eq!(status, 200, "body: {body}");
    let json: serde_json::Value = serde_json::from_str(&body).expect("valid JSON body");
    let results = json["results"].as_array().expect("results array");
    assert_eq!(results.len(), 2);
    assert_eq!(
        results[0]["chunk"]["content"],
        "retrieval augmented generation"
    );
    assert!(results[0]["score"].as_f64().expect("numeric score") > 0.999);
    assert!(results[0]["chunk"]["metadata"]["source"].is_string());

    // top_k omitted falls back to the default and still succeeds.
    let (status, body) = post_retrieve(&addr, r#"{"query": "vector search"}"#);
    assert_eq!(status, 200, "body: {body}");
    let json: serde_json::Value = serde_json::from_str(&body).expect("valid JSON body");
    assert!(json["results"].as_array().expect("results array").len() <= 5);

    // Validation: a whitespace-only query is a 400 with an error message, not a 5xx.
    let (status, body) = post_retrieve(&addr, r#"{"query": "   "}"#);
    assert_eq!(status, 400, "body: {body}");
    let json: serde_json::Value = serde_json::from_str(&body).expect("valid JSON body");
    assert!(json["error"]
        .as_str()
        .expect("error message")
        .contains("empty"));

    // Malformed JSON is rejected by the extractor with a client error, not a crash.
    let (status, _) = post_retrieve(&addr, "{not json");
    assert!((400..500).contains(&status));
}
