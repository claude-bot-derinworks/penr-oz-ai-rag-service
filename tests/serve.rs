//! End-to-end test of `penr-oz-rag serve`: boot the real binary on a random port,
//! then exercise `POST /retrieve` and `POST /answer` over plain HTTP/1.1 the way
//! `curl` would.

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

/// Send one `POST` to `path` with a JSON `body` and return `(status, response body)`.
fn post_json(addr: &str, path: &str, body: &str) -> (u16, String) {
    let mut stream = TcpStream::connect(addr).expect("connect to served address");
    // Generous timeout: both serve tests boot a real server and can run concurrently
    // on a loaded CI machine, so a tight deadline would flake, not catch bugs.
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();

    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\n\
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
    let (status, body) = post_json(
        &addr,
        "/retrieve",
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
    let (status, body) = post_json(&addr, "/retrieve", r#"{"query": "vector search"}"#);
    assert_eq!(status, 200, "body: {body}");
    let json: serde_json::Value = serde_json::from_str(&body).expect("valid JSON body");
    assert!(json["results"].as_array().expect("results array").len() <= 5);

    // Validation: a whitespace-only query is a 400 with an error message, not a 5xx.
    let (status, body) = post_json(&addr, "/retrieve", r#"{"query": "   "}"#);
    assert_eq!(status, 400, "body: {body}");
    let json: serde_json::Value = serde_json::from_str(&body).expect("valid JSON body");
    assert!(json["error"]
        .as_str()
        .expect("error message")
        .contains("empty"));

    // Malformed JSON is rejected by the extractor with a client error, not a crash.
    let (status, _) = post_json(&addr, "/retrieve", "{not json");
    assert!((400..500).contains(&status));
}

#[test]
fn serves_answer_generation_over_http() {
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

    // Happy path: the served mock LLM echoes the prompt it was shown, so the answer
    // proves the retrieved context and the question reached the model, and the sources
    // attribute the answer to the chunks it was grounded in.
    let (status, body) = post_json(
        &addr,
        "/answer",
        r#"{"query": "retrieval augmented generation", "top_k": 2}"#,
    );
    assert_eq!(status, 200, "body: {body}");
    let json: serde_json::Value = serde_json::from_str(&body).expect("valid JSON body");
    let answer = json["answer"].as_str().expect("answer string");
    assert!(answer.contains("retrieval augmented generation"));
    assert!(answer.contains("Question: retrieval augmented generation"));
    let sources = json["sources"].as_array().expect("sources array");
    assert!(!sources.is_empty());
    assert!(sources[0]["id"].is_string());
    assert!(sources[0]["source"].is_string());
    assert!(sources[0]["score"].as_f64().expect("numeric score") > 0.999);

    // Confidence gating: only the exact-text match clears a 0.99 minimum score, so the
    // unrelated chunk appears in neither the prompt (echoed answer) nor the sources.
    let (status, body) = post_json(
        &addr,
        "/answer",
        r#"{"query": "retrieval augmented generation", "top_k": 2, "min_score": 0.99}"#,
    );
    assert_eq!(status, 200, "body: {body}");
    let json: serde_json::Value = serde_json::from_str(&body).expect("valid JSON body");
    assert_eq!(json["sources"].as_array().expect("sources array").len(), 1);
    assert!(!json["answer"]
        .as_str()
        .expect("answer string")
        .contains("cosine similarity vector search"));

    // A gate no chunk can clear (cosine scores never exceed 1.0) yields the
    // no-context answer with no sources — the model is never prompted with noise.
    let (status, body) = post_json(
        &addr,
        "/answer",
        r#"{"query": "something unrelated", "min_score": 1.1}"#,
    );
    assert_eq!(status, 200, "body: {body}");
    let json: serde_json::Value = serde_json::from_str(&body).expect("valid JSON body");
    assert_eq!(
        json["answer"].as_str().expect("answer string"),
        penr_oz_ai_rag_service::NO_CONTEXT_ANSWER
    );
    assert!(json["sources"]
        .as_array()
        .expect("sources array")
        .is_empty());

    // Validation: a whitespace-only query is a 400 with an error message, not a 5xx.
    let (status, body) = post_json(&addr, "/answer", r#"{"query": "   "}"#);
    assert_eq!(status, 400, "body: {body}");
    let json: serde_json::Value = serde_json::from_str(&body).expect("valid JSON body");
    assert!(json["error"]
        .as_str()
        .expect("error message")
        .contains("empty"));
}
