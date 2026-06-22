//! End-to-end tests driving the pipeline through its public API.

use std::fs;

use penr_oz_ai_rag_service::{
    Chunk, ChunkStore, FixedSizeChunker, InMemoryStorage, IngestionPipeline, JsonlStorage,
};
use tempfile::tempdir;

#[test]
fn ingest_directory_to_jsonl_round_trips() {
    let dir = tempdir().unwrap();
    let docs = dir.path().join("docs");
    fs::create_dir(&docs).unwrap();
    fs::write(
        docs.join("alpha.txt"),
        "Retrieval augmented generation grounds a language model in external text. \
         The ingestion pipeline loads documents, splits them into overlapping chunks, \
         attaches provenance metadata, and persists the result for later retrieval.",
    )
    .unwrap();
    fs::write(
        docs.join("beta.text"),
        "A second document so the directory walk has more than one supported file to ingest.",
    )
    .unwrap();
    // Unsupported format: should be skipped, not fail the run.
    fs::write(docs.join("ignore.bin"), [0u8, 159, 146, 150]).unwrap();

    let output = dir.path().join("out/chunks.jsonl");
    let chunker = FixedSizeChunker::new(80, 16).unwrap();

    let mut pipeline = IngestionPipeline::builder(JsonlStorage::create(&output).unwrap())
        .chunker(chunker)
        .build();
    let report = pipeline.ingest_path(&docs).unwrap();
    pipeline.flush().unwrap();

    assert_eq!(report.files_ingested, 2);
    assert_eq!(report.files_skipped, 1);
    assert!(report.chunks_created >= 2);

    // Every line is a valid Chunk, and the count matches the report.
    let body = fs::read_to_string(&output).unwrap();
    let chunks: Vec<Chunk> = body
        .lines()
        .map(|line| serde_json::from_str(line).expect("each line is a JSON chunk"))
        .collect();
    assert_eq!(chunks.len(), report.chunks_created);

    // Chunk indices and provenance are coherent per source document.
    for chunk in &chunks {
        assert!(chunk.id.starts_with(&chunk.metadata.source));
        assert!(chunk.metadata.chunk_index < chunk.metadata.total_chunks);
        assert_eq!(
            chunk.metadata.extra.get("loader").map(String::as_str),
            Some("text")
        );
        assert!(chunk.metadata.start_char < chunk.metadata.end_char);
    }
}

#[test]
fn default_pipeline_uses_text_loader_and_fixed_chunker() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("note.txt");
    fs::write(&path, "short note").unwrap();

    // No chunker / loaders configured -> defaults apply.
    let mut pipeline = IngestionPipeline::builder(InMemoryStorage::new()).build();
    let report = pipeline.ingest_path(&path).unwrap();

    assert_eq!(report.files_ingested, 1);
    let store = pipeline.into_store();
    assert_eq!(store.len().unwrap(), report.chunks_created);
    assert_eq!(store.chunks()[0].content, "short note");
}
