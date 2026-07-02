//! End-to-end retrieval test, exercising the [`Retriever`] the way a `POST /retrieve`
//! handler would: index a corpus, then answer queries with ranked, scored chunks and
//! reject invalid input.

use penr_oz_ai_rag_service::{
    Chunk, ChunkMetadata, InMemoryVectorStore, Metadata, MockEmbeddingProvider, RetrievalError,
    RetrievalRequest, Retriever,
};

/// Build a chunk with the given id and content; metadata is filler the retriever carries
/// through to results untouched.
fn chunk(id: &str, content: &str) -> Chunk {
    Chunk {
        id: id.to_string(),
        content: content.to_string(),
        metadata: ChunkMetadata {
            source: "corpus".to_string(),
            chunk_index: 0,
            total_chunks: 1,
            start_char: 0,
            end_char: content.chars().count(),
            extra: Metadata::from([("loader".to_string(), "text".to_string())]),
        },
    }
}

fn retriever() -> Retriever<MockEmbeddingProvider, InMemoryVectorStore> {
    Retriever::new(MockEmbeddingProvider::new(), InMemoryVectorStore::new())
}

#[tokio::test]
async fn indexes_then_retrieves_the_matching_chunk() {
    let retriever = retriever();

    // The mock embeds deterministically, so re-embedding identical text (the query reuses
    // one chunk's exact words) yields an identical vector — a perfect cosine match — which
    // lets us assert retrieval precisely.
    let indexed = retriever
        .index(vec![
            chunk("c0", "retrieval augmented generation"),
            chunk("c1", "fixed size character chunking"),
            chunk("c2", "cosine similarity vector search"),
        ])
        .await
        .expect("indexing succeeds");
    assert_eq!(indexed, 3);

    let results = retriever
        .retrieve("cosine similarity vector search", 2)
        .await
        .expect("retrieval succeeds");

    assert_eq!(results.len(), 2);
    // The exact-text match ranks first with a (near-)perfect score, text and metadata
    // travelling back with it.
    assert_eq!(results[0].chunk.id, "c2");
    assert!((results[0].score - 1.0).abs() < 1e-6);
    assert_eq!(results[0].content(), "cosine similarity vector search");
    assert_eq!(results[0].metadata().source, "corpus");
    assert_eq!(
        results[0]
            .metadata()
            .extra
            .get("loader")
            .map(String::as_str),
        Some("text")
    );
    // The runner-up is a real, distinct chunk scored no higher than the top hit.
    assert_ne!(results[1].chunk.id, "c2");
    assert!(results[1].score <= results[0].score);
}

#[tokio::test]
async fn handle_returns_the_endpoint_response_shape() {
    let retriever = retriever();
    retriever
        .index(vec![
            chunk("c0", "hello world"),
            chunk("c1", "goodbye moon"),
        ])
        .await
        .unwrap();

    // A request with top_k omitted from JSON falls back to the default and still works.
    let request: RetrievalRequest = serde_json::from_str(r#"{"query": "hello world"}"#).unwrap();
    let response = retriever.handle(&request).await.unwrap();

    assert_eq!(response.results.len(), 2);
    assert_eq!(response.results[0].chunk.id, "c0");

    // Round-trips through JSON the way the endpoint would serialize it.
    let json = serde_json::to_string(&response).unwrap();
    let decoded: penr_oz_ai_rag_service::RetrievalResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, response);
}

#[tokio::test]
async fn rejects_empty_and_oversized_queries() {
    let retriever = retriever().with_max_query_chars(16);

    assert!(matches!(
        retriever.retrieve("", 5).await,
        Err(RetrievalError::EmptyQuery)
    ));

    let long = "x".repeat(17);
    match retriever.retrieve(&long, 5).await {
        Err(RetrievalError::QueryTooLong { chars, max }) => {
            assert_eq!(chars, 17);
            assert_eq!(max, 16);
        }
        other => panic!("expected QueryTooLong, got {other:?}"),
    }
}

#[tokio::test]
async fn searching_an_empty_index_yields_no_results() {
    // A valid query against a retriever with nothing indexed is not an error; it just
    // comes back empty.
    let results = retriever().retrieve("anything", 5).await.unwrap();
    assert!(results.is_empty());
}
