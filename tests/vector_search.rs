//! End-to-end retrieval test: embed chunk text with an [`EmbeddingProvider`], index the
//! results in a [`VectorStore`], and search by embedding the query the same way.

use penr_oz_ai_rag_service::{
    Chunk, ChunkMetadata, EmbeddedChunk, EmbeddingProvider, InMemoryVectorStore, Metadata,
    MockEmbeddingProvider, VectorStore,
};

/// Build a chunk with the given id and content; metadata is filler the store carries
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

#[tokio::test]
async fn embed_index_and_retrieve_the_matching_chunk() {
    let provider = MockEmbeddingProvider::new();
    let store = InMemoryVectorStore::new();

    // A tiny corpus. The mock embeds deterministically, so re-embedding identical text
    // (the query below reuses one chunk's exact words) yields an identical vector — a
    // perfect cosine match — which lets us assert retrieval precisely.
    let corpus = [
        chunk("c0", "retrieval augmented generation"),
        chunk("c1", "fixed size character chunking"),
        chunk("c2", "cosine similarity vector search"),
    ];

    let texts: Vec<&str> = corpus.iter().map(|c| c.content.as_str()).collect();
    let embeddings = provider.embed(&texts).await.unwrap();

    let items: Vec<EmbeddedChunk> = corpus
        .iter()
        .cloned()
        .zip(embeddings)
        .map(|(chunk, embedding)| EmbeddedChunk::new(chunk, embedding))
        .collect();
    store.insert(&items).await.unwrap();
    assert_eq!(store.len().await.unwrap(), corpus.len());

    // Query with the exact text of c2; its embedding matches c2's stored vector.
    let query = provider
        .embed(&["cosine similarity vector search"])
        .await
        .unwrap();
    let results = store.search(&query[0], 2).await.unwrap();

    assert_eq!(results.len(), 2);
    // The exact-text match ranks first, with a (near-)perfect cosine score, and its
    // text and metadata travel back with it.
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
async fn dimensions_flow_from_provider_to_store() {
    // A provider's dimensionality determines the store's: vectors embedded by the same
    // provider all share a length, so they index without a dimension mismatch, and a
    // query embedded the same way searches cleanly.
    let provider = MockEmbeddingProvider::new().with_dimensions(16);
    let store = InMemoryVectorStore::new();

    let docs = chunk("only", "a single document");
    let embedding = provider.embed(&[docs.content.as_str()]).await.unwrap();
    assert_eq!(embedding[0].len(), provider.dimensions());

    store
        .insert(&[EmbeddedChunk::new(docs, embedding[0].clone())])
        .await
        .unwrap();

    let query = provider.embed(&["any query text"]).await.unwrap();
    let results = store.search(&query[0], 5).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].chunk.id, "only");
}
