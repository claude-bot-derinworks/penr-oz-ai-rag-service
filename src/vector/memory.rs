//! An in-memory [`VectorStore`] for development and tests.

use std::sync::RwLock;

use async_trait::async_trait;

use super::{cosine_similarity, EmbeddedChunk, SearchResult, VectorStore, VectorStoreError};

/// The guarded state: the indexed vectors plus the dimensionality they all share.
///
/// Bundling both under one lock keeps them consistent — the dimensionality is only ever
/// read or updated together with the vectors it describes.
#[derive(Debug, Default)]
struct Index {
    items: Vec<EmbeddedChunk>,
    /// Established by the first inserted vector; `None` while the store is empty.
    dimensions: Option<usize>,
}

/// A [`VectorStore`] that keeps every [`EmbeddedChunk`] in memory and answers searches
/// with an exact (brute-force) cosine-similarity scan.
///
/// It is the natural default for development and tests: no external service, fully
/// deterministic, and exact rather than approximate. Search is `O(n)` in the number of
/// indexed chunks, which is fine for the modest corpora those settings involve; a
/// production deployment would swap in an approximate-nearest-neighbor backend behind
/// the same [`VectorStore`] trait.
///
/// Inserts and searches take `&self` (the index lives behind an [`RwLock`]), so the
/// store can be shared — e.g. as an `Arc<InMemoryVectorStore>` — and searched
/// concurrently.
#[derive(Debug, Default)]
pub struct InMemoryVectorStore {
    index: RwLock<Index>,
}

impl InMemoryVectorStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn insert(&self, items: &[EmbeddedChunk]) -> Result<(), VectorStoreError> {
        if items.is_empty() {
            return Ok(());
        }

        let mut index = self.index.write().expect("vector store lock poisoned");

        // The expected dimensionality is the store's established one, or — if this is the
        // first insert — the first item's. Validate the whole batch before touching the
        // index so a single bad item can't leave a partial insert behind.
        let expected = index.dimensions.unwrap_or(items[0].embedding.len());
        for item in items {
            if item.embedding.is_empty() {
                return Err(VectorStoreError::EmptyEmbedding {
                    id: item.chunk.id.clone(),
                });
            }
            if item.embedding.len() != expected {
                return Err(VectorStoreError::DimensionMismatch {
                    expected,
                    actual: item.embedding.len(),
                });
            }
        }

        index.dimensions = Some(expected);
        index.items.extend_from_slice(items);
        Ok(())
    }

    async fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, VectorStoreError> {
        let index = self.index.read().expect("vector store lock poisoned");

        if k == 0 || index.items.is_empty() {
            return Ok(Vec::new());
        }

        if let Some(dimensions) = index.dimensions {
            if query.len() != dimensions {
                return Err(VectorStoreError::DimensionMismatch {
                    expected: dimensions,
                    actual: query.len(),
                });
            }
        }

        // Score every indexed vector by reference first, keeping only `(position, score)`
        // pairs. The chunks themselves are cloned afterwards for just the top `k` hits, so
        // a search never clones the whole store to throw most of it away.
        let mut scored: Vec<(usize, f32)> = index
            .items
            .iter()
            .enumerate()
            .map(|(position, item)| (position, cosine_similarity(query, &item.embedding)))
            .collect();

        // Most similar first. `total_cmp` gives a total order over floats, so ties and
        // any stray NaN sort deterministically rather than triggering a panic.
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(k);

        Ok(scored
            .into_iter()
            .map(|(position, score)| SearchResult {
                chunk: index.items[position].chunk.clone(),
                score,
            })
            .collect())
    }

    async fn len(&self) -> Result<usize, VectorStoreError> {
        Ok(self
            .index
            .read()
            .expect("vector store lock poisoned")
            .items
            .len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Chunk, ChunkMetadata, Metadata};

    fn embedded(id: &str, embedding: impl Into<Vec<f32>>) -> EmbeddedChunk {
        EmbeddedChunk::new(
            Chunk {
                id: id.to_string(),
                content: format!("content of {id}"),
                metadata: ChunkMetadata {
                    source: "doc".to_string(),
                    chunk_index: 0,
                    total_chunks: 1,
                    start_char: 0,
                    end_char: 0,
                    extra: Metadata::new(),
                },
            },
            embedding,
        )
    }

    #[tokio::test]
    async fn empty_store_reports_empty() {
        let store = InMemoryVectorStore::new();
        assert_eq!(store.len().await.unwrap(), 0);
        assert!(store.is_empty().await.unwrap());
        // Searching an empty index is not an error; it just returns nothing.
        assert!(store.search(&[1.0, 0.0], 5).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn insert_accumulates_across_calls() {
        let store = InMemoryVectorStore::new();
        store.insert(&[embedded("a", [1.0, 0.0])]).await.unwrap();
        store
            .insert(&[embedded("b", [0.0, 1.0]), embedded("c", [1.0, 1.0])])
            .await
            .unwrap();

        assert_eq!(store.len().await.unwrap(), 3);
        assert!(!store.is_empty().await.unwrap());
    }

    #[tokio::test]
    async fn search_ranks_by_similarity_and_returns_text_and_metadata() {
        let store = InMemoryVectorStore::new();
        store
            .insert(&[
                embedded("east", [1.0, 0.0]),
                embedded("north", [0.0, 1.0]),
                embedded("northeast", [1.0, 1.0]),
            ])
            .await
            .unwrap();

        // A query pointing mostly east should rank "east" first, then "northeast".
        let results = store.search(&[0.9, 0.1], 2).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk.id, "east");
        assert_eq!(results[1].chunk.id, "northeast");
        // Scores are sorted descending...
        assert!(results[0].score >= results[1].score);
        // ...and each result carries the chunk's text and metadata.
        assert_eq!(results[0].content(), "content of east");
        assert_eq!(results[0].metadata().source, "doc");
    }

    #[tokio::test]
    async fn search_caps_results_at_k() {
        let store = InMemoryVectorStore::new();
        store
            .insert(&[
                embedded("a", [1.0, 0.0]),
                embedded("b", [0.0, 1.0]),
                embedded("c", [1.0, 1.0]),
            ])
            .await
            .unwrap();

        assert_eq!(store.search(&[1.0, 0.0], 1).await.unwrap().len(), 1);
        // k larger than the corpus simply returns everything.
        assert_eq!(store.search(&[1.0, 0.0], 99).await.unwrap().len(), 3);
        // k == 0 returns nothing.
        assert!(store.search(&[1.0, 0.0], 0).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn insert_rejects_dimension_mismatch_without_mutating() {
        let store = InMemoryVectorStore::new();
        store.insert(&[embedded("a", [1.0, 0.0])]).await.unwrap();

        match store.insert(&[embedded("b", [1.0, 0.0, 0.0])]).await {
            Err(VectorStoreError::DimensionMismatch { expected, actual }) => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 3);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
        // The rejected batch left the store untouched.
        assert_eq!(store.len().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn insert_is_all_or_nothing_within_a_batch() {
        let store = InMemoryVectorStore::new();
        // The second item is malformed, so the whole (first inclusive) batch is rejected.
        let err = store
            .insert(&[embedded("a", [1.0, 0.0]), embedded("b", [1.0, 0.0, 0.0])])
            .await
            .unwrap_err();

        assert!(matches!(err, VectorStoreError::DimensionMismatch { .. }));
        assert_eq!(store.len().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn insert_rejects_empty_embedding() {
        let store = InMemoryVectorStore::new();
        match store.insert(&[embedded("a", Vec::<f32>::new())]).await {
            Err(VectorStoreError::EmptyEmbedding { id }) => assert_eq!(id, "a"),
            other => panic!("expected EmptyEmbedding, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn search_rejects_query_of_wrong_dimension() {
        let store = InMemoryVectorStore::new();
        store.insert(&[embedded("a", [1.0, 0.0])]).await.unwrap();

        match store.search(&[1.0, 0.0, 0.0], 1).await {
            Err(VectorStoreError::DimensionMismatch { expected, actual }) => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 3);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn usable_as_a_shared_trait_object() {
        use std::sync::Arc;

        let store: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::new());
        store.insert(&[embedded("a", [1.0, 0.0])]).await.unwrap();
        let results = store.search(&[1.0, 0.0], 1).await.unwrap();
        assert_eq!(results[0].chunk.id, "a");
    }
}
