//! Indexing embedded chunks and retrieving them by similarity.
//!
//! Once a [`Chunk`](crate::document::Chunk) has been turned into an embedding by an
//! [`EmbeddingProvider`](crate::embedding::EmbeddingProvider), it needs somewhere to
//! live that can answer the retrieval half of RAG: *given a query vector, which chunks
//! are most similar?* That responsibility sits behind the [`VectorStore`] trait, so the
//! concrete backend — the in-process [`InMemoryVectorStore`] used for development and
//! tests, or a hosted vector database (Qdrant, Pinecone, pgvector, …) later — can be
//! swapped without touching callers.
//!
//! The trait is intentionally small:
//!
//! 1. [`insert`](VectorStore::insert) adds [`EmbeddedChunk`]s (a chunk plus its vector,
//!    metadata included) to the index.
//! 2. [`search`](VectorStore::search) returns the top-k [`SearchResult`]s for a query
//!    vector, each carrying the matching chunk's text, metadata, and a similarity
//!    score.
//!
//! Like [`EmbeddingProvider`](crate::embedding::EmbeddingProvider), the trait is async
//! and `Send + Sync` so a single store can be shared (e.g. behind an `Arc`) and called
//! concurrently, and its errors are kept in a dedicated [`VectorStoreError`] so storage
//! concerns stay isolated from the ingestion pipeline's [`RagError`](crate::error::RagError).
//!
//! ## Example
//!
//! ```
//! use penr_oz_ai_rag_service::{
//!     Chunk, ChunkMetadata, EmbeddedChunk, InMemoryVectorStore, VectorStore,
//! };
//!
//! # async fn run() -> Result<(), penr_oz_ai_rag_service::VectorStoreError> {
//! # fn chunk(id: &str) -> Chunk {
//! #     Chunk {
//! #         id: id.to_string(),
//! #         content: id.to_string(),
//! #         metadata: ChunkMetadata {
//! #             source: "doc".into(), chunk_index: 0, total_chunks: 1,
//! #             start_char: 0, end_char: 0, extra: Default::default(),
//! #         },
//! #     }
//! # }
//! let store = InMemoryVectorStore::new();
//! store
//!     .insert(&[
//!         EmbeddedChunk::new(chunk("a"), [1.0, 0.0]),
//!         EmbeddedChunk::new(chunk("b"), [0.0, 1.0]),
//!     ])
//!     .await?;
//!
//! let hits = store.search(&[0.9, 0.1], 1).await?;
//! assert_eq!(hits[0].chunk.id, "a");
//! # Ok(())
//! # }
//! ```

mod memory;

pub use memory::InMemoryVectorStore;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::document::{Chunk, ChunkMetadata};

/// An embedding vector: a chunk's content mapped into a fixed-dimensional space.
///
/// Components are `f32` to match what [`EmbeddingProvider`](crate::embedding::EmbeddingProvider)
/// produces, which keeps embeddings flowing from provider to store without conversion.
pub type Embedding = Vec<f32>;

/// A [`Chunk`] paired with its [`Embedding`], ready to be indexed by a [`VectorStore`].
///
/// The chunk travels with its vector so that a later [`search`](VectorStore::search) can
/// return the original text and metadata alongside the score — no second lookup needed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddedChunk {
    /// The chunk being indexed, including its id, text content, and metadata.
    pub chunk: Chunk,
    /// The embedding vector for `chunk`'s content.
    pub embedding: Embedding,
}

impl EmbeddedChunk {
    /// Pair a `chunk` with its `embedding`.
    pub fn new(chunk: Chunk, embedding: impl Into<Embedding>) -> Self {
        Self {
            chunk,
            embedding: embedding.into(),
        }
    }
}

/// A single hit from a similarity [`search`](VectorStore::search): the matching chunk
/// and how similar it is to the query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    /// The matching chunk, carrying its id, text [`content`](SearchResult::content), and
    /// [`metadata`](SearchResult::metadata).
    pub chunk: Chunk,
    /// Similarity of `chunk` to the query. For the built-in cosine metric this lies in
    /// `[-1.0, 1.0]`, where higher is more similar.
    pub score: f32,
}

impl SearchResult {
    /// The matching chunk's text content.
    pub fn content(&self) -> &str {
        &self.chunk.content
    }

    /// The matching chunk's metadata (source, offsets, propagated provenance, …).
    pub fn metadata(&self) -> &ChunkMetadata {
        &self.chunk.metadata
    }
}

/// An index of [`EmbeddedChunk`]s that can be queried by vector similarity.
///
/// Implementations must be cheap to share (`Send + Sync`) so one store can be used
/// concurrently, and the trait is object-safe so stores can be held behind a
/// `Box<dyn VectorStore>` / `Arc<dyn VectorStore>` and chosen at runtime. Because both
/// methods take `&self`, implementations that mutate (like [`InMemoryVectorStore`]) use
/// interior mutability rather than exposing a `&mut self` API.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Insert `items` into the index.
    ///
    /// All vectors in a store must share one dimensionality: the first inserted vector
    /// establishes it, and every later vector — across calls — must match. Insertion is
    /// all-or-nothing; if any item is invalid the store is left unchanged.
    ///
    /// # Errors
    /// Returns [`VectorStoreError::EmptyEmbedding`] if any item's vector has no
    /// components, or [`VectorStoreError::DimensionMismatch`] if any vector's length
    /// differs from the store's established dimensionality.
    async fn insert(&self, items: &[EmbeddedChunk]) -> Result<(), VectorStoreError>;

    /// Return the `k` chunks most similar to `query`, most similar first.
    ///
    /// At most `k` results are returned (fewer if the index holds fewer chunks). A `k`
    /// of `0`, or a search against an empty index, yields an empty `Vec`.
    ///
    /// # Errors
    /// Returns [`VectorStoreError::DimensionMismatch`] if `query`'s length differs from
    /// the dimensionality of the indexed vectors.
    async fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, VectorStoreError>;

    /// The number of vectors currently indexed.
    async fn len(&self) -> Result<usize, VectorStoreError>;

    /// Whether the index holds no vectors.
    async fn is_empty(&self) -> Result<bool, VectorStoreError> {
        Ok(self.len().await? == 0)
    }
}

/// The set of errors a [`VectorStore`] can produce.
///
/// Kept separate from [`RagError`](crate::error::RagError) — mirroring
/// [`EmbeddingError`](crate::embedding::EmbeddingError) — so vector-store concerns stay
/// isolated from the ingestion pipeline's error surface.
#[derive(Debug, Error)]
pub enum VectorStoreError {
    /// A vector's length does not match the dimensionality the store was established
    /// with (by its first insert).
    #[error(
        "embedding dimension mismatch: store holds {expected}-dimensional vectors, got {actual}"
    )]
    DimensionMismatch {
        /// The dimensionality the store expects.
        expected: usize,
        /// The dimensionality that was supplied.
        actual: usize,
    },

    /// An embedding had no components; a zero-length vector cannot be indexed or scored.
    #[error("embedding for chunk `{id}` is empty; embeddings must have at least one dimension")]
    EmptyEmbedding {
        /// Identifier of the chunk whose embedding was empty.
        id: String,
    },

    /// A backing store (network, serialization, …) failed. Unused by
    /// [`InMemoryVectorStore`]; provided for backends that talk to an external service.
    #[error("vector store backend `{backend}` failed: {message}")]
    Backend {
        /// Name of the backend that failed.
        backend: String,
        /// Human-readable description of the failure.
        message: String,
    },
}

/// Cosine similarity between two equal-length vectors, in `[-1.0, 1.0]`.
///
/// Higher is more similar: `1.0` for parallel vectors, `0.0` for orthogonal ones, and
/// `-1.0` for opposite ones. The metric is magnitude-invariant, so it compares
/// *direction* rather than length.
///
/// Returns `0.0` when either vector has zero magnitude (cosine is otherwise undefined)
/// rather than producing a `NaN`.
///
/// The two vectors must be the same length. [`VectorStore`] implementations enforce this
/// up front via [`VectorStoreError::DimensionMismatch`], so a mismatch here signals a
/// programmer error rather than bad data; it is caught by a `debug_assert` in debug and
/// test builds. In a release build a mismatch is not checked and only the overlapping
/// prefix is compared, which would silently skew the score — hence the assertion.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "cosine_similarity requires equal-length vectors"
    );

    // Accumulate in f64 so high-dimensional sums don't lose precision before the final
    // ratio is taken; the result is narrowed back to f32 to match the stored vectors.
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot / (norm_a.sqrt() * norm_b.sqrt())) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_score_one() {
        let v = [0.2, 0.5, 0.9, 0.1];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn orthogonal_vectors_score_zero() {
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn opposite_vectors_score_negative_one() {
        assert!((cosine_similarity(&[1.0, 1.0], &[-1.0, -1.0]) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn similarity_is_magnitude_invariant() {
        // Scaling a vector leaves its direction — and so its cosine similarity — unchanged.
        let a = [1.0, 2.0, 3.0];
        let b = [2.0, 4.0, 6.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn zero_vector_scores_zero_rather_than_nan() {
        let score = cosine_similarity(&[0.0, 0.0], &[1.0, 2.0]);
        assert_eq!(score, 0.0);
        assert!(!score.is_nan());
    }

    #[test]
    fn search_result_exposes_text_and_metadata() {
        let result = SearchResult {
            chunk: Chunk {
                id: "doc#0".into(),
                content: "hello world".into(),
                metadata: ChunkMetadata {
                    source: "doc".into(),
                    chunk_index: 0,
                    total_chunks: 1,
                    start_char: 0,
                    end_char: 11,
                    extra: Default::default(),
                },
            },
            score: 0.5,
        };

        assert_eq!(result.content(), "hello world");
        assert_eq!(result.metadata().source, "doc");
    }
}
