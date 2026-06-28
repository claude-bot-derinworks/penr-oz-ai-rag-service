//! Turning text into embedding vectors.
//!
//! Embedding is decoupled from the rest of the pipeline behind the
//! [`EmbeddingProvider`] trait: the service depends only on the trait, so the concrete
//! backend — a hosted API (OpenAI, Cohere, …), a local model, or the in-process
//! [`MockEmbeddingProvider`] used in tests — can be swapped without touching callers.
//!
//! Provider-specific code (HTTP clients, auth, request shaping) lives inside each
//! implementation; everything a caller needs is the trait and the
//! [`EmbeddingError`] it surfaces.
//!
//! ## Example
//!
//! ```
//! use penr_oz_ai_rag_service::{EmbeddingProvider, MockEmbeddingProvider};
//!
//! # async fn run() -> Result<(), penr_oz_ai_rag_service::EmbeddingError> {
//! let provider = MockEmbeddingProvider::new();
//! let vectors = provider.embed(&["hello", "world"]).await?;
//!
//! assert_eq!(vectors.len(), 2);
//! assert_eq!(vectors[0].len(), provider.dimensions());
//! # Ok(())
//! # }
//! ```

mod mock;

pub use mock::MockEmbeddingProvider;

use async_trait::async_trait;
use thiserror::Error;

/// A source of embedding vectors for text.
///
/// Implementations must be cheap to share (`Send + Sync`) so a single provider can be
/// used concurrently, and the trait is object-safe so providers can be stored behind a
/// `Box<dyn EmbeddingProvider>` and chosen at runtime.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a batch of `inputs`, returning one vector per input.
    ///
    /// The returned `Vec` has the same length as `inputs` and preserves order: the
    /// vector at index `i` is the embedding of `inputs[i]`. Each vector has
    /// [`dimensions`](EmbeddingProvider::dimensions) components. An empty `inputs`
    /// slice yields an empty result without contacting the backend.
    ///
    /// Inputs are taken as `&[&str]` so callers that already hold borrowed text — the
    /// common case when embedding chunk content — need not allocate owned `String`s.
    ///
    /// # Errors
    /// Returns [`EmbeddingError`] when the provider rejects an input or the backend
    /// fails to produce embeddings.
    async fn embed(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError>;

    /// The number of components in every vector this provider produces.
    fn dimensions(&self) -> usize;
}

/// The set of errors an [`EmbeddingProvider`] can produce.
///
/// Kept separate from [`RagError`](crate::error::RagError) so embedding concerns stay
/// isolated from the ingestion pipeline's error surface.
#[derive(Debug, Error)]
pub enum EmbeddingError {
    /// The backend rejected the request or failed to return embeddings.
    #[error("embedding provider `{provider}` failed: {message}")]
    Provider {
        /// Name of the provider that failed.
        provider: String,
        /// Human-readable description of the failure.
        message: String,
    },

    /// An input in the batch was empty, which providers cannot embed.
    #[error("embedding input at index {index} is empty")]
    EmptyInput {
        /// Position of the empty input within the batch.
        index: usize,
    },
}
