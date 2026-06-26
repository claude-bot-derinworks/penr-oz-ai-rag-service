//! Provider-agnostic embedding abstraction.
//!
//! Embeddings turn text into dense vectors that the retrieval layer can compare
//! for semantic similarity. Different deployments use different embedding
//! backends, so the service talks to them exclusively through the
//! [`EmbeddingProvider`] trait. Provider-specific code (HTTP clients, request
//! shapes, authentication, …) lives behind that trait and never leaks into the
//! rest of the service.
//!
//! # Example
//!
//! ```
//! use rag_service::embedding::{EmbeddingProvider, MockEmbeddingProvider};
//!
//! # async fn run() -> Result<(), rag_service::embedding::EmbeddingError> {
//! let provider = MockEmbeddingProvider::new(8);
//! let vectors = provider
//!     .embed(&["hello world".to_string(), "goodbye".to_string()])
//!     .await?;
//!
//! assert_eq!(vectors.len(), 2);
//! assert_eq!(vectors[0].len(), 8);
//! # Ok(())
//! # }
//! ```

mod error;
mod mock;
mod provider;

pub use error::EmbeddingError;
pub use mock::MockEmbeddingProvider;
pub use provider::EmbeddingProvider;
