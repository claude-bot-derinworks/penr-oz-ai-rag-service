//! Generating text with a language model.
//!
//! Answer generation is decoupled from the rest of the service behind the
//! [`LlmProvider`] trait — mirroring how [`EmbeddingProvider`](crate::embedding::EmbeddingProvider)
//! decouples embedding. The service depends only on the trait, so the concrete backend —
//! a hosted API (OpenAI, Anthropic, …), a local model, or the in-process
//! [`MockLlmProvider`] used in tests — can be swapped without touching callers.
//!
//! Provider-specific code (HTTP clients, auth, request shaping) lives inside each
//! implementation; everything a caller needs is the trait and the [`LlmError`] it
//! surfaces.
//!
//! ## Example
//!
//! ```
//! use penr_oz_ai_rag_service::{LlmProvider, MockLlmProvider};
//!
//! # async fn run() -> Result<(), penr_oz_ai_rag_service::LlmError> {
//! let provider = MockLlmProvider::with_reply("Chunking splits documents.");
//! let answer = provider.generate("How does chunking work?").await?;
//!
//! assert_eq!(answer, "Chunking splits documents.");
//! # Ok(())
//! # }
//! ```

mod mock;

pub use mock::MockLlmProvider;

use async_trait::async_trait;
use thiserror::Error;

/// A language model that completes a prompt with generated text.
///
/// Implementations must be cheap to share (`Send + Sync`) so a single provider can be
/// used concurrently, and the trait is object-safe so providers can be stored behind a
/// `Box<dyn LlmProvider>` and chosen at runtime.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Generate a completion for `prompt`.
    ///
    /// The prompt is the full text shown to the model — for RAG, the retrieved context
    /// and the user's question already assembled (see
    /// [`build_prompt`](crate::generation::build_prompt)). The returned `String` is the
    /// model's answer, unwrapped from any provider-specific envelope.
    ///
    /// # Errors
    /// Returns [`LlmError::EmptyPrompt`] for an empty (or whitespace-only) prompt, and
    /// [`LlmError::Provider`] when the backend rejects the request or fails to generate.
    async fn generate(&self, prompt: &str) -> Result<String, LlmError>;
}

/// The set of errors an [`LlmProvider`] can produce.
///
/// Kept separate from [`RagError`](crate::error::RagError) — mirroring
/// [`EmbeddingError`](crate::embedding::EmbeddingError) — so generation concerns stay
/// isolated from the ingestion pipeline's error surface.
#[derive(Debug, Error)]
pub enum LlmError {
    /// The backend rejected the request or failed to generate a completion.
    #[error("llm provider `{provider}` failed: {message}")]
    Provider {
        /// Name of the provider that failed.
        provider: String,
        /// Human-readable description of the failure.
        message: String,
    },

    /// The prompt was empty or contained only whitespace, which providers cannot
    /// complete.
    #[error("prompt must not be empty")]
    EmptyPrompt,
}
