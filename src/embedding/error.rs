//! Errors that an [`EmbeddingProvider`](super::EmbeddingProvider) can surface.

use thiserror::Error;

/// Errors returned by an embedding provider.
///
/// The variants are intentionally provider-neutral: a concrete backend maps its
/// own failure modes (HTTP errors, rate limits, malformed payloads, …) onto
/// these so callers can react without knowing which provider is in use. Each
/// variant carries a human-readable message describing what went wrong.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum EmbeddingError {
    /// The caller supplied no inputs to embed.
    #[error("no inputs were provided to embed")]
    EmptyInput,

    /// An individual input was empty and the provider cannot embed it.
    #[error("input at index {index} is empty")]
    EmptyInputItem {
        /// Position of the offending input within the batch.
        index: usize,
    },

    /// The provider is misconfigured (missing API key, invalid model, …).
    #[error("embedding provider configuration error: {0}")]
    Configuration(String),

    /// Authenticating with the provider failed.
    #[error("embedding provider authentication failed: {0}")]
    Authentication(String),

    /// The request to the provider failed to complete (network, timeout, …).
    #[error("embedding provider request failed: {0}")]
    Request(String),

    /// The provider responded, but the payload could not be understood.
    #[error("embedding provider returned an invalid response: {0}")]
    InvalidResponse(String),

    /// The provider rejected the request because a rate limit was exceeded.
    #[error("embedding provider rate limit exceeded: {0}")]
    RateLimited(String),
}
