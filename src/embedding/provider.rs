//! The [`EmbeddingProvider`] trait.

use async_trait::async_trait;

use super::error::EmbeddingError;

/// A backend capable of turning text into embedding vectors.
///
/// Implementors isolate all provider-specific concerns (transport,
/// authentication, request/response shapes) behind this interface so the rest of
/// the service stays vendor-neutral. Providers are expected to be cheap to share
/// (`Send + Sync`) so a single instance can be reused across concurrent
/// requests.
///
/// # Batching
///
/// [`embed`](EmbeddingProvider::embed) accepts a batch of inputs and returns one
/// vector per input, in the same order. Batching is the primary API because most
/// providers charge per request and embedding many texts in one call is far more
/// efficient than issuing one request per text. [`embed_one`] is provided as a
/// convenience for the single-input case.
///
/// [`embed_one`]: EmbeddingProvider::embed_one
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a batch of inputs.
    ///
    /// Returns one vector per input, preserving the order of `inputs`. Every
    /// returned vector has [`dimensions`](EmbeddingProvider::dimensions)
    /// elements.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError::EmptyInput`] when `inputs` is empty, and any
    /// provider-specific failure (network, authentication, malformed response,
    /// …) mapped onto an [`EmbeddingError`] variant.
    async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError>;

    /// The dimensionality of the vectors this provider produces.
    ///
    /// Downstream components (vector stores, similarity search) need a fixed,
    /// known dimension to allocate storage and validate inputs.
    fn dimensions(&self) -> usize;

    /// A short, stable identifier for the provider/model, useful for logging and
    /// metrics.
    fn name(&self) -> &str;

    /// Embed a single input.
    ///
    /// A convenience wrapper around [`embed`](EmbeddingProvider::embed) for the
    /// common single-text case. The default implementation forwards to `embed`
    /// and unwraps the sole result.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`embed`](EmbeddingProvider::embed), and returns
    /// [`EmbeddingError::InvalidResponse`] if the provider does not return
    /// exactly one vector.
    async fn embed_one(&self, input: &str) -> Result<Vec<f32>, EmbeddingError> {
        let mut vectors = self.embed(std::slice::from_ref(&input.to_string())).await?;
        if vectors.len() != 1 {
            return Err(EmbeddingError::InvalidResponse(format!(
                "expected exactly 1 embedding, got {}",
                vectors.len()
            )));
        }
        Ok(vectors.pop().expect("length checked above"))
    }
}
