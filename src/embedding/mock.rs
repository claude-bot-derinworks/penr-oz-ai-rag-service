//! A deterministic, dependency-free [`EmbeddingProvider`] for tests and examples.

use async_trait::async_trait;

use super::{EmbeddingError, EmbeddingProvider};

/// An [`EmbeddingProvider`] that fabricates embeddings locally, without any network or
/// model.
///
/// The vectors are **deterministic**: the same input always maps to the same vector,
/// and distinct inputs map to distinct vectors. That makes it a stable stand-in for a
/// real provider in unit and integration tests. A mock can also be put into a failure
/// mode with [`failing`](MockEmbeddingProvider::failing) to exercise error handling.
#[derive(Debug, Clone)]
pub struct MockEmbeddingProvider {
    dimensions: usize,
    failure: Option<String>,
}

impl MockEmbeddingProvider {
    /// Name reported in [`EmbeddingError::Provider`] errors.
    const NAME: &'static str = "mock";

    /// Default vector dimensionality.
    pub const DEFAULT_DIMENSIONS: usize = 8;

    /// Create a mock producing [`DEFAULT_DIMENSIONS`](Self::DEFAULT_DIMENSIONS)-length
    /// vectors.
    pub fn new() -> Self {
        Self {
            dimensions: Self::DEFAULT_DIMENSIONS,
            failure: None,
        }
    }

    /// Set the dimensionality of produced vectors (builder style).
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = dimensions;
        self
    }

    /// Create a mock whose [`embed`](EmbeddingProvider::embed) always fails with an
    /// [`EmbeddingError::Provider`] carrying `message`, for testing error paths.
    pub fn failing(message: impl Into<String>) -> Self {
        Self {
            dimensions: Self::DEFAULT_DIMENSIONS,
            failure: Some(message.into()),
        }
    }

    /// Produce the deterministic embedding for a single input.
    ///
    /// An FNV-1a hash of the input seeds an `xorshift64` generator; each dimension is
    /// the next generator output scaled into `[0, 1)`. Identical inputs therefore yield
    /// identical vectors, and differing inputs almost always diverge.
    fn embed_one(&self, input: &str) -> Vec<f32> {
        // `| 1` guarantees a non-zero seed, which `xorshift64` requires.
        let mut state = fnv1a(input.as_bytes()) | 1;
        (0..self.dimensions)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % 1_000_000) as f32 / 1_000_000.0
            })
            .collect()
    }
}

impl Default for MockEmbeddingProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl EmbeddingProvider for MockEmbeddingProvider {
    async fn embed(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        // Per the trait contract, an empty batch yields an empty result without
        // contacting the backend — so this short-circuits before the failure mode.
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        if let Some(message) = &self.failure {
            return Err(EmbeddingError::Provider {
                provider: Self::NAME.to_string(),
                message: message.clone(),
            });
        }

        let mut embeddings = Vec::with_capacity(inputs.len());
        for (index, &input) in inputs.iter().enumerate() {
            if input.is_empty() {
                return Err(EmbeddingError::EmptyInput { index });
            }
            embeddings.push(self.embed_one(input));
        }
        Ok(embeddings)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

/// FNV-1a 64-bit hash, used to seed the per-input vector generator.
fn fnv1a(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET_BASIS;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn embeds_a_batch_preserving_length_and_dimensions() {
        let provider = MockEmbeddingProvider::new().with_dimensions(4);
        let inputs = ["alpha", "beta", "gamma"];

        let vectors = provider.embed(&inputs).await.unwrap();

        assert_eq!(vectors.len(), inputs.len());
        assert!(vectors.iter().all(|v| v.len() == 4));
        assert_eq!(provider.dimensions(), 4);
    }

    #[tokio::test]
    async fn is_deterministic_and_input_dependent() {
        let provider = MockEmbeddingProvider::new();
        let inputs = ["same", "same", "different"];

        let vectors = provider.embed(&inputs).await.unwrap();

        // Identical inputs produce identical vectors...
        assert_eq!(vectors[0], vectors[1]);
        // ...and distinct inputs produce distinct vectors.
        assert_ne!(vectors[0], vectors[2]);
    }

    #[tokio::test]
    async fn empty_batch_yields_empty_result() {
        let provider = MockEmbeddingProvider::new();
        assert!(provider.embed(&[]).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn empty_input_is_rejected_with_its_index() {
        let provider = MockEmbeddingProvider::new();
        let inputs = ["ok", ""];

        match provider.embed(&inputs).await {
            Err(EmbeddingError::EmptyInput { index }) => assert_eq!(index, 1),
            other => panic!("expected EmptyInput error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn failing_provider_surfaces_a_provider_error() {
        let provider = MockEmbeddingProvider::failing("rate limited");

        match provider.embed(&["hi"]).await {
            Err(EmbeddingError::Provider { provider, message }) => {
                assert_eq!(provider, "mock");
                assert_eq!(message, "rate limited");
            }
            other => panic!("expected Provider error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn empty_batch_short_circuits_even_when_failing() {
        // The empty-batch contract takes precedence over the failure mode: an empty
        // slice never "contacts the backend", so it cannot fail.
        let provider = MockEmbeddingProvider::failing("should not be reached");
        assert!(provider.embed(&[]).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn usable_as_a_trait_object() {
        let provider: Box<dyn EmbeddingProvider> = Box::new(MockEmbeddingProvider::new());
        let vectors = provider.embed(&["boxed"]).await.unwrap();
        assert_eq!(vectors[0].len(), provider.dimensions());
    }
}
