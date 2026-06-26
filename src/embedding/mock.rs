//! An in-memory [`EmbeddingProvider`] for tests and local development.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use async_trait::async_trait;

use super::error::EmbeddingError;
use super::provider::EmbeddingProvider;

/// A deterministic, dependency-free embedding provider for use in tests.
///
/// `MockEmbeddingProvider` derives each vector from a hash of the input text, so
/// the same input always yields the same embedding and different inputs almost
/// always differ. The vectors are L2-normalised, which makes them well behaved
/// under cosine/dot-product similarity without reaching for a real model or the
/// network.
///
/// It can also be told to fail, which lets tests exercise the error-handling
/// paths of code that depends on [`EmbeddingProvider`].
///
/// # Example
///
/// ```
/// use rag_service::embedding::{EmbeddingProvider, MockEmbeddingProvider};
///
/// # async fn run() -> Result<(), rag_service::embedding::EmbeddingError> {
/// let provider = MockEmbeddingProvider::new(16);
/// let a = provider.embed_one("retrieval augmented generation").await?;
/// let b = provider.embed_one("retrieval augmented generation").await?;
/// assert_eq!(a, b); // deterministic
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct MockEmbeddingProvider {
    dimensions: usize,
    name: String,
    failure: Option<MockFailure>,
}

/// How a [`MockEmbeddingProvider`] should fail when configured to do so.
#[derive(Debug, Clone)]
enum MockFailure {
    Request(String),
    RateLimited(String),
}

impl MockEmbeddingProvider {
    /// Create a provider that produces normalised vectors of `dimensions`
    /// elements.
    ///
    /// # Panics
    ///
    /// Panics if `dimensions` is zero, since a zero-length embedding is never
    /// useful.
    pub fn new(dimensions: usize) -> Self {
        assert!(dimensions > 0, "dimensions must be greater than zero");
        Self {
            dimensions,
            name: "mock".to_string(),
            failure: None,
        }
    }

    /// Override the provider name reported by [`EmbeddingProvider::name`].
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Configure the provider to fail every [`embed`](EmbeddingProvider::embed)
    /// call with an [`EmbeddingError::Request`] carrying `message`.
    pub fn with_request_failure(mut self, message: impl Into<String>) -> Self {
        self.failure = Some(MockFailure::Request(message.into()));
        self
    }

    /// Configure the provider to fail every [`embed`](EmbeddingProvider::embed)
    /// call with an [`EmbeddingError::RateLimited`] carrying `message`.
    pub fn with_rate_limit(mut self, message: impl Into<String>) -> Self {
        self.failure = Some(MockFailure::RateLimited(message.into()));
        self
    }

    /// Produce the deterministic vector for a single input.
    fn embed_text(&self, text: &str) -> Vec<f32> {
        // Seed a simple PRNG-ish sequence from the input hash, then fill the
        // vector and L2-normalise it so similarity comparisons behave sensibly.
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let mut state = hasher.finish();

        let mut vector = Vec::with_capacity(self.dimensions);
        for _ in 0..self.dimensions {
            // xorshift64* keeps successive components decorrelated.
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
            // Map to the range [-1.0, 1.0).
            let unit = (bits >> 40) as f32 / (1u64 << 24) as f32;
            vector.push(unit * 2.0 - 1.0);
        }

        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vector {
                *v /= norm;
            }
        }
        vector
    }
}

#[async_trait]
impl EmbeddingProvider for MockEmbeddingProvider {
    async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if let Some(failure) = &self.failure {
            return Err(match failure {
                MockFailure::Request(msg) => EmbeddingError::Request(msg.clone()),
                MockFailure::RateLimited(msg) => EmbeddingError::RateLimited(msg.clone()),
            });
        }

        if inputs.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        let mut vectors = Vec::with_capacity(inputs.len());
        for (index, input) in inputs.iter().enumerate() {
            if input.is_empty() {
                return Err(EmbeddingError::EmptyInputItem { index });
            }
            vectors.push(self.embed_text(input));
        }
        Ok(vectors)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &str {
        &self.name
    }
}
