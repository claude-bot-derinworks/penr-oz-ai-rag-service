//! A deterministic, dependency-free [`LlmProvider`] for tests and examples.

use async_trait::async_trait;

use super::{LlmError, LlmProvider};

/// An [`LlmProvider`] that fabricates completions locally, without any network or model.
///
/// By default the mock **echoes the prompt back verbatim**, which is deterministic and
/// input-dependent — the same properties [`MockEmbeddingProvider`](crate::embedding::MockEmbeddingProvider)
/// has — and lets tests assert on exactly what the model was shown (the retrieved
/// context, the question, …). A canned reply can be set with
/// [`with_reply`](MockLlmProvider::with_reply), and the mock can be put into a failure
/// mode with [`failing`](MockLlmProvider::failing) to exercise error handling.
#[derive(Debug, Clone, Default)]
pub struct MockLlmProvider {
    reply: Option<String>,
    failure: Option<String>,
}

impl MockLlmProvider {
    /// Name reported in [`LlmError::Provider`] errors.
    const NAME: &'static str = "mock";

    /// Create a mock that echoes each prompt back as its completion.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a mock that answers every prompt with the fixed `reply`.
    pub fn with_reply(reply: impl Into<String>) -> Self {
        Self {
            reply: Some(reply.into()),
            failure: None,
        }
    }

    /// Create a mock whose [`generate`](LlmProvider::generate) always fails with an
    /// [`LlmError::Provider`] carrying `message`, for testing error paths.
    pub fn failing(message: impl Into<String>) -> Self {
        Self {
            reply: None,
            failure: Some(message.into()),
        }
    }
}

#[async_trait]
impl LlmProvider for MockLlmProvider {
    async fn generate(&self, prompt: &str) -> Result<String, LlmError> {
        if let Some(message) = &self.failure {
            return Err(LlmError::Provider {
                provider: Self::NAME.to_string(),
                message: message.clone(),
            });
        }

        if prompt.trim().is_empty() {
            return Err(LlmError::EmptyPrompt);
        }

        Ok(match &self.reply {
            Some(reply) => reply.clone(),
            None => prompt.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn echoes_the_prompt_by_default() {
        let provider = MockLlmProvider::new();
        let answer = provider.generate("What is RAG?").await.unwrap();
        assert_eq!(answer, "What is RAG?");
    }

    #[tokio::test]
    async fn canned_reply_answers_every_prompt() {
        let provider = MockLlmProvider::with_reply("42");
        assert_eq!(provider.generate("first").await.unwrap(), "42");
        assert_eq!(provider.generate("second").await.unwrap(), "42");
    }

    #[tokio::test]
    async fn empty_or_whitespace_prompt_is_rejected() {
        let provider = MockLlmProvider::new();
        assert!(matches!(
            provider.generate("").await,
            Err(LlmError::EmptyPrompt)
        ));
        assert!(matches!(
            provider.generate("  \t\n").await,
            Err(LlmError::EmptyPrompt)
        ));
    }

    #[tokio::test]
    async fn failing_provider_surfaces_a_provider_error() {
        let provider = MockLlmProvider::failing("rate limited");

        match provider.generate("hi").await {
            Err(LlmError::Provider { provider, message }) => {
                assert_eq!(provider, "mock");
                assert_eq!(message, "rate limited");
            }
            other => panic!("expected Provider error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn usable_as_a_trait_object() {
        let provider: Box<dyn LlmProvider> = Box::new(MockLlmProvider::with_reply("boxed"));
        assert_eq!(provider.generate("prompt").await.unwrap(), "boxed");
    }
}
