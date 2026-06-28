//! Integration coverage for the embedding abstraction, exercising it through the public
//! API the way a downstream caller would.

use penr_oz_ai_rag_service::{EmbeddingError, EmbeddingProvider, MockEmbeddingProvider};

/// Pick a provider at runtime and embed through the trait object, the way the service
/// would once a backend is configured.
fn provider_for(name: &str) -> Box<dyn EmbeddingProvider> {
    match name {
        "mock" => Box::new(MockEmbeddingProvider::new().with_dimensions(16)),
        _ => Box::new(MockEmbeddingProvider::failing("unknown provider")),
    }
}

#[tokio::test]
async fn embeds_chunk_text_through_a_trait_object() {
    let provider = provider_for("mock");
    let inputs = ["Retrieval augmented generation", "ingests documents"];

    let vectors = provider.embed(&inputs).await.expect("embedding succeeds");

    assert_eq!(vectors.len(), inputs.len());
    assert!(vectors.iter().all(|v| v.len() == provider.dimensions()));
}

#[tokio::test]
async fn provider_errors_are_reported_clearly() {
    let provider = provider_for("does-not-exist");

    match provider.embed(&["anything"]).await {
        Err(EmbeddingError::Provider { provider, message }) => {
            assert_eq!(provider, "mock");
            assert_eq!(message, "unknown provider");
        }
        other => panic!("expected a provider error, got {other:?}"),
    }
}
