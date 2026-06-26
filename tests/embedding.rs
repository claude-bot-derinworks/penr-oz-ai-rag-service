//! Integration tests for the embedding provider abstraction.
//!
//! These exercise the public API through the [`EmbeddingProvider`] trait, the
//! same way the rest of the service would, including dynamic dispatch via
//! `dyn EmbeddingProvider`.

use rag_service::embedding::{EmbeddingError, EmbeddingProvider, MockEmbeddingProvider};

#[tokio::test]
async fn embeds_a_batch_preserving_order_and_dimension() {
    let provider = MockEmbeddingProvider::new(12);
    let inputs = vec![
        "the quick brown fox".to_string(),
        "jumps over the lazy dog".to_string(),
        "retrieval augmented generation".to_string(),
    ];

    let vectors = provider
        .embed(&inputs)
        .await
        .expect("embedding should succeed");

    assert_eq!(vectors.len(), inputs.len());
    for vector in &vectors {
        assert_eq!(vector.len(), provider.dimensions());
        // Vectors are L2-normalised, so the norm should be ~1.
        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "expected unit norm, got {norm}");
    }
}

#[tokio::test]
async fn embeddings_are_deterministic() {
    let provider = MockEmbeddingProvider::new(8);
    let first = provider.embed_one("hello").await.unwrap();
    let second = provider.embed_one("hello").await.unwrap();
    assert_eq!(first, second);
}

#[tokio::test]
async fn distinct_inputs_produce_distinct_vectors() {
    let provider = MockEmbeddingProvider::new(8);
    let a = provider.embed_one("alpha").await.unwrap();
    let b = provider.embed_one("beta").await.unwrap();
    assert_ne!(a, b);
}

#[tokio::test]
async fn empty_batch_is_rejected() {
    let provider = MockEmbeddingProvider::new(8);
    let err = provider.embed(&[]).await.unwrap_err();
    assert!(matches!(err, EmbeddingError::EmptyInput));
}

#[tokio::test]
async fn empty_item_is_rejected_with_index() {
    let provider = MockEmbeddingProvider::new(8);
    let inputs = vec!["ok".to_string(), String::new()];
    let err = provider.embed(&inputs).await.unwrap_err();
    assert!(matches!(err, EmbeddingError::EmptyInputItem { index: 1 }));
}

#[tokio::test]
async fn configured_failures_surface_as_errors() {
    let provider = MockEmbeddingProvider::new(8).with_request_failure("boom");
    let err = provider.embed_one("x").await.unwrap_err();
    assert!(matches!(err, EmbeddingError::Request(msg) if msg == "boom"));

    let provider = MockEmbeddingProvider::new(8).with_rate_limit("slow down");
    let err = provider.embed_one("x").await.unwrap_err();
    assert!(matches!(err, EmbeddingError::RateLimited(msg) if msg == "slow down"));
}

#[tokio::test]
async fn usable_through_dynamic_dispatch() {
    // The whole point of the abstraction: the service can hold any provider
    // behind a trait object without knowing the concrete type.
    let provider: Box<dyn EmbeddingProvider> =
        Box::new(MockEmbeddingProvider::new(4).with_name("test-model"));
    assert_eq!(provider.name(), "test-model");

    let vectors = provider.embed(&["text".to_string()]).await.unwrap();
    assert_eq!(vectors.len(), 1);
    assert_eq!(vectors[0].len(), 4);
}
