# penr-oz-ai-rag-service
Implementation of retrieval augmented generation service for AI

## Embedding providers

The service is provider-agnostic: it depends on the
[`EmbeddingProvider`](src/embedding/provider.rs) trait rather than any single
vendor. A provider turns a batch of texts into embedding vectors and reports a
clear [`EmbeddingError`](src/embedding/error.rs) on failure. Provider-specific
code (transport, authentication, request shapes) stays isolated behind the
trait.

```rust
use rag_service::embedding::{EmbeddingProvider, MockEmbeddingProvider};

# async fn example() -> Result<(), rag_service::embedding::EmbeddingError> {
let provider = MockEmbeddingProvider::new(8);
let vectors = provider
    .embed(&["hello world".to_string(), "goodbye".to_string()])
    .await?;
assert_eq!(vectors.len(), 2);
# Ok(())
# }
```

[`MockEmbeddingProvider`](src/embedding/mock.rs) is a deterministic,
dependency-free implementation for tests and local development; it can also be
configured to fail so error-handling paths can be exercised.

## Development

```sh
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```
