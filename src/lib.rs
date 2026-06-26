//! Core library for the retrieval augmented generation (RAG) service.
//!
//! The crate is organised around small, provider-agnostic abstractions so that
//! the service never depends on a single vendor. The [`embedding`] module is the
//! first of these: it defines the [`embedding::EmbeddingProvider`] trait that any
//! concrete embedding backend (OpenAI, a local model, a fake used in tests, …)
//! can implement.

pub mod embedding;

pub use embedding::{EmbeddingError, EmbeddingProvider, MockEmbeddingProvider};
