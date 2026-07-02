//! # penr-oz-ai-rag-service
//!
//! A modular document **ingestion pipeline** for a Retrieval-Augmented Generation
//! (RAG) service. The pipeline turns raw source files into metadata-rich [`Chunk`]s
//! and persists them to a pluggable storage backend.
//!
//! The flow is three decoupled stages, each behind a trait so it can be swapped or
//! extended independently:
//!
//! 1. **Load** — a [`Loader`] reads a file and normalizes it into a [`Document`]. New
//!    formats (PDF, HTML, Markdown, …) are added by implementing [`Loader`] and
//!    registering it with a [`LoaderRegistry`].
//! 2. **Chunk** — a [`Chunker`] splits a [`Document`] into [`Chunk`]s, attaching
//!    positional and provenance [`ChunkMetadata`].
//! 3. **Store** — a [`ChunkStore`] persists the chunks (in memory, as JSON Lines, or
//!    a future vector store).
//!
//! [`IngestionPipeline`] composes the three stages.
//!
//! Embedding chunks for retrieval is handled separately, behind the
//! [`EmbeddingProvider`] trait, so the embedding backend can be chosen independently of
//! how documents are loaded, chunked, and stored. Embedded chunks are then indexed and
//! retrieved through the [`VectorStore`] trait — top-k similarity search returns each
//! matching [`Chunk`] with its score — backed by an in-process [`InMemoryVectorStore`]
//! for development and tests.
//!
//! A [`Retriever`] composes those two abstractions into the read half of RAG: it
//! validates a query, embeds it, and searches the vector store, returning the top
//! matching chunks with their scores. It is the engine behind a `POST /retrieve`
//! endpoint, with [`RetrievalRequest`] / [`RetrievalResponse`] as the wire shapes.
//!
//! ## Example
//!
//! ```
//! use penr_oz_ai_rag_service::{Chunker, Document, FixedSizeChunker};
//!
//! let document = Document::new("notes.txt", "Retrieval augmented generation ingests documents.");
//! let chunker = FixedSizeChunker::new(24, 6).unwrap();
//! let chunks = chunker.chunk(&document).unwrap();
//!
//! assert!(!chunks.is_empty());
//! assert_eq!(chunks[0].metadata.source, "notes.txt");
//! ```

pub mod chunker;
pub mod document;
pub mod embedding;
pub mod error;
pub mod loader;
pub mod pipeline;
pub mod retrieval;
pub mod storage;
pub mod vector;

pub use chunker::{Chunker, FixedSizeChunker};
pub use document::{Chunk, ChunkMetadata, Document, Metadata};
pub use embedding::{EmbeddingError, EmbeddingProvider, MockEmbeddingProvider};
pub use error::{RagError, Result};
pub use loader::{Loader, LoaderRegistry, TextLoader};
pub use pipeline::{FileReport, IngestReport, IngestionPipeline, PipelineBuilder};
pub use retrieval::{
    RetrievalError, RetrievalRequest, RetrievalResponse, Retriever, DEFAULT_MAX_QUERY_CHARS,
    DEFAULT_TOP_K,
};
pub use storage::{ChunkStore, InMemoryStorage, JsonlStorage};
pub use vector::{
    cosine_similarity, EmbeddedChunk, Embedding, InMemoryVectorStore, SearchResult, VectorStore,
    VectorStoreError,
};
