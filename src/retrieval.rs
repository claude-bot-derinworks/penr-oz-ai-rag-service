//! Query-time retrieval: turn a user query into the most relevant chunks.
//!
//! This is the read half of RAG and the engine behind a `POST /retrieve` endpoint. A
//! [`Retriever`] composes the two abstractions the rest of the crate already provides:
//!
//! 1. an [`EmbeddingProvider`](crate::embedding::EmbeddingProvider) embeds the query into
//!    the same vector space as the indexed chunks, and
//! 2. a [`VectorStore`](crate::vector::VectorStore) returns the top-k chunks whose vectors
//!    are most similar.
//!
//! [`Retriever::retrieve`] ties them together and guards the query first: empty (or
//! whitespace-only) and oversized queries are rejected with a [`RetrievalError`] before
//! any embedding or search work is done. Each hit comes back as a
//! [`SearchResult`](crate::vector::SearchResult), carrying the matching chunk's text and
//! metadata alongside its similarity score — exactly what a retrieval endpoint returns.
//!
//! The serde-friendly [`RetrievalRequest`] / [`RetrievalResponse`] pair is the wire shape
//! of that endpoint: deserialize the `POST` body into a [`RetrievalRequest`], call
//! [`Retriever::handle`], and serialize the [`RetrievalResponse`] back. No web framework
//! is pulled in here — like the embedding and vector-store layers, retrieval is kept a
//! plain, runtime-agnostic library so the binary that hosts the endpoint can choose its
//! own HTTP stack.
//!
//! ## Example
//!
//! ```
//! use penr_oz_ai_rag_service::{
//!     Chunk, ChunkMetadata, InMemoryVectorStore, MockEmbeddingProvider, Retriever,
//! };
//!
//! # async fn run() -> Result<(), penr_oz_ai_rag_service::RetrievalError> {
//! # fn chunk(id: &str, content: &str) -> Chunk {
//! #     Chunk {
//! #         id: id.to_string(),
//! #         content: content.to_string(),
//! #         metadata: ChunkMetadata {
//! #             source: "corpus".into(), chunk_index: 0, total_chunks: 1,
//! #             start_char: 0, end_char: 0, extra: Default::default(),
//! #         },
//! #     }
//! # }
//! let retriever = Retriever::new(MockEmbeddingProvider::new(), InMemoryVectorStore::new());
//!
//! // Index a small corpus, then retrieve the chunks most relevant to a query.
//! retriever
//!     .index(vec![
//!         chunk("c0", "retrieval augmented generation"),
//!         chunk("c1", "cosine similarity vector search"),
//!     ])
//!     .await?;
//!
//! let hits = retriever.retrieve("cosine similarity vector search", 1).await?;
//! assert_eq!(hits[0].chunk.id, "c1");
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::document::Chunk;
use crate::embedding::{EmbeddingError, EmbeddingProvider};
use crate::vector::{EmbeddedChunk, SearchResult, VectorStore, VectorStoreError};

/// Number of results returned when a [`RetrievalRequest`] omits `top_k`.
pub const DEFAULT_TOP_K: usize = 5;

/// Default ceiling, in characters, on a query's length.
///
/// Queries longer than this are rejected with [`RetrievalError::QueryTooLong`] before
/// they are embedded, guarding the embedding backend against pathologically large inputs.
/// It is a coarse, character-based guard (not a token count); tune it for a specific
/// embedding model with [`Retriever::with_max_query_chars`].
pub const DEFAULT_MAX_QUERY_CHARS: usize = 8_192;

/// The body of a `POST /retrieve` request: a user query and how many results to return.
///
/// `top_k` defaults to [`DEFAULT_TOP_K`] when omitted, so a minimal request is just
/// `{"query": "..."}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetrievalRequest {
    /// The user's query text.
    pub query: String,
    /// Maximum number of chunks to return, most relevant first.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

fn default_top_k() -> usize {
    DEFAULT_TOP_K
}

impl RetrievalRequest {
    /// Build a request for `query` returning up to `top_k` results.
    pub fn new(query: impl Into<String>, top_k: usize) -> Self {
        Self {
            query: query.into(),
            top_k,
        }
    }
}

/// The body of a `POST /retrieve` response: the matching chunks, most relevant first.
///
/// Each [`SearchResult`] carries the chunk's text and metadata together with its
/// similarity score, so the response is self-contained — no second lookup needed to
/// render it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalResponse {
    /// The retrieved chunks with their similarity scores, ordered most similar first.
    pub results: Vec<SearchResult>,
}

/// Retrieves the chunks most relevant to a query.
///
/// A `Retriever` owns an [`EmbeddingProvider`] and a [`VectorStore`] and wires them into
/// the query path: validate, embed, search. Both backends are generic so the concrete
/// embedding model and vector-store backend are chosen by the caller and cost nothing at
/// runtime; because every method takes `&self`, a `Retriever` can be shared (e.g. behind
/// an `Arc`) and queried concurrently, the way a request handler would.
pub struct Retriever<E, V> {
    embedder: E,
    store: V,
    max_query_chars: usize,
}

impl<E, V> Retriever<E, V>
where
    E: EmbeddingProvider,
    V: VectorStore,
{
    /// Create a retriever over `embedder` and `store`, using [`DEFAULT_MAX_QUERY_CHARS`]
    /// as the query-length ceiling.
    pub fn new(embedder: E, store: V) -> Self {
        Self {
            embedder,
            store,
            max_query_chars: DEFAULT_MAX_QUERY_CHARS,
        }
    }

    /// Override the maximum query length, in characters (builder style).
    pub fn with_max_query_chars(mut self, max_query_chars: usize) -> Self {
        self.max_query_chars = max_query_chars;
        self
    }

    /// The maximum query length, in characters, this retriever accepts.
    pub fn max_query_chars(&self) -> usize {
        self.max_query_chars
    }

    /// Borrow the embedding provider.
    pub fn embedder(&self) -> &E {
        &self.embedder
    }

    /// Borrow the vector store.
    pub fn store(&self) -> &V {
        &self.store
    }

    /// Embed `chunks` and index them so they become retrievable, returning how many were
    /// indexed.
    ///
    /// This is the write half that populates the store a later [`retrieve`](Self::retrieve)
    /// searches: each chunk's text is embedded with this retriever's provider and inserted
    /// alongside the chunk, so dimensions line up by construction. Chunks are taken by
    /// value — they are moved into the index rather than cloned. An empty `Vec` is a
    /// no-op.
    ///
    /// # Errors
    /// Returns [`RetrievalError::Embedding`] if the provider fails (e.g. a chunk has empty
    /// content), [`RetrievalError::EmbeddingCountMismatch`] if the provider returns a
    /// different number of vectors than chunks it was given, and
    /// [`RetrievalError::VectorStore`] if the store rejects the batch.
    pub async fn index(&self, chunks: Vec<Chunk>) -> Result<usize, RetrievalError> {
        if chunks.is_empty() {
            return Ok(0);
        }

        let texts: Vec<&str> = chunks.iter().map(|chunk| chunk.content.as_str()).collect();
        let embeddings = self.embedder.embed(&texts).await?;

        // `zip` would silently drop chunks (or pair them with the wrong vectors) if a
        // misbehaving provider returned the wrong number of embeddings, so reject that
        // before anything reaches the store.
        if embeddings.len() != chunks.len() {
            return Err(RetrievalError::EmbeddingCountMismatch {
                expected: chunks.len(),
                actual: embeddings.len(),
            });
        }

        let items: Vec<EmbeddedChunk> = chunks
            .into_iter()
            .zip(embeddings)
            .map(|(chunk, embedding)| EmbeddedChunk::new(chunk, embedding))
            .collect();

        self.store.insert(&items).await?;
        Ok(items.len())
    }

    /// Retrieve up to `top_k` chunks most relevant to `query`, most similar first.
    ///
    /// The query is validated, embedded, and used to search the vector store. The result
    /// is the store's ranked [`SearchResult`]s — each chunk with its similarity score.
    /// A `top_k` of `0`, or a search against an empty index, yields an empty `Vec`; the
    /// `top_k == 0` case returns right after validation, without contacting the embedding
    /// backend for a result that is empty by construction.
    ///
    /// # Errors
    /// - [`RetrievalError::EmptyQuery`] if `query` is empty or only whitespace.
    /// - [`RetrievalError::QueryTooLong`] if `query` exceeds
    ///   [`max_query_chars`](Self::max_query_chars).
    /// - [`RetrievalError::Embedding`] if embedding the query fails.
    /// - [`RetrievalError::EmbeddingCountMismatch`] if the provider returns anything but
    ///   exactly one vector for the query.
    /// - [`RetrievalError::VectorStore`] if the search fails (e.g. a dimension mismatch
    ///   between the query embedding and the indexed vectors).
    pub async fn retrieve(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, RetrievalError> {
        if query.trim().is_empty() {
            return Err(RetrievalError::EmptyQuery);
        }
        let chars = query.chars().count();
        if chars > self.max_query_chars {
            return Err(RetrievalError::QueryTooLong {
                chars,
                max: self.max_query_chars,
            });
        }

        // Zero results are empty by construction, so don't spend an embedding-backend
        // call (cost, rate limits) computing a vector that would never be searched.
        // This sits after validation so invalid queries still fail regardless of `top_k`.
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let embeddings = self.embedder.embed(&[query]).await?;
        if embeddings.len() != 1 {
            return Err(RetrievalError::EmbeddingCountMismatch {
                expected: 1,
                actual: embeddings.len(),
            });
        }
        let query_vector = embeddings
            .into_iter()
            .next()
            .expect("length was just checked to be 1");

        let results = self.store.search(&query_vector, top_k).await?;
        Ok(results)
    }

    /// Handle a [`RetrievalRequest`], returning a [`RetrievalResponse`].
    ///
    /// This is the request/response adapter a `POST /retrieve` handler calls: it maps the
    /// request onto [`retrieve`](Self::retrieve) and wraps the hits in a response.
    pub async fn handle(
        &self,
        request: &RetrievalRequest,
    ) -> Result<RetrievalResponse, RetrievalError> {
        let results = self.retrieve(&request.query, request.top_k).await?;
        Ok(RetrievalResponse { results })
    }
}

/// The set of errors retrieval can produce.
///
/// Validation failures ([`EmptyQuery`](RetrievalError::EmptyQuery),
/// [`QueryTooLong`](RetrievalError::QueryTooLong)) are distinguished from backend failures
/// so a caller — or an HTTP layer — can map the former to a `400 Bad Request` and the
/// latter to a `5xx`. The embedding and vector-store errors are carried through unchanged
/// rather than flattened, preserving their detail.
#[derive(Debug, Error)]
pub enum RetrievalError {
    /// The query was empty or contained only whitespace.
    #[error("query must not be empty")]
    EmptyQuery,

    /// The query exceeded the configured maximum length.
    #[error("query is too long: {chars} characters exceeds the maximum of {max}")]
    QueryTooLong {
        /// The query's length in characters.
        chars: usize,
        /// The maximum number of characters allowed.
        max: usize,
    },

    /// Embedding the query failed.
    #[error("failed to embed the query: {0}")]
    Embedding(#[from] EmbeddingError),

    /// Searching the vector store failed.
    #[error("vector search failed: {0}")]
    VectorStore(#[from] VectorStoreError),

    /// The embedding provider returned a different number of vectors than inputs it was
    /// given. A correct provider never does this; it signals a misbehaving backend, and
    /// pairing chunks with the wrong vectors would silently corrupt the index.
    #[error("the embedding provider returned {actual} vector(s) for {expected} input(s)")]
    EmbeddingCountMismatch {
        /// How many vectors were expected (one per input).
        expected: usize,
        /// How many vectors the provider actually returned.
        actual: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{ChunkMetadata, Metadata};
    use crate::embedding::MockEmbeddingProvider;
    use crate::vector::InMemoryVectorStore;

    fn chunk(id: &str, content: &str) -> Chunk {
        Chunk {
            id: id.to_string(),
            content: content.to_string(),
            metadata: ChunkMetadata {
                source: "corpus".to_string(),
                chunk_index: 0,
                total_chunks: 1,
                start_char: 0,
                end_char: content.chars().count(),
                extra: Metadata::new(),
            },
        }
    }

    fn retriever() -> Retriever<MockEmbeddingProvider, InMemoryVectorStore> {
        Retriever::new(MockEmbeddingProvider::new(), InMemoryVectorStore::new())
    }

    #[tokio::test]
    async fn rejects_empty_or_whitespace_query() {
        let retriever = retriever();
        assert!(matches!(
            retriever.retrieve("", 5).await,
            Err(RetrievalError::EmptyQuery)
        ));
        assert!(matches!(
            retriever.retrieve("   \t\n", 5).await,
            Err(RetrievalError::EmptyQuery)
        ));
    }

    #[tokio::test]
    async fn rejects_oversized_query_by_character_count() {
        // A four-character ceiling; the multi-byte query is seven *characters* (not the
        // nine bytes its UTF-8 encoding occupies).
        let retriever = retriever().with_max_query_chars(4);
        match retriever.retrieve("naïveté", 5).await {
            Err(RetrievalError::QueryTooLong { chars, max }) => {
                assert_eq!(chars, 7);
                assert_eq!(max, 4);
            }
            other => panic!("expected QueryTooLong, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn validation_runs_before_touching_the_backend() {
        // A failing provider would surface an Embedding error *if* it were reached; an
        // empty query must be rejected before that happens.
        let retriever = Retriever::new(
            MockEmbeddingProvider::failing("should not be reached"),
            InMemoryVectorStore::new(),
        );
        assert!(matches!(
            retriever.retrieve("  ", 5).await,
            Err(RetrievalError::EmptyQuery)
        ));
    }

    #[tokio::test]
    async fn top_k_of_zero_short_circuits_before_embedding() {
        // A valid query with `top_k == 0` is empty by construction, so the embedding
        // backend must not be contacted at all — a failing provider proves it wasn't.
        let retriever = Retriever::new(
            MockEmbeddingProvider::failing("should not be reached"),
            InMemoryVectorStore::new(),
        );
        assert!(retriever
            .retrieve("a valid query", 0)
            .await
            .unwrap()
            .is_empty());

        // Validation still wins over the short-circuit: a bad query errors even at k == 0.
        assert!(matches!(
            retriever.retrieve("  ", 0).await,
            Err(RetrievalError::EmptyQuery)
        ));
    }

    #[tokio::test]
    async fn retrieves_ranked_scored_results() {
        let retriever = retriever();
        let indexed = retriever
            .index(vec![
                chunk("c0", "retrieval augmented generation"),
                chunk("c1", "fixed size character chunking"),
                chunk("c2", "cosine similarity vector search"),
            ])
            .await
            .unwrap();
        assert_eq!(indexed, 3);

        // The mock embeds deterministically, so querying c2's exact text yields its exact
        // vector — a perfect cosine match that must rank first.
        let results = retriever
            .retrieve("cosine similarity vector search", 2)
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk.id, "c2");
        assert!((results[0].score - 1.0).abs() < 1e-6);
        assert!(results[0].score >= results[1].score);
        // The hit carries the chunk's text and metadata back with it.
        assert_eq!(results[0].content(), "cosine similarity vector search");
        assert_eq!(results[0].metadata().source, "corpus");
    }

    #[tokio::test]
    async fn top_k_caps_results_and_zero_returns_none() {
        let retriever = retriever();
        retriever
            .index(vec![
                chunk("a", "alpha"),
                chunk("b", "beta"),
                chunk("c", "gamma"),
            ])
            .await
            .unwrap();

        assert_eq!(retriever.retrieve("alpha", 1).await.unwrap().len(), 1);
        // k larger than the corpus returns everything.
        assert_eq!(retriever.retrieve("alpha", 99).await.unwrap().len(), 3);
        // k == 0 returns nothing, without error.
        assert!(retriever.retrieve("alpha", 0).await.unwrap().is_empty());
    }

    /// A provider that violates the [`EmbeddingProvider`] contract by returning a fixed
    /// number of vectors regardless of how many inputs it was given, to exercise the
    /// count-mismatch guard.
    struct MiscountingProvider {
        count: usize,
    }

    #[async_trait::async_trait]
    impl EmbeddingProvider for MiscountingProvider {
        async fn embed(&self, _inputs: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            Ok(vec![vec![1.0, 0.0]; self.count])
        }

        fn dimensions(&self) -> usize {
            2
        }
    }

    #[tokio::test]
    async fn index_rejects_a_provider_that_miscounts_embeddings() {
        let retriever =
            Retriever::new(MiscountingProvider { count: 1 }, InMemoryVectorStore::new());
        match retriever
            .index(vec![chunk("a", "alpha"), chunk("b", "beta")])
            .await
        {
            Err(RetrievalError::EmbeddingCountMismatch { expected, actual }) => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 1);
            }
            other => panic!("expected EmbeddingCountMismatch, got {other:?}"),
        }
        // Nothing reached the store.
        assert_eq!(retriever.store().len().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn retrieve_rejects_a_provider_that_returns_no_vector() {
        let retriever =
            Retriever::new(MiscountingProvider { count: 0 }, InMemoryVectorStore::new());
        match retriever.retrieve("a valid query", 5).await {
            Err(RetrievalError::EmbeddingCountMismatch { expected, actual }) => {
                assert_eq!(expected, 1);
                assert_eq!(actual, 0);
            }
            other => panic!("expected EmbeddingCountMismatch, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn surfaces_embedding_failures() {
        let retriever = Retriever::new(
            MockEmbeddingProvider::failing("rate limited"),
            InMemoryVectorStore::new(),
        );
        match retriever.retrieve("a valid query", 5).await {
            Err(RetrievalError::Embedding(EmbeddingError::Provider { message, .. })) => {
                assert_eq!(message, "rate limited");
            }
            other => panic!("expected an Embedding error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn handle_maps_request_to_response() {
        let retriever = retriever();
        retriever
            .index(vec![
                chunk("c0", "hello world"),
                chunk("c1", "goodbye moon"),
            ])
            .await
            .unwrap();

        let request = RetrievalRequest::new("hello world", 1);
        let response = retriever.handle(&request).await.unwrap();

        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].chunk.id, "c0");

        // The response serializes to the endpoint's JSON shape, scores included.
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"results\""));
        assert!(json.contains("\"score\""));
    }

    #[test]
    fn request_top_k_defaults_when_omitted() {
        let request: RetrievalRequest = serde_json::from_str(r#"{"query": "hi"}"#).unwrap();
        assert_eq!(request.query, "hi");
        assert_eq!(request.top_k, DEFAULT_TOP_K);
    }
}
