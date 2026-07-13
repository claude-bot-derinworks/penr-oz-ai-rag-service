//! Answer generation: retrieve relevant chunks, prompt a language model with them, and
//! return a grounded answer with its sources.
//!
//! This is the generative half of RAG and the engine behind a `POST /answer` endpoint.
//! An [`AnswerGenerator`] composes the abstractions the crate already provides:
//!
//! 1. a [`Retriever`](crate::retrieval::Retriever) turns the query into the top-k most
//!    relevant chunks, each scored by similarity, and
//! 2. an [`LlmProvider`](crate::llm::LlmProvider) completes a prompt assembled from the
//!    query and the retrieved context.
//!
//! Between the two sits a **confidence gate**: hits scoring below a minimum similarity
//! are dropped before the prompt is built, so unrelated chunks never reach the model
//! when retrieval confidence is low. If *no* hit clears the gate, the model is not
//! called at all and the response says so ([`NO_CONTEXT_ANSWER`]) with an empty source
//! list — the generator refuses to answer from context it does not trust rather than
//! prompting the model with noise.
//!
//! Every answer carries [`SourceRef`]s — the id, source document, chunk index, and
//! similarity score of each chunk the prompt was built from — so callers can attribute
//! the answer to the material it was grounded in.
//!
//! The serde-friendly [`AnswerRequest`] / [`AnswerResponse`] pair is the wire shape of
//! the endpoint: deserialize the `POST` body into an [`AnswerRequest`], call
//! [`AnswerGenerator::handle`], and serialize the [`AnswerResponse`] back. Like the
//! retrieval layer, no web framework is pulled in here; the crate's own `penr-oz-rag
//! serve` command hosts exactly this endpoint.
//!
//! ## Example
//!
//! ```
//! use penr_oz_ai_rag_service::{
//!     AnswerGenerator, Chunk, ChunkMetadata, InMemoryVectorStore, MockEmbeddingProvider,
//!     MockLlmProvider, Retriever,
//! };
//!
//! # async fn run() -> Result<(), penr_oz_ai_rag_service::GenerationError> {
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
//! retriever.index(vec![chunk("c0", "chunking splits documents")]).await?;
//!
//! let generator = AnswerGenerator::new(retriever, MockLlmProvider::new());
//! let response = generator.answer("chunking splits documents", 1, 0.5).await?;
//!
//! assert!(!response.answer.is_empty());
//! assert_eq!(response.sources[0].id, "c0");
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::embedding::EmbeddingProvider;
use crate::llm::{LlmError, LlmProvider};
use crate::retrieval::{RetrievalError, Retriever, DEFAULT_TOP_K};
use crate::vector::{SearchResult, VectorStore};

/// Minimum similarity a retrieved chunk must score to be included in the prompt, when
/// neither the [`AnswerGenerator`] nor the [`AnswerRequest`] overrides it.
///
/// `0.0` drops only chunks that are *anti*-correlated with the query (cosine similarity
/// is negative). The right threshold depends on the embedding model — semantically
/// "related" scores differently across models — so deployments should tune it with
/// [`AnswerGenerator::with_min_score`] or per request via [`AnswerRequest::min_score`].
pub const DEFAULT_MIN_SCORE: f32 = 0.0;

/// The answer returned — without calling the language model — when no retrieved chunk
/// clears the minimum-score gate.
///
/// Exposed as a constant so clients and tests can distinguish "the model answered" from
/// "retrieval found nothing trustworthy to ground an answer in".
pub const NO_CONTEXT_ANSWER: &str =
    "No sufficiently relevant context was found to answer this query.";

/// The body of a `POST /answer` request: a user query, how many chunks to retrieve, and
/// optionally a minimum similarity for a chunk to be used as context.
///
/// `top_k` defaults to [`DEFAULT_TOP_K`] and `min_score` to the generator's configured
/// threshold when omitted, so a minimal request is just `{"query": "..."}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnswerRequest {
    /// The user's question.
    pub query: String,
    /// Maximum number of chunks to retrieve as candidate context.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Minimum similarity score a retrieved chunk must reach to be included in the
    /// prompt. When omitted, the generator's configured threshold applies.
    #[serde(default)]
    pub min_score: Option<f32>,
}

fn default_top_k() -> usize {
    DEFAULT_TOP_K
}

impl AnswerRequest {
    /// Build a request for `query` retrieving up to `top_k` context chunks, deferring
    /// to the generator's minimum-score threshold.
    pub fn new(query: impl Into<String>, top_k: usize) -> Self {
        Self {
            query: query.into(),
            top_k,
            min_score: None,
        }
    }

    /// Override the minimum similarity score for this request (builder style).
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = Some(min_score);
        self
    }
}

/// A reference to a chunk the answer was grounded in: which chunk, from where, and how
/// relevant retrieval judged it.
///
/// Deliberately lighter than a full [`SearchResult`] — the chunk's text already appears
/// in the prompt and (with a real model) is paraphrased in the answer, so the response
/// carries provenance rather than repeating content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceRef {
    /// The chunk's identifier (`{document_id}#{chunk_index}`).
    pub id: String,
    /// Identifier of the document the chunk came from.
    pub source: String,
    /// Zero-based position of the chunk within its document.
    pub chunk_index: usize,
    /// Similarity of the chunk to the query, as scored by retrieval.
    pub score: f32,
}

impl From<&SearchResult> for SourceRef {
    fn from(hit: &SearchResult) -> Self {
        Self {
            id: hit.chunk.id.clone(),
            source: hit.chunk.metadata.source.clone(),
            chunk_index: hit.chunk.metadata.chunk_index,
            score: hit.score,
        }
    }
}

/// The body of a `POST /answer` response: the generated answer and the chunks it was
/// grounded in.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnswerResponse {
    /// The language model's answer — or [`NO_CONTEXT_ANSWER`] when no retrieved chunk
    /// cleared the minimum-score gate (in which case `sources` is empty and the model
    /// was never called).
    pub answer: String,
    /// References to the chunks the prompt was built from, most relevant first.
    pub sources: Vec<SourceRef>,
}

/// Assemble the prompt sent to the language model: the retrieved `context` passages,
/// numbered and attributed, followed by the user's `query`.
///
/// The instruction pins the model to the supplied passages so answers stay grounded in
/// the corpus. Exposed as a plain function so alternative generators (or tests) can
/// reuse or inspect the exact prompt shape.
pub fn build_prompt(query: &str, context: &[SearchResult]) -> String {
    let mut prompt = String::from(
        "Answer the question using only the numbered context passages below. \
         If the passages do not contain the answer, say so instead of guessing.\n\nContext:\n",
    );
    for (index, hit) in context.iter().enumerate() {
        prompt.push_str(&format!(
            "[{}] {} (score {:.3})\n{}\n\n",
            index + 1,
            hit.chunk.id,
            hit.score,
            hit.chunk.content
        ));
    }
    prompt.push_str(&format!("Question: {query}\nAnswer:"));
    prompt
}

/// Generates grounded answers: retrieve, gate by confidence, prompt, and attribute.
///
/// An `AnswerGenerator` owns a [`Retriever`] and an [`LlmProvider`] and wires them into
/// the answer path: retrieve the top-k chunks, drop those scoring below the minimum,
/// build the prompt, call the model, and return the answer with [`SourceRef`]s to the
/// chunks it was grounded in. All backends are generic so the concrete embedding model,
/// vector store, and language model are chosen by the caller and cost nothing at
/// runtime; because every method takes `&self`, a generator can be shared (e.g. behind
/// an `Arc`) and queried concurrently, the way a request handler would.
pub struct AnswerGenerator<E, V, L> {
    retriever: Retriever<E, V>,
    llm: L,
    min_score: f32,
}

impl<E, V, L> AnswerGenerator<E, V, L>
where
    E: EmbeddingProvider,
    V: VectorStore,
    L: LlmProvider,
{
    /// Create a generator over `retriever` and `llm`, using [`DEFAULT_MIN_SCORE`] as
    /// the confidence gate.
    pub fn new(retriever: Retriever<E, V>, llm: L) -> Self {
        Self {
            retriever,
            llm,
            min_score: DEFAULT_MIN_SCORE,
        }
    }

    /// Override the minimum similarity a chunk must score to be used as context
    /// (builder style). Requests can still override this per call via
    /// [`AnswerRequest::min_score`].
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = min_score;
        self
    }

    /// The minimum similarity score this generator requires of context chunks.
    pub fn min_score(&self) -> f32 {
        self.min_score
    }

    /// Borrow the underlying retriever (e.g. to [`index`](Retriever::index) chunks, or
    /// to serve a plain retrieval endpoint alongside answering).
    pub fn retriever(&self) -> &Retriever<E, V> {
        &self.retriever
    }

    /// Borrow the language-model provider.
    pub fn llm(&self) -> &L {
        &self.llm
    }

    /// Answer `query` from the indexed corpus: retrieve up to `top_k` chunks, keep those
    /// scoring at least `min_score`, prompt the model with them, and return the answer
    /// with its sources.
    ///
    /// When no chunk clears `min_score` — retrieval confidence is too low across the
    /// board, or the index is empty — the model is **not** called: the response carries
    /// [`NO_CONTEXT_ANSWER`] and no sources, so unrelated chunks are never presented to
    /// the model as facts.
    ///
    /// # Errors
    /// - [`GenerationError::Retrieval`] if the query is invalid (empty, oversized) or
    ///   the embedding/vector-store backend fails; see [`RetrievalError`].
    /// - [`GenerationError::Llm`] if the language model fails to generate.
    pub async fn answer(
        &self,
        query: &str,
        top_k: usize,
        min_score: f32,
    ) -> Result<AnswerResponse, GenerationError> {
        let hits = self.retriever.retrieve(query, top_k).await?;
        let context: Vec<SearchResult> = hits
            .into_iter()
            .filter(|hit| hit.score >= min_score)
            .collect();

        if context.is_empty() {
            return Ok(AnswerResponse {
                answer: NO_CONTEXT_ANSWER.to_string(),
                sources: Vec::new(),
            });
        }

        let prompt = build_prompt(query, &context);
        let answer = self.llm.generate(&prompt).await?;
        let sources = context.iter().map(SourceRef::from).collect();

        Ok(AnswerResponse { answer, sources })
    }

    /// Handle an [`AnswerRequest`], returning an [`AnswerResponse`].
    ///
    /// This is the request/response adapter a `POST /answer` handler calls: it maps the
    /// request onto [`answer`](Self::answer), falling back to the generator's configured
    /// [`min_score`](Self::min_score) when the request does not set one.
    pub async fn handle(&self, request: &AnswerRequest) -> Result<AnswerResponse, GenerationError> {
        let min_score = request.min_score.unwrap_or(self.min_score);
        self.answer(&request.query, request.top_k, min_score).await
    }
}

/// The set of errors answer generation can produce.
///
/// Retrieval errors pass through unchanged so an HTTP layer can keep mapping validation
/// failures to `400 Bad Request` exactly as it does for the retrieval endpoint; model
/// failures get their own variant so they can be mapped (and retried) separately.
#[derive(Debug, Error)]
pub enum GenerationError {
    /// Retrieving context for the query failed (invalid query or backend failure).
    #[error(transparent)]
    Retrieval(#[from] RetrievalError),

    /// The language model failed to generate an answer.
    #[error("failed to generate an answer: {0}")]
    Llm(#[from] LlmError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Chunk, ChunkMetadata, Metadata};
    use crate::embedding::MockEmbeddingProvider;
    use crate::llm::MockLlmProvider;
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

    fn hit(id: &str, content: &str, score: f32) -> SearchResult {
        SearchResult {
            chunk: chunk(id, content),
            score,
        }
    }

    async fn generator_over(
        llm: MockLlmProvider,
        chunks: Vec<Chunk>,
    ) -> AnswerGenerator<MockEmbeddingProvider, InMemoryVectorStore, MockLlmProvider> {
        let retriever = Retriever::new(MockEmbeddingProvider::new(), InMemoryVectorStore::new());
        retriever.index(chunks).await.unwrap();
        AnswerGenerator::new(retriever, llm)
    }

    #[test]
    fn prompt_numbers_context_and_ends_with_the_question() {
        let context = [
            hit("a#0", "alpha content", 0.9),
            hit("b#0", "beta content", 0.8),
        ];
        let prompt = build_prompt("what is alpha?", &context);

        assert!(prompt.contains("[1] a#0 (score 0.900)\nalpha content"));
        assert!(prompt.contains("[2] b#0 (score 0.800)\nbeta content"));
        assert!(prompt.ends_with("Question: what is alpha?\nAnswer:"));
        // The grounding instruction precedes the context.
        assert!(prompt.starts_with("Answer the question using only"));
    }

    #[tokio::test]
    async fn answers_with_sources_for_the_context_used() {
        // The mock embeds deterministically, so querying a chunk's exact text is a
        // perfect cosine match; the echoing mock LLM returns the prompt, proving the
        // matched chunk's content reached the model.
        let generator = generator_over(
            MockLlmProvider::new(),
            vec![
                chunk("c0", "retrieval augmented generation"),
                chunk("c1", "cosine similarity vector search"),
            ],
        )
        .await;

        let response = generator
            .answer("retrieval augmented generation", 2, 0.0)
            .await
            .unwrap();

        assert!(response.answer.contains("retrieval augmented generation"));
        assert_eq!(response.sources.len(), 2);
        assert_eq!(response.sources[0].id, "c0");
        assert_eq!(response.sources[0].source, "corpus");
        assert!((response.sources[0].score - 1.0).abs() < 1e-6);
        // Sources stay ordered most relevant first, mirroring retrieval.
        assert!(response.sources[0].score >= response.sources[1].score);
    }

    #[tokio::test]
    async fn low_confidence_chunks_are_kept_out_of_prompt_and_sources() {
        let generator = generator_over(
            MockLlmProvider::new(),
            vec![
                chunk("c0", "retrieval augmented generation"),
                chunk("c1", "cosine similarity vector search"),
            ],
        )
        .await;

        // Only the exact-text match scores ~1.0; the other chunk scores well below the
        // 0.99 gate and must appear in neither the prompt (echoed answer) nor sources.
        let response = generator
            .answer("retrieval augmented generation", 2, 0.99)
            .await
            .unwrap();

        assert_eq!(response.sources.len(), 1);
        assert_eq!(response.sources[0].id, "c0");
        assert!(!response.answer.contains("cosine similarity vector search"));
    }

    #[tokio::test]
    async fn no_relevant_context_skips_the_model_entirely() {
        // A failing LLM would surface an error *if* it were called; a gate no chunk can
        // clear (scores never exceed 1.0) must answer without contacting it.
        let generator = generator_over(
            MockLlmProvider::failing("should not be reached"),
            vec![chunk("c0", "retrieval augmented generation")],
        )
        .await;

        let response = generator
            .answer("unrelated question", 5, 1.1)
            .await
            .unwrap();

        assert_eq!(response.answer, NO_CONTEXT_ANSWER);
        assert!(response.sources.is_empty());
    }

    #[tokio::test]
    async fn empty_index_answers_no_context_without_calling_the_model() {
        let generator =
            generator_over(MockLlmProvider::failing("should not be reached"), vec![]).await;

        let response = generator.answer("anything", 5, 0.0).await.unwrap();

        assert_eq!(response.answer, NO_CONTEXT_ANSWER);
        assert!(response.sources.is_empty());
    }

    #[tokio::test]
    async fn invalid_queries_fail_validation_before_any_generation() {
        let generator = generator_over(
            MockLlmProvider::failing("should not be reached"),
            vec![chunk("c0", "content")],
        )
        .await;

        assert!(matches!(
            generator.answer("   ", 5, 0.0).await,
            Err(GenerationError::Retrieval(RetrievalError::EmptyQuery))
        ));
    }

    #[tokio::test]
    async fn model_failures_surface_as_llm_errors() {
        let generator = generator_over(
            MockLlmProvider::failing("rate limited"),
            vec![chunk("c0", "retrieval augmented generation")],
        )
        .await;

        match generator
            .answer("retrieval augmented generation", 1, 0.0)
            .await
        {
            Err(GenerationError::Llm(LlmError::Provider { message, .. })) => {
                assert_eq!(message, "rate limited");
            }
            other => panic!("expected an Llm error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn handle_applies_request_overrides_and_generator_defaults() {
        let generator = generator_over(
            MockLlmProvider::with_reply("grounded answer"),
            vec![
                chunk("c0", "retrieval augmented generation"),
                chunk("c1", "cosine similarity vector search"),
            ],
        )
        .await
        // A generator-level gate no chunk can clear...
        .with_min_score(1.1);

        // ...applies when the request does not override it...
        let request = AnswerRequest::new("retrieval augmented generation", 2);
        let response = generator.handle(&request).await.unwrap();
        assert_eq!(response.answer, NO_CONTEXT_ANSWER);

        // ...and is overridden when it does.
        let request = request.with_min_score(0.99);
        let response = generator.handle(&request).await.unwrap();
        assert_eq!(response.answer, "grounded answer");
        assert_eq!(response.sources.len(), 1);
        assert_eq!(response.sources[0].id, "c0");
    }

    #[tokio::test]
    async fn response_serializes_answer_and_sources() {
        let generator = generator_over(
            MockLlmProvider::with_reply("an answer"),
            vec![chunk("c0", "hello world")],
        )
        .await;

        let response = generator
            .handle(&AnswerRequest::new("hello world", 1))
            .await
            .unwrap();

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"answer\":\"an answer\""));
        assert!(json.contains("\"sources\""));
        assert!(json.contains("\"id\":\"c0\""));
        assert!(json.contains("\"score\""));
    }

    #[test]
    fn request_defaults_apply_when_fields_are_omitted() {
        let request: AnswerRequest = serde_json::from_str(r#"{"query": "hi"}"#).unwrap();
        assert_eq!(request.query, "hi");
        assert_eq!(request.top_k, DEFAULT_TOP_K);
        assert_eq!(request.min_score, None);

        let request: AnswerRequest =
            serde_json::from_str(r#"{"query": "hi", "min_score": 0.5}"#).unwrap();
        assert_eq!(request.min_score, Some(0.5));
    }
}
