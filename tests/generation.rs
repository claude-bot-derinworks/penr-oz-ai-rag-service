//! End-to-end answer-generation test, exercising the [`AnswerGenerator`] the way a
//! `POST /answer` handler would: index a corpus, then answer questions grounded in the
//! retrieved chunks — filtering low-confidence context, attributing sources, and
//! refusing to prompt the model when nothing relevant was found.

use penr_oz_ai_rag_service::{
    AnswerGenerator, AnswerRequest, Chunk, ChunkMetadata, GenerationError, InMemoryVectorStore,
    Metadata, MockEmbeddingProvider, MockLlmProvider, RetrievalError, Retriever, NO_CONTEXT_ANSWER,
};

/// Build a chunk with the given id and content; metadata is filler the generator turns
/// into source references untouched.
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

/// Index the standard three-chunk corpus and wrap it in a generator over `llm`.
async fn generator(
    llm: MockLlmProvider,
) -> AnswerGenerator<MockEmbeddingProvider, InMemoryVectorStore, MockLlmProvider> {
    let retriever = Retriever::new(MockEmbeddingProvider::new(), InMemoryVectorStore::new());
    retriever
        .index(vec![
            chunk("c0", "retrieval augmented generation"),
            chunk("c1", "fixed size character chunking"),
            chunk("c2", "cosine similarity vector search"),
        ])
        .await
        .expect("indexing succeeds");
    AnswerGenerator::new(retriever, llm)
}

#[tokio::test]
async fn answers_grounded_in_retrieved_context_with_sources() {
    // The echoing mock LLM returns the prompt it was shown, so the answer proves the
    // retrieved context and the question both reached the model.
    let generator = generator(MockLlmProvider::new()).await;

    let response = generator
        .answer("cosine similarity vector search", 2, 0.0)
        .await
        .expect("generation succeeds");

    assert!(response.answer.contains("cosine similarity vector search"));
    assert!(response
        .answer
        .contains("Question: cosine similarity vector search"));

    // Sources reference the chunks the prompt was built from, most relevant first.
    assert_eq!(response.sources.len(), 2);
    assert_eq!(response.sources[0].id, "c2");
    assert_eq!(response.sources[0].source, "corpus");
    assert!((response.sources[0].score - 1.0).abs() < 1e-6);
    assert!(response.sources[1].score <= response.sources[0].score);
}

#[tokio::test]
async fn low_confidence_chunks_are_excluded_from_context() {
    let generator = generator(MockLlmProvider::new()).await;

    // Only the exact-text match clears a 0.99 gate; the unrelated chunks must reach
    // neither the prompt (visible via the echoed answer) nor the source list.
    let response = generator
        .answer("retrieval augmented generation", 3, 0.99)
        .await
        .unwrap();

    assert_eq!(response.sources.len(), 1);
    assert_eq!(response.sources[0].id, "c0");
    assert!(!response.answer.contains("fixed size character chunking"));
    assert!(!response.answer.contains("cosine similarity vector search"));
}

#[tokio::test]
async fn refuses_to_answer_when_no_chunk_is_confident_enough() {
    // A failing LLM proves the model is never contacted: cosine scores cannot exceed
    // 1.0, so a 1.1 gate filters every hit and the generator answers on its own.
    let generator = generator(MockLlmProvider::failing("should not be reached")).await;

    let response = generator.answer("anything at all", 3, 1.1).await.unwrap();

    assert_eq!(response.answer, NO_CONTEXT_ANSWER);
    assert!(response.sources.is_empty());
}

#[tokio::test]
async fn handle_returns_the_endpoint_response_shape() {
    let generator = generator(MockLlmProvider::with_reply("a grounded answer")).await;

    // A request with top_k and min_score omitted from JSON falls back to the defaults
    // and still works.
    let request: AnswerRequest =
        serde_json::from_str(r#"{"query": "retrieval augmented generation"}"#).unwrap();
    let response = generator.handle(&request).await.unwrap();

    assert_eq!(response.answer, "a grounded answer");
    assert!(!response.sources.is_empty());

    // Round-trips through JSON the way the endpoint would serialize it.
    let json = serde_json::to_string(&response).unwrap();
    let decoded: penr_oz_ai_rag_service::AnswerResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, response);
}

#[tokio::test]
async fn invalid_queries_and_model_failures_surface_as_errors() {
    let generator = generator(MockLlmProvider::failing("model offline")).await;

    // Validation failures come through as retrieval errors, before any generation.
    assert!(matches!(
        generator.answer("   ", 5, 0.0).await,
        Err(GenerationError::Retrieval(RetrievalError::EmptyQuery))
    ));

    // With relevant context retrieved, the model failure itself surfaces.
    assert!(matches!(
        generator
            .answer("retrieval augmented generation", 1, 0.0)
            .await,
        Err(GenerationError::Llm(_))
    ));
}
