# CLAUDE.md

Context for Claude when working in this repository. Every statement below is
verifiable in the code; where this file, the README, or doc comments disagree with the
code, the code wins.

## What this repo is

A Retrieval-Augmented Generation (RAG) service written in Rust (edition 2021, MSRV
1.74): a library crate (`penr_oz_ai_rag_service`) plus a CLI binary (`penr-oz-rag`)
that ingests text files into metadata-rich chunks, indexes them for vector search, and
serves `POST /retrieve` (top-k similarity) and `POST /answer` (grounded LLM answers
with source references) over axum. Every stage sits behind a trait, so loaders,
chunkers, chunk stores, embedding providers, vector stores, and LLM providers are all
swappable without touching the other stages. See [README.md](README.md) for full usage,
wire formats, and extension examples.

## Module map

| Path | Contents |
| --- | --- |
| `src/lib.rs` | Crate root; re-exports the entire public API |
| `src/main.rs` | `penr-oz-rag` CLI (clap): `ingest` + `serve`, axum handlers |
| `src/error.rs` | `RagError` / `Result` for the ingestion side |
| `src/document.rs` | `Document`, `Chunk`, `ChunkMetadata`, `Metadata` |
| `src/loader/` | `Loader` trait, `LoaderRegistry`, `TextLoader` (`.txt`/`.text` only) |
| `src/chunker/` | `Chunker` trait, `FixedSizeChunker` (char windows, overlap, word-aware) |
| `src/storage/` | `ChunkStore` trait, `InMemoryStorage`, `JsonlStorage` |
| `src/embedding/` | `EmbeddingProvider` trait, `EmbeddingError`, `MockEmbeddingProvider` |
| `src/vector/` | `VectorStore` trait, `InMemoryVectorStore`, `cosine_similarity` |
| `src/retrieval.rs` | `Retriever` (validate → embed → search), request/response types |
| `src/llm/` | `LlmProvider` trait, `LlmError`, `MockLlmProvider` |
| `src/generation.rs` | `AnswerGenerator` (retrieve → gate → prompt → answer), `build_prompt` |
| `src/pipeline.rs` | `IngestionPipeline` + `PipelineBuilder` |
| `tests/` | One integration file per stage: `ingestion`, `embedding`, `vector_search`, `retrieval`, `generation`, `serve` (HTTP end-to-end) |

## Commands

No install step beyond a stable Rust toolchain (≥ 1.74, via rustup). No external
services, API keys, or network access are needed to build or test.

```bash
cargo build --release                                    # binary at target/release/penr-oz-rag
cargo test                                               # unit + integration + doc tests
cargo clippy --all-targets --all-features -- -D warnings # CI fails on any warning
cargo fmt --all -- --check                               # CI enforces formatting
```

CI (`.github/workflows/ci.yml`) runs exactly: fmt check, clippy (deny warnings), build,
test — run all four locally before pushing.

```bash
penr-oz-rag ingest <INPUT> [-o out.jsonl] [--chunk-size 800] [--overlap 100] [--no-word-aware]
penr-oz-rag serve  <INPUT> [--addr 127.0.0.1:8080] [--min-score 0]
```

## Conventions

- Each layer defines its own error enum with `thiserror` 2.x (`RagError`,
  `EmbeddingError`, `VectorStoreError`, `RetrievalError`, `LlmError`,
  `GenerationError`). No `anyhow` anywhere.
- Provider/store traits use `async-trait` and are object-safe
  (`Box<dyn LlmProvider>`, `Arc<dyn VectorStore>`).
- The library is web-framework-free: axum, clap, and tokio are used only in
  `src/main.rs`. Keep new library code runtime-agnostic.
- Unit tests live in `#[cfg(test)]` modules beside the code; cross-stage tests live in
  `tests/`. `tempfile` is the only dev-dependency.
- Public items carry doc comments with runnable examples (they execute as doc tests).
  Update README.md when public behavior changes.

## Gotchas

- All offsets and sizes are **characters, not bytes** (`start_char`/`end_char`,
  `--chunk-size`); this is deliberate for Unicode text. `overlap` must be strictly less
  than `chunk_size` or `FixedSizeChunker::new` returns an error.
- The only built-in providers are mocks: `MockEmbeddingProvider` produces
  deterministic hash-based vectors (similarity is **not** semantic) and
  `MockLlmProvider` echoes the prompt back (`with_reply` / `failing` variants exist for
  tests). Real providers are drop-in implementations of `EmbeddingProvider` /
  `LlmProvider`; no handler changes needed.
- `/answer` gates by confidence: chunks scoring below `min_score` are dropped, and if
  none survive the LLM is **not called** — the response is still `200` with the
  `NO_CONTEXT_ANSWER` sentinel and an empty `sources` list.
- Defaults live in code: `DEFAULT_TOP_K = 5`, `DEFAULT_MAX_QUERY_CHARS = 8192`,
  `DEFAULT_MIN_SCORE = 0.0`. Empty/whitespace or oversized queries are rejected with
  `400` before any embedding or search work happens.
- HTTP error mapping in `serve`: validation → `400`, embedding/LLM backend failure →
  `502`, vector-store failure → `500`.
- Ingesting a **directory** silently skips (and counts) files with no registered
  loader; ingesting a **single file** of an unsupported format is an error.
- The first vector inserted into `InMemoryVectorStore` fixes its dimensionality; later
  mismatches return `VectorStoreError::DimensionMismatch`.
- `--addr` with port `0` binds a free port (this is how `tests/serve.rs` avoids
  collisions).
