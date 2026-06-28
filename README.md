# penr-oz-ai-rag-service

Implementation of a Retrieval-Augmented Generation (RAG) service for AI.

This repository currently provides the **document ingestion pipeline**: the stage that
turns raw source files into metadata-rich, retrievable chunks. It is written in Rust as
a reusable library (`penr_oz_ai_rag_service`) with a thin command-line front-end
(`penr-oz-rag`).

## Overview

Ingestion is modeled as three decoupled stages, each behind a trait so it can be
replaced or extended independently:

```
 file ──▶ Loader ──▶ Document ──▶ Chunker ──▶ Chunk[] ──▶ ChunkStore ──▶ storage
          (load)                  (split +              (persist)
                                   metadata)
```

1. **Load** — a [`Loader`] reads a file and normalizes it into a `Document`. The only
   built-in loader today is `TextLoader` (`.txt`, `.text`). New formats are added by
   implementing `Loader` and registering it with a `LoaderRegistry` — **nothing
   downstream changes**, which is what keeps PDF / HTML / Markdown loaders a drop-in
   addition later.
2. **Chunk** — a `Chunker` splits a `Document` into ordered `Chunk`s and attaches
   positional and provenance metadata. The built-in `FixedSizeChunker` produces
   fixed-size, overlapping, character-based windows and prefers word boundaries.
3. **Store** — a `ChunkStore` persists the chunks. Built-in backends are
   `InMemoryStorage` and `JsonlStorage` (one JSON object per line).

`IngestionPipeline` composes the three stages and reports what it did.

## Features

- Ingest a single text file or a directory tree (walked recursively, deterministic
  order).
- Character-based chunking (correct for multi-byte/Unicode text) with configurable
  size and overlap, and optional word-boundary awareness.
- Rich per-chunk metadata: source id, chunk index, total chunks, character offsets, and
  provenance propagated from the loader (e.g. `loader`, `filename`).
- Pluggable loaders, chunkers, and storage backends via small traits.
- Meaningful, specific error messages for invalid input (missing file, unsupported
  format, non-UTF-8 data, empty document, bad chunker configuration).

## Requirements

- Rust 1.74 or newer (stable). Install via [rustup](https://rustup.rs/).

## Build & test

```bash
cargo build --release   # binary at target/release/penr-oz-rag
cargo test              # unit + integration + doc tests
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

## Command-line usage

```bash
penr-oz-rag ingest <INPUT> [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `<INPUT>` | — | Path to a file or directory to ingest. |
| `-o, --output <FILE>` | _(none)_ | Write chunks as JSON Lines to `FILE`. If omitted, chunks are produced and counted in memory but not persisted. |
| `--chunk-size <N>` | `800` | Maximum number of characters per chunk. |
| `--overlap <N>` | `100` | Characters shared between consecutive chunks (must be `< chunk-size`). |
| `--no-word-aware` | off | Make exact character cuts instead of preferring word boundaries. |

### Examples

```bash
# Ingest one file and just report how many chunks it produces (in memory)
penr-oz-rag ingest ./docs/notes.txt

# Ingest a whole directory and persist chunks as JSON Lines
penr-oz-rag ingest ./docs --output chunks.jsonl --chunk-size 800 --overlap 100
```

Example run:

```text
$ penr-oz-rag ingest ./docs --output out/chunks.jsonl --chunk-size 120 --overlap 24
Ingested 2 file(s), skipped 1, created 4 chunk(s).
  ./docs/intro.txt -> 3 chunk(s)
  ./docs/notes.text -> 1 chunk(s)
Wrote 4 chunk(s) to out/chunks.jsonl
```

When ingesting a **directory**, files whose format has no registered loader are skipped
and counted (`skipped`). When ingesting a **single file**, an unsupported format is an
error, since you asked for that file explicitly. The process exits non-zero on any
error.

## Output format

Each persisted chunk is one JSON object (pretty-printed here for readability):

```json
{
  "id": "docs/intro.txt#0",
  "content": "Retrieval augmented generation grounds a language model in external knowledge. The ingestion pipeline loads raw",
  "metadata": {
    "source": "docs/intro.txt",
    "chunk_index": 0,
    "total_chunks": 3,
    "start_char": 0,
    "end_char": 111,
    "extra": {
      "filename": "intro.txt",
      "loader": "text"
    }
  }
}
```

`start_char` / `end_char` are character (not byte) offsets into the source document.

## Library usage

Add the crate to another workspace member or use it directly:

```rust
use penr_oz_ai_rag_service::{
    FixedSizeChunker, IngestionPipeline, InMemoryStorage,
};

fn main() -> penr_oz_ai_rag_service::Result<()> {
    let chunker = FixedSizeChunker::new(800, 100)?; // chunk_size, overlap (in chars)

    let mut pipeline = IngestionPipeline::builder(InMemoryStorage::new())
        .chunker(chunker)
        .build();

    let report = pipeline.ingest_path("docs")?;
    println!("created {} chunks", report.chunks_created);

    for chunk in pipeline.into_store().chunks() {
        println!("{} ({} chars)", chunk.id, chunk.content.chars().count());
    }
    Ok(())
}
```

To persist instead, swap the store for `JsonlStorage::create("chunks.jsonl")?` and call
`pipeline.flush()?` when done.

## Extending the pipeline

The pipeline is built to grow. To add support for a new format, implement `Loader` and
register it:

```rust
use std::path::Path;
use std::sync::Arc;
use penr_oz_ai_rag_service::{Document, Loader, LoaderRegistry, Result};

struct MarkdownLoader;

impl Loader for MarkdownLoader {
    fn extensions(&self) -> &[&str] {
        &["md", "markdown"]
    }

    fn load(&self, path: &Path) -> Result<Document> {
        // read `path`, strip markup, and return a normalized `Document`
        todo!()
    }
}

let mut loaders = LoaderRegistry::with_defaults();
loaders.register(Arc::new(MarkdownLoader));
// IngestionPipeline::builder(store).loaders(loaders).build()
```

The same pattern applies to chunking (implement `Chunker`) and storage (implement
`ChunkStore`, e.g. to write to a vector database) — each is selected on the
`IngestionPipeline` builder without changing the other stages.

## Embeddings

Turning chunks into vectors is decoupled from ingestion behind the `EmbeddingProvider`
trait, so the embedding backend (a hosted API, a local model, …) can be swapped without
touching the rest of the service. The trait embeds a **batch** at a time, is object-safe
(usable as `Box<dyn EmbeddingProvider>`), and surfaces a dedicated `EmbeddingError`:

```rust
use penr_oz_ai_rag_service::{EmbeddingError, EmbeddingProvider, MockEmbeddingProvider};

#[tokio::main]
async fn main() -> Result<(), EmbeddingError> {
    // `MockEmbeddingProvider` produces deterministic vectors with no network — handy in
    // tests and examples. Swap in a real provider behind the same trait.
    let provider = MockEmbeddingProvider::new();
    let vectors = provider.embed(&["hello", "world"]).await?;

    assert_eq!(vectors.len(), 2);
    assert_eq!(vectors[0].len(), provider.dimensions());
    Ok(())
}
```

Provider-specific code (HTTP, auth, request shaping) lives inside each implementation, so
adding a real provider is a matter of implementing `EmbeddingProvider` and returning
`EmbeddingError` for failures.

## Project layout

```
src/
├── lib.rs            crate root and re-exports
├── main.rs           `penr-oz-rag` CLI
├── error.rs          RagError / Result
├── document.rs       Document, Chunk, ChunkMetadata
├── loader/           Loader trait, LoaderRegistry, TextLoader
├── chunker/          Chunker trait, FixedSizeChunker
├── storage/          ChunkStore trait, InMemoryStorage, JsonlStorage
├── embedding/        EmbeddingProvider trait, EmbeddingError, MockEmbeddingProvider
└── pipeline.rs       IngestionPipeline + builder
tests/
├── ingestion.rs      end-to-end ingestion tests
└── embedding.rs      embedding abstraction tests
```

## License

Licensed under the [MIT License](LICENSE).

[`Loader`]: src/loader/mod.rs
