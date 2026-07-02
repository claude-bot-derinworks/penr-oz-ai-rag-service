//! Command-line front-end for the RAG service.
//!
//! ```text
//! # Count chunks from a file in memory
//! penr-oz-rag ingest ./docs/notes.txt
//!
//! # Ingest a directory and write chunks as JSON Lines
//! penr-oz-rag ingest ./docs --output chunks.jsonl --chunk-size 800 --overlap 100
//!
//! # Ingest a directory and serve retrieval over it
//! penr-oz-rag serve ./docs --addr 127.0.0.1:8080
//! ```

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use clap::{Args, Parser, Subcommand};

use penr_oz_ai_rag_service::{
    ChunkStore, FixedSizeChunker, InMemoryStorage, InMemoryVectorStore, IngestReport,
    IngestionPipeline, JsonlStorage, MockEmbeddingProvider, Result, RetrievalError,
    RetrievalRequest, Retriever,
};

/// The retriever the `serve` command hosts: the in-process embedding provider and vector
/// store, shared across request handlers.
///
/// `MockEmbeddingProvider` is the only provider in the crate today, so served similarity
/// is deterministic-hash based rather than semantic; a real provider drops in here once
/// one exists, without touching the handler.
type ServedRetriever = Retriever<MockEmbeddingProvider, InMemoryVectorStore>;

/// Document ingestion and retrieval for a RAG service.
#[derive(Debug, Parser)]
#[command(name = "penr-oz-rag", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Ingest a text file (or a directory of text files) into chunks.
    Ingest(IngestArgs),
    /// Ingest a file or directory, index it, and serve `POST /retrieve` over HTTP.
    Serve(ServeArgs),
}

#[derive(Debug, Args)]
struct IngestArgs {
    /// Path to a file or directory to ingest.
    input: PathBuf,

    /// Write chunks as JSON Lines to this file. When omitted, chunks are produced and
    /// counted in memory but not persisted.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Maximum number of characters per chunk.
    #[arg(long, default_value_t = 800)]
    chunk_size: usize,

    /// Number of overlapping characters shared between consecutive chunks.
    #[arg(long, default_value_t = 100)]
    overlap: usize,

    /// Make exact character cuts instead of preferring word boundaries.
    #[arg(long)]
    no_word_aware: bool,
}

#[derive(Debug, Args)]
struct ServeArgs {
    /// Path to a file or directory to ingest and serve retrieval over.
    input: PathBuf,

    /// Address to bind the HTTP server to. Use port 0 to pick a free port.
    #[arg(long, default_value = "127.0.0.1:8080")]
    addr: SocketAddr,

    /// Maximum number of characters per chunk.
    #[arg(long, default_value_t = 800)]
    chunk_size: usize,

    /// Number of overlapping characters shared between consecutive chunks.
    #[arg(long, default_value_t = 100)]
    overlap: usize,

    /// Make exact character cuts instead of preferring word boundaries.
    #[arg(long)]
    no_word_aware: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> std::result::Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Command::Ingest(args) => Ok(ingest(args)?),
        Command::Serve(args) => serve(args),
    }
}

fn ingest(args: IngestArgs) -> Result<()> {
    let chunker =
        FixedSizeChunker::new(args.chunk_size, args.overlap)?.word_aware(!args.no_word_aware);

    match args.output {
        Some(output) => {
            let store = JsonlStorage::create(&output)?;
            // Exclude the output file so it is not re-ingested if it lives inside the
            // directory being ingested (e.g. `--output docs/chunks.txt docs`).
            let (report, store) = run_pipeline(store, chunker, &args.input, Some(&output))?;
            print_report(&report);
            println!("Wrote {} chunk(s) to {}", store.len()?, output.display());
        }
        None => {
            let (report, _) = run_pipeline(InMemoryStorage::new(), chunker, &args.input, None)?;
            print_report(&report);
        }
    }

    Ok(())
}

/// Ingest `input` into memory, index every chunk, and serve `POST /retrieve` on
/// `args.addr` until interrupted.
fn serve(args: ServeArgs) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let chunker =
        FixedSizeChunker::new(args.chunk_size, args.overlap)?.word_aware(!args.no_word_aware);
    let (report, store) = run_pipeline(InMemoryStorage::new(), chunker, &args.input, None)?;
    print_report(&report);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let retriever = Retriever::new(MockEmbeddingProvider::new(), InMemoryVectorStore::new());
        let indexed = retriever.index(store.into_chunks()).await?;
        println!("Indexed {indexed} chunk(s) for retrieval.");

        let app = Router::new()
            .route("/retrieve", post(retrieve_handler))
            .with_state(Arc::new(retriever));

        let listener = tokio::net::TcpListener::bind(args.addr).await?;
        // Report the *bound* address, which differs from the requested one when the
        // caller asked for port 0.
        println!(
            "Serving POST /retrieve on http://{}",
            listener.local_addr()?
        );
        axum::serve(listener, app).await?;
        Ok(())
    })
}

/// The `POST /retrieve` handler: deserialize a [`RetrievalRequest`], run it through the
/// [`Retriever`], and serialize the [`RetrievalResponse`] — mapping validation failures
/// to `400` and backend failures to `5xx`.
async fn retrieve_handler(
    State(retriever): State<Arc<ServedRetriever>>,
    Json(request): Json<RetrievalRequest>,
) -> Response {
    match retriever.handle(&request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(err) => {
            let status = match &err {
                RetrievalError::EmptyQuery | RetrievalError::QueryTooLong { .. } => {
                    StatusCode::BAD_REQUEST
                }
                RetrievalError::Embedding(_) | RetrievalError::EmbeddingCountMismatch { .. } => {
                    StatusCode::BAD_GATEWAY
                }
                RetrievalError::VectorStore(_) => StatusCode::INTERNAL_SERVER_ERROR,
            };
            let body = serde_json::json!({ "error": err.to_string() });
            (status, Json(body)).into_response()
        }
    }
}

/// Build a pipeline around `store`, ingest `input`, flush, and return the report along
/// with the (now populated) store. `exclude`, when set, is omitted from directory walks.
fn run_pipeline<S: ChunkStore>(
    store: S,
    chunker: FixedSizeChunker,
    input: &Path,
    exclude: Option<&Path>,
) -> Result<(IngestReport, S)> {
    let mut builder = IngestionPipeline::builder(store).chunker(chunker);
    if let Some(path) = exclude {
        builder = builder.exclude(path);
    }
    let mut pipeline = builder.build();
    let report = pipeline.ingest_path(input)?;
    pipeline.flush()?;
    Ok((report, pipeline.into_store()))
}

fn print_report(report: &IngestReport) {
    println!(
        "Ingested {} file(s), skipped {}, created {} chunk(s).",
        report.files_ingested, report.files_skipped, report.chunks_created
    );
    for file in &report.files {
        println!("  {} -> {} chunk(s)", file.path.display(), file.chunks);
    }
}
