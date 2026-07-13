//! Command-line front-end for the RAG service.
//!
//! ```text
//! # Count chunks from a file in memory
//! penr-oz-rag ingest ./docs/notes.txt
//!
//! # Ingest a directory and write chunks as JSON Lines
//! penr-oz-rag ingest ./docs --output chunks.jsonl --chunk-size 800 --overlap 100
//!
//! # Ingest a directory and serve retrieval + answer generation over it
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
    AnswerGenerator, AnswerRequest, ChunkStore, FixedSizeChunker, GenerationError, InMemoryStorage,
    InMemoryVectorStore, IngestReport, IngestionPipeline, JsonlStorage, MockEmbeddingProvider,
    MockLlmProvider, Result, RetrievalError, RetrievalRequest, Retriever, DEFAULT_MIN_SCORE,
};

/// The answer generator the `serve` command hosts — wrapping the retriever it also
/// serves — with the in-process embedding provider, vector store, and LLM, shared
/// across request handlers.
///
/// `MockEmbeddingProvider` and `MockLlmProvider` are the only providers in the crate
/// today, so served similarity is deterministic-hash based rather than semantic and the
/// served "answer" is the mock's echo of the grounded prompt; real providers drop in
/// here once they exist, without touching the handlers.
type ServedGenerator = AnswerGenerator<MockEmbeddingProvider, InMemoryVectorStore, MockLlmProvider>;

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
    /// Ingest a file or directory, index it, and serve `POST /retrieve` and
    /// `POST /answer` over HTTP.
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

    /// Minimum similarity score a retrieved chunk must reach to be used as answer
    /// context. Requests can override it per call via `min_score`.
    #[arg(long, default_value_t = DEFAULT_MIN_SCORE)]
    min_score: f32,
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

/// Ingest `input` into memory, index every chunk, and serve `POST /retrieve` and
/// `POST /answer` on `args.addr` until interrupted.
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

        let generator =
            AnswerGenerator::new(retriever, MockLlmProvider::new()).with_min_score(args.min_score);

        let app = Router::new()
            .route("/retrieve", post(retrieve_handler))
            .route("/answer", post(answer_handler))
            .with_state(Arc::new(generator));

        let listener = tokio::net::TcpListener::bind(args.addr).await?;
        // Report the *bound* address, which differs from the requested one when the
        // caller asked for port 0.
        let addr = listener.local_addr()?;
        println!("Serving POST /retrieve on http://{addr}");
        println!("Serving POST /answer on http://{addr}");
        axum::serve(listener, app).await?;
        Ok(())
    })
}

/// The `POST /retrieve` handler: deserialize a [`RetrievalRequest`], run it through the
/// generator's [`Retriever`], and serialize the [`RetrievalResponse`] — mapping
/// validation failures to `400` and backend failures to `5xx`.
async fn retrieve_handler(
    State(generator): State<Arc<ServedGenerator>>,
    Json(request): Json<RetrievalRequest>,
) -> Response {
    match generator.retriever().handle(&request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(err) => error_response(retrieval_status(&err), &err),
    }
}

/// The `POST /answer` handler: deserialize an [`AnswerRequest`], run it through the
/// [`AnswerGenerator`], and serialize the [`AnswerResponse`] — mapping validation
/// failures to `400` and backend (embedding or LLM) failures to `5xx`.
async fn answer_handler(
    State(generator): State<Arc<ServedGenerator>>,
    Json(request): Json<AnswerRequest>,
) -> Response {
    match generator.handle(&request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(err) => {
            let status = match &err {
                GenerationError::Retrieval(err) => retrieval_status(err),
                GenerationError::Llm(_) => StatusCode::BAD_GATEWAY,
            };
            error_response(status, &err)
        }
    }
}

/// Map a [`RetrievalError`] onto the HTTP status both endpoints use for it: validation
/// failures are the client's fault (`400`), backend failures are not (`5xx`).
fn retrieval_status(err: &RetrievalError) -> StatusCode {
    match err {
        RetrievalError::EmptyQuery | RetrievalError::QueryTooLong { .. } => StatusCode::BAD_REQUEST,
        RetrievalError::Embedding(_) | RetrievalError::EmbeddingCountMismatch { .. } => {
            StatusCode::BAD_GATEWAY
        }
        RetrievalError::VectorStore(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// Serialize `err` into the JSON error body both endpoints share.
fn error_response(status: StatusCode, err: &dyn std::error::Error) -> Response {
    let body = serde_json::json!({ "error": err.to_string() });
    (status, Json(body)).into_response()
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
