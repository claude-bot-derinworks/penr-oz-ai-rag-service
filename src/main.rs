//! Command-line front-end for the ingestion pipeline.
//!
//! ```text
//! # Count chunks from a file in memory
//! penr-oz-rag ingest ./docs/notes.txt
//!
//! # Ingest a directory and write chunks as JSON Lines
//! penr-oz-rag ingest ./docs --output chunks.jsonl --chunk-size 800 --overlap 100
//! ```

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};

use penr_oz_ai_rag_service::{
    ChunkStore, FixedSizeChunker, InMemoryStorage, IngestReport, IngestionPipeline, JsonlStorage,
    Result,
};

/// Document ingestion pipeline for a RAG service.
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

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Ingest(args) => ingest(args),
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
