//! The end-to-end ingestion pipeline.
//!
//! [`IngestionPipeline`] wires a [`LoaderRegistry`], a [`Chunker`], and a
//! [`ChunkStore`] together: it loads each input file, splits it into chunks, and
//! persists them. Build one with [`IngestionPipeline::builder`].

use std::path::{Path, PathBuf};

use crate::chunker::{Chunker, FixedSizeChunker};
use crate::error::{RagError, Result};
use crate::loader::LoaderRegistry;
use crate::storage::ChunkStore;

/// Per-file outcome recorded in an [`IngestReport`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileReport {
    /// The file that was ingested.
    pub path: PathBuf,
    /// How many chunks it produced.
    pub chunks: usize,
}

/// A summary of one ingestion run.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct IngestReport {
    /// Number of files successfully ingested.
    pub files_ingested: usize,
    /// Number of files skipped because no loader was registered for their format
    /// (only relevant when ingesting a directory).
    pub files_skipped: usize,
    /// Total number of chunks created across all ingested files.
    pub chunks_created: usize,
    /// Per-file breakdown, in ingestion order.
    pub files: Vec<FileReport>,
}

/// Loads, chunks, and stores documents.
///
/// The store type `S` is a generic parameter so callers can recover the concrete
/// backend afterwards (e.g. pull chunks back out of an [`InMemoryStorage`]
/// (crate::storage::InMemoryStorage)).
pub struct IngestionPipeline<S: ChunkStore> {
    loaders: LoaderRegistry,
    chunker: Box<dyn Chunker>,
    store: S,
    excludes: Vec<PathBuf>,
}

impl<S: ChunkStore> IngestionPipeline<S> {
    /// Start building a pipeline that writes to `store`.
    pub fn builder(store: S) -> PipelineBuilder<S> {
        PipelineBuilder::new(store)
    }

    /// Ingest a single file: load it, chunk it, and store the chunks. Returns the
    /// number of chunks produced.
    ///
    /// # Errors
    /// Propagates loader errors (unsupported format, unreadable or non-UTF-8 file),
    /// chunker errors (empty document), and storage errors.
    pub fn ingest_file(&mut self, path: impl AsRef<Path>) -> Result<usize> {
        let path = path.as_ref();
        let document = self.loaders.load(path)?;
        let chunks = self.chunker.chunk(&document)?;
        self.store.store(&chunks)?;
        Ok(chunks.len())
    }

    /// Ingest a file or a directory.
    ///
    /// When `path` is a directory it is walked recursively; files whose format has no
    /// registered loader are skipped and counted in [`IngestReport::files_skipped`].
    /// When `path` is a single file, an unsupported format is an error rather than a
    /// skip, since the caller asked for that file explicitly.
    pub fn ingest_path(&mut self, path: impl AsRef<Path>) -> Result<IngestReport> {
        let path = path.as_ref();
        if path.is_dir() {
            self.ingest_dir(path)
        } else {
            let chunks = self.ingest_file(path)?;
            Ok(IngestReport {
                files_ingested: 1,
                files_skipped: 0,
                chunks_created: chunks,
                files: vec![FileReport {
                    path: path.to_path_buf(),
                    chunks,
                }],
            })
        }
    }

    fn ingest_dir(&mut self, dir: &Path) -> Result<IngestReport> {
        let mut files = Vec::new();
        collect_files(dir, &mut files)?;

        // Resolve excluded paths to their canonical form once so the comparison is
        // robust to relative-vs-absolute and `.`/symlink differences. This is what
        // keeps the pipeline from ingesting its own output file when that file lives
        // inside the directory being ingested.
        let excluded: Vec<PathBuf> = self
            .excludes
            .iter()
            .filter_map(|path| std::fs::canonicalize(path).ok())
            .collect();

        let mut report = IngestReport::default();
        for path in files {
            if is_excluded(&path, &excluded) {
                continue;
            }
            if !self.loaders.supports(&path) {
                report.files_skipped += 1;
                continue;
            }
            let chunks = self.ingest_file(&path)?;
            report.files_ingested += 1;
            report.chunks_created += chunks;
            report.files.push(FileReport { path, chunks });
        }
        Ok(report)
    }

    /// Flush the underlying store's buffered writes.
    pub fn flush(&mut self) -> Result<()> {
        self.store.flush()
    }

    /// Borrow the underlying store.
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Consume the pipeline and return the underlying store.
    pub fn into_store(self) -> S {
        self.store
    }
}

/// Whether `path` resolves to one of the already-canonicalized `excluded` paths.
fn is_excluded(path: &Path, excluded: &[PathBuf]) -> bool {
    match std::fs::canonicalize(path) {
        Ok(canonical) => excluded.contains(&canonical),
        Err(_) => false,
    }
}

/// Recursively collect the files under `dir` in a deterministic (sorted) order.
fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|source| RagError::Io {
            path: dir.to_path_buf(),
            source,
        })?
        .map(|entry| {
            entry.map(|e| e.path()).map_err(|source| RagError::Io {
                path: dir.to_path_buf(),
                source,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    entries.sort();

    for path in entries {
        // Inspect the entry itself via `symlink_metadata` rather than `is_dir`, which
        // follows symlinks. Recursing into a symlinked directory could loop forever on
        // a cyclic link (e.g. one pointing back at a parent) and overflow the stack.
        let metadata = std::fs::symlink_metadata(&path).map_err(|source| RagError::Io {
            path: path.clone(),
            source,
        })?;
        if metadata.is_dir() {
            collect_files(&path, out)?;
        } else if metadata.is_file() {
            out.push(path);
        } else if metadata.is_symlink() {
            // Follow symlinks that resolve to regular files, but never recurse into a
            // symlinked directory.
            if std::fs::metadata(&path)
                .map(|m| m.is_file())
                .unwrap_or(false)
            {
                out.push(path);
            }
        }
    }
    Ok(())
}

/// Builder for [`IngestionPipeline`].
///
/// Defaults to the built-in loaders ([`LoaderRegistry::with_defaults`]) and a default
/// [`FixedSizeChunker`] when not overridden.
pub struct PipelineBuilder<S: ChunkStore> {
    loaders: Option<LoaderRegistry>,
    chunker: Option<Box<dyn Chunker>>,
    store: S,
    excludes: Vec<PathBuf>,
}

impl<S: ChunkStore> PipelineBuilder<S> {
    /// Create a builder targeting `store`.
    pub fn new(store: S) -> Self {
        Self {
            loaders: None,
            chunker: None,
            store,
            excludes: Vec::new(),
        }
    }

    /// Use a custom loader registry instead of the defaults.
    pub fn loaders(mut self, registry: LoaderRegistry) -> Self {
        self.loaders = Some(registry);
        self
    }

    /// Use a custom chunking strategy instead of the default [`FixedSizeChunker`].
    pub fn chunker(mut self, chunker: impl Chunker + 'static) -> Self {
        self.chunker = Some(Box::new(chunker));
        self
    }

    /// Exclude `path` from directory ingestion (matched by canonical path).
    ///
    /// Typically used to stop the pipeline from ingesting its own output file when
    /// that file is written inside the directory being ingested.
    pub fn exclude(mut self, path: impl Into<PathBuf>) -> Self {
        self.excludes.push(path.into());
        self
    }

    /// Finish building the pipeline.
    pub fn build(self) -> IngestionPipeline<S> {
        IngestionPipeline {
            loaders: self.loaders.unwrap_or_else(LoaderRegistry::with_defaults),
            chunker: self
                .chunker
                .unwrap_or_else(|| Box::new(FixedSizeChunker::default())),
            store: self.store,
            excludes: self.excludes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::InMemoryStorage;
    use tempfile::tempdir;

    #[test]
    fn ingests_a_single_file_into_memory() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("doc.txt");
        std::fs::write(&path, "alpha beta gamma delta epsilon zeta eta theta").unwrap();

        let chunker = FixedSizeChunker::new(20, 5).unwrap();
        let mut pipeline = IngestionPipeline::builder(InMemoryStorage::new())
            .chunker(chunker)
            .build();

        let report = pipeline.ingest_path(&path).unwrap();
        assert_eq!(report.files_ingested, 1);
        assert!(report.chunks_created >= 1);
        assert_eq!(report.files[0].path, path);

        let chunks = pipeline.into_store().into_chunks();
        assert_eq!(chunks.len(), report.chunks_created);
        // Provenance metadata from the loader is present on the chunks.
        assert_eq!(
            chunks[0].metadata.extra.get("filename").map(String::as_str),
            Some("doc.txt")
        );
    }

    #[test]
    fn directory_ingestion_skips_unsupported_formats() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "first document body here").unwrap();
        std::fs::write(dir.path().join("b.text"), "second document body here").unwrap();
        std::fs::write(dir.path().join("c.pdf"), "not actually a pdf").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/d.txt"), "nested document body here").unwrap();

        let mut pipeline = IngestionPipeline::builder(InMemoryStorage::new()).build();
        let report = pipeline.ingest_path(dir.path()).unwrap();

        assert_eq!(report.files_ingested, 3); // a.txt, b.text, sub/d.txt
        assert_eq!(report.files_skipped, 1); // c.pdf
        assert!(report.chunks_created >= 3);
    }

    #[test]
    fn single_unsupported_file_is_an_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("paper.pdf");
        std::fs::write(&path, "content").unwrap();

        let mut pipeline = IngestionPipeline::builder(InMemoryStorage::new()).build();
        let err = pipeline.ingest_path(&path).unwrap_err();
        assert!(matches!(err, RagError::UnsupportedFormat { .. }));
    }

    #[cfg(unix)]
    #[test]
    fn directory_walk_handles_symlinks_without_cycling() {
        use std::os::unix::fs::symlink;

        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "real document body here").unwrap();

        // A symlink to a regular file is still ingested.
        std::fs::write(dir.path().join("target.txt"), "link target body here").unwrap();
        symlink(dir.path().join("target.txt"), dir.path().join("link.txt")).unwrap();

        // A circular directory symlink (sub/loop -> the root) must not be followed,
        // otherwise the recursive walk would loop forever and overflow the stack.
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        symlink(dir.path(), sub.join("loop")).unwrap();

        let mut pipeline = IngestionPipeline::builder(InMemoryStorage::new()).build();
        let report = pipeline.ingest_path(dir.path()).unwrap(); // terminates

        // a.txt, link.txt, target.txt — the symlinked directory is skipped.
        assert_eq!(report.files_ingested, 3);
    }

    #[test]
    fn excluded_output_file_is_not_ingested() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("input.txt"), "real input body here").unwrap();
        // An (empty) output file living inside the ingested directory, as would be
        // created when `--output` points at a `.txt` under the input directory.
        let output = dir.path().join("chunks.txt");
        std::fs::write(&output, "").unwrap();

        let mut pipeline = IngestionPipeline::builder(InMemoryStorage::new())
            .exclude(&output)
            .build();
        // Without the exclusion, the empty `chunks.txt` would trigger EmptyDocument.
        let report = pipeline.ingest_path(dir.path()).unwrap();

        assert_eq!(report.files_ingested, 1); // only input.txt
    }
}
