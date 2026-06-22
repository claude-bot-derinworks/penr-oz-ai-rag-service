//! A JSON Lines (`.jsonl`) chunk store.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use super::ChunkStore;
use crate::document::Chunk;
use crate::error::{RagError, Result};

/// A [`ChunkStore`] that appends each chunk to a file as a single line of JSON
/// ([JSON Lines](https://jsonlines.org/)).
///
/// This keeps ingested output easy to inspect, diff, and stream into a downstream
/// indexer or vector store. Writes are buffered and flushed on [`flush`](ChunkStore::flush)
/// and when the store is dropped.
pub struct JsonlStorage {
    path: PathBuf,
    writer: BufWriter<File>,
    written: usize,
}

impl JsonlStorage {
    /// Open `path` for appending, creating the file (and any missing parent
    /// directories) if necessary.
    pub fn create(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();

        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|source| RagError::Io {
                    path: parent.to_path_buf(),
                    source,
                })?;
            }
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|source| RagError::Io {
                path: path.clone(),
                source,
            })?;

        Ok(Self {
            path,
            writer: BufWriter::new(file),
            written: 0,
        })
    }

    /// The path being written to.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl ChunkStore for JsonlStorage {
    fn store(&mut self, chunks: &[Chunk]) -> Result<()> {
        for chunk in chunks {
            let line = serde_json::to_string(chunk).map_err(|source| RagError::Serialization {
                id: chunk.id.clone(),
                source,
            })?;
            self.writer
                .write_all(line.as_bytes())
                .and_then(|()| self.writer.write_all(b"\n"))
                .map_err(|source| RagError::Io {
                    path: self.path.clone(),
                    source,
                })?;
            self.written += 1;
        }
        Ok(())
    }

    /// The number of chunks written through this instance (not counting lines that may
    /// already have existed in the file when it was opened).
    fn len(&self) -> Result<usize> {
        Ok(self.written)
    }

    fn flush(&mut self) -> Result<()> {
        self.writer.flush().map_err(|source| RagError::Io {
            path: self.path.clone(),
            source,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Chunk, ChunkMetadata, Metadata};
    use tempfile::tempdir;

    fn sample(index: usize) -> Chunk {
        Chunk {
            id: format!("doc#{index}"),
            content: format!("content {index}"),
            metadata: ChunkMetadata {
                source: "doc".to_string(),
                chunk_index: index,
                total_chunks: 2,
                start_char: index * 10,
                end_char: index * 10 + 9,
                extra: Metadata::new(),
            },
        }
    }

    #[test]
    fn writes_one_json_object_per_line_and_round_trips() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nested/chunks.jsonl");

        let mut store = JsonlStorage::create(&path).unwrap();
        store.store(&[sample(0), sample(1)]).unwrap();
        store.flush().unwrap();
        assert_eq!(store.len().unwrap(), 2);

        let body = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2);

        let restored: Vec<Chunk> = lines
            .iter()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(restored, vec![sample(0), sample(1)]);
    }

    #[test]
    fn appends_rather_than_truncates() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("chunks.jsonl");

        let mut first = JsonlStorage::create(&path).unwrap();
        first.store(&[sample(0)]).unwrap();
        first.flush().unwrap();
        drop(first);

        let mut second = JsonlStorage::create(&path).unwrap();
        second.store(&[sample(1)]).unwrap();
        second.flush().unwrap();

        let body = std::fs::read_to_string(&path).unwrap();
        assert_eq!(body.lines().count(), 2);
    }
}
