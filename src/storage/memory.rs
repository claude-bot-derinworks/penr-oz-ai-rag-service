//! An in-memory chunk store.

use super::ChunkStore;
use crate::document::Chunk;
use crate::error::Result;

/// A [`ChunkStore`] that keeps chunks in a `Vec`.
///
/// Handy as a default backend and in tests, and as a building block for callers that
/// want to post-process chunks before sending them elsewhere.
#[derive(Debug, Default, Clone)]
pub struct InMemoryStorage {
    chunks: Vec<Chunk>,
}

impl InMemoryStorage {
    /// Create an empty in-memory store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Borrow the stored chunks.
    pub fn chunks(&self) -> &[Chunk] {
        &self.chunks
    }

    /// Consume the store and return the accumulated chunks.
    pub fn into_chunks(self) -> Vec<Chunk> {
        self.chunks
    }
}

impl ChunkStore for InMemoryStorage {
    fn store(&mut self, chunks: &[Chunk]) -> Result<()> {
        self.chunks.extend_from_slice(chunks);
        Ok(())
    }

    fn len(&self) -> Result<usize> {
        Ok(self.chunks.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Chunk, ChunkMetadata, Metadata};

    fn sample(id: &str) -> Chunk {
        Chunk {
            id: id.to_string(),
            content: "content".to_string(),
            metadata: ChunkMetadata {
                source: "doc".to_string(),
                chunk_index: 0,
                total_chunks: 1,
                start_char: 0,
                end_char: 7,
                extra: Metadata::new(),
            },
        }
    }

    #[test]
    fn accumulates_across_calls() {
        let mut store = InMemoryStorage::new();
        assert!(store.is_empty().unwrap());

        store.store(&[sample("a#0")]).unwrap();
        store.store(&[sample("b#0"), sample("b#1")]).unwrap();

        assert_eq!(store.len().unwrap(), 3);
        assert!(!store.is_empty().unwrap());
        assert_eq!(store.into_chunks().len(), 3);
    }
}
