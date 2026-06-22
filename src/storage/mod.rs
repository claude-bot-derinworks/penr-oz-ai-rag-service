//! Persisting [`Chunk`]s to a storage backend.
//!
//! The pipeline is agnostic to where chunks end up: it only depends on the
//! [`ChunkStore`] trait. Swapping the destination — an in-memory buffer for tests, a
//! JSON Lines file for inspection, or a vector database later — is a matter of
//! providing a different implementation.

mod jsonl;
mod memory;

pub use jsonl::JsonlStorage;
pub use memory::InMemoryStorage;

use crate::document::Chunk;
use crate::error::Result;

/// A destination for ingested [`Chunk`]s.
pub trait ChunkStore {
    /// Persist a batch of chunks. Implementations should be append-friendly so that
    /// ingesting many files accumulates rather than replaces.
    fn store(&mut self, chunks: &[Chunk]) -> Result<()>;

    /// The number of chunks written through this store.
    fn len(&self) -> Result<usize>;

    /// Whether the store has had any chunks written to it.
    fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Flush any buffered writes to the underlying medium. The default is a no-op for
    /// stores that do not buffer.
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}
