//! Splitting [`Document`]s into [`Chunk`]s.
//!
//! Chunking is deliberately decoupled from loading: the output of any
//! [`Loader`](crate::loader::Loader) can be fed to any [`Chunker`], and new chunking
//! strategies can be dropped in without touching loaders or storage.

mod fixed_size;

pub use fixed_size::FixedSizeChunker;

use crate::document::{Chunk, Document};
use crate::error::Result;

/// A strategy for partitioning a [`Document`] into an ordered list of [`Chunk`]s.
pub trait Chunker: Send + Sync {
    /// Split `document` into chunks, in document order.
    ///
    /// Implementations must return [`RagError::EmptyDocument`](crate::error::RagError::EmptyDocument)
    /// when the document has no chunkable content, and should populate each chunk's
    /// [`ChunkMetadata`](crate::document::ChunkMetadata) (indices, offsets, and the
    /// document's propagated metadata).
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>>;
}
