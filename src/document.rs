//! Core data types that flow through the pipeline: [`Document`], [`Chunk`], and
//! their metadata.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Arbitrary, string-keyed metadata attached to documents and propagated to chunks.
///
/// A [`BTreeMap`] is used (rather than a `HashMap`) so that serialized output is
/// deterministic, which keeps snapshots and diffs stable.
pub type Metadata = BTreeMap<String, String>;

/// A raw document produced by a [`Loader`](crate::loader::Loader), before it is split
/// into [`Chunk`]s.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Document {
    /// Stable identifier for the document, typically its source path.
    pub id: String,
    /// The full text content of the document.
    pub content: String,
    /// Document-level metadata that is copied onto every chunk produced from it.
    #[serde(default)]
    pub metadata: Metadata,
}

impl Document {
    /// Create a document with the given identifier and content and no metadata.
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            metadata: Metadata::new(),
        }
    }

    /// Builder-style helper to attach a single metadata entry.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// A contiguous slice of a [`Document`], ready to be embedded and indexed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunk {
    /// Identifier of the form `{document_id}#{chunk_index}`.
    pub id: String,
    /// The chunk's text content.
    pub content: String,
    /// Provenance and positional metadata for this chunk.
    pub metadata: ChunkMetadata,
}

/// Metadata describing where a [`Chunk`] came from and where it sits in its document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Identifier of the document this chunk was derived from.
    pub source: String,
    /// Zero-based position of this chunk within the document.
    pub chunk_index: usize,
    /// Total number of chunks produced from the document.
    pub total_chunks: usize,
    /// Character offset (inclusive) of the chunk's start within the document.
    pub start_char: usize,
    /// Character offset (exclusive) of the chunk's end within the document.
    pub end_char: usize,
    /// Document-level metadata propagated from the source [`Document`].
    #[serde(default)]
    pub extra: Metadata,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_builder_collects_metadata() {
        let doc = Document::new("a.txt", "hello")
            .with_metadata("loader", "text")
            .with_metadata("lang", "en");

        assert_eq!(doc.id, "a.txt");
        assert_eq!(doc.content, "hello");
        assert_eq!(doc.metadata.get("loader").map(String::as_str), Some("text"));
        assert_eq!(doc.metadata.get("lang").map(String::as_str), Some("en"));
    }

    #[test]
    fn chunk_round_trips_through_json() {
        let chunk = Chunk {
            id: "a.txt#0".to_string(),
            content: "hello world".to_string(),
            metadata: ChunkMetadata {
                source: "a.txt".to_string(),
                chunk_index: 0,
                total_chunks: 1,
                start_char: 0,
                end_char: 11,
                extra: Metadata::from([("loader".to_string(), "text".to_string())]),
            },
        };

        let json = serde_json::to_string(&chunk).expect("serialize");
        let restored: Chunk = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(chunk, restored);
    }
}
