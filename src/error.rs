//! Error types for the ingestion pipeline.
//!
//! Every fallible operation returns [`Result`], whose error half is [`RagError`].
//! The variants are intentionally specific so that invalid inputs (missing files,
//! unsupported formats, non-UTF-8 data, bad configuration, …) produce actionable,
//! human-readable messages rather than opaque failures.

use std::path::PathBuf;

use thiserror::Error;

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, RagError>;

/// The set of errors the ingestion pipeline can produce.
#[derive(Debug, Error)]
pub enum RagError {
    /// An underlying I/O operation failed (reading a source file, writing the store, …).
    #[error("failed to access `{path}`: {source}")]
    Io {
        /// The path the operation was acting on.
        path: PathBuf,
        /// The originating I/O error.
        #[source]
        source: std::io::Error,
    },

    /// The file has an extension, but no loader is registered to handle it.
    #[error("unsupported document format `.{extension}` for `{path}`: no loader is registered (run with a registered loader or convert the file first)")]
    UnsupportedFormat {
        /// The offending source path.
        path: PathBuf,
        /// The lower-cased extension that could not be handled.
        extension: String,
    },

    /// The file has no extension, so its format cannot be determined.
    #[error("cannot determine the document format of `{path}`: the file has no extension")]
    MissingExtension {
        /// The offending source path.
        path: PathBuf,
    },

    /// A text loader was handed bytes that are not valid UTF-8.
    #[error("`{path}` is not valid UTF-8 text: {source}")]
    InvalidUtf8 {
        /// The offending source path.
        path: PathBuf,
        /// The originating decoding error.
        #[source]
        source: std::str::Utf8Error,
    },

    /// A document contained no usable (non-whitespace) content to chunk.
    #[error("document `{id}` has no textual content to ingest")]
    EmptyDocument {
        /// Identifier of the empty document.
        id: String,
    },

    /// A chunker was constructed with parameters that cannot produce valid chunks.
    #[error("invalid chunker configuration: {0}")]
    InvalidChunkerConfig(String),

    /// A chunk could not be serialized for persistence.
    #[error("failed to serialize chunk `{id}`: {source}")]
    Serialization {
        /// Identifier of the chunk that failed to serialize.
        id: String,
        /// The originating serialization error.
        #[source]
        source: serde_json::Error,
    },

    /// A storage backend reported a failure that is not an I/O or serialization error.
    #[error("storage backend error: {0}")]
    Storage(String),
}
