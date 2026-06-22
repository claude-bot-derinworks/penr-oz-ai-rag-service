//! Plain-text loader.

use std::path::Path;

use super::Loader;
use crate::document::Document;
use crate::error::{RagError, Result};

/// Extensions handled by [`TextLoader`].
const TEXT_EXTENSIONS: &[&str] = &["txt", "text"];

/// Loads UTF-8 encoded plain-text files (`.txt`, `.text`).
///
/// The whole file is read into memory and validated as UTF-8. The resulting
/// [`Document`] is tagged with `loader = "text"` and, when available, the original
/// `filename`, so that provenance survives onto every chunk.
#[derive(Debug, Default, Clone, Copy)]
pub struct TextLoader;

impl TextLoader {
    /// Create a new text loader.
    pub fn new() -> Self {
        Self
    }
}

impl Loader for TextLoader {
    fn extensions(&self) -> &[&str] {
        TEXT_EXTENSIONS
    }

    fn load(&self, path: &Path) -> Result<Document> {
        // Read raw bytes so we can surface a precise UTF-8 error rather than the
        // generic `InvalidData` that `read_to_string` would produce. Reading a
        // directory or a missing file also yields a clear I/O message here.
        let bytes = std::fs::read(path).map_err(|source| RagError::Io {
            path: path.to_path_buf(),
            source,
        })?;

        let content = std::str::from_utf8(&bytes)
            .map_err(|source| RagError::InvalidUtf8 {
                path: path.to_path_buf(),
                source,
            })?
            .to_owned();

        let mut document = Document::new(path.to_string_lossy().into_owned(), content)
            .with_metadata("loader", "text");

        if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
            document = document.with_metadata("filename", name);
        }

        Ok(document)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};

    #[test]
    fn loads_utf8_text_with_provenance_metadata() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("note.txt");
        std::fs::write(&path, "héllo, world").unwrap();

        let document = TextLoader::new().load(&path).unwrap();

        assert_eq!(document.content, "héllo, world");
        assert_eq!(
            document.metadata.get("loader").map(String::as_str),
            Some("text")
        );
        assert_eq!(
            document.metadata.get("filename").map(String::as_str),
            Some("note.txt")
        );
    }

    #[test]
    fn rejects_non_utf8_bytes() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&[0xff, 0xfe, 0x00]).unwrap();

        let err = TextLoader::new().load(file.path()).unwrap_err();
        assert!(matches!(err, RagError::InvalidUtf8 { .. }));
    }

    #[test]
    fn missing_file_yields_io_error() {
        let err = TextLoader::new()
            .load(Path::new("/this/path/does/not/exist.txt"))
            .unwrap_err();
        assert!(matches!(err, RagError::Io { .. }));
    }
}
