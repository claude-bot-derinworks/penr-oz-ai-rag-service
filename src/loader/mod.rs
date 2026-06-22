//! Loading raw sources into normalized [`Document`]s.
//!
//! A [`Loader`] turns a file of a particular format into a [`Document`]. Support for
//! new formats — PDF, HTML, Markdown, … — is added by implementing [`Loader`] and
//! registering it with a [`LoaderRegistry`]; nothing downstream of loading needs to
//! change. The registry routes each file to the loader registered for its extension.

mod text;

pub use text::TextLoader;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::document::Document;
use crate::error::{RagError, Result};

/// A source of [`Document`]s for a particular set of file formats.
///
/// Implementations are responsible for reading their input and normalizing it into
/// plain text. They must be cheap to share (`Send + Sync`) so a single instance can
/// back multiple registry entries.
pub trait Loader: Send + Sync {
    /// The lower-cased file extensions (without a leading dot) this loader handles,
    /// e.g. `["txt", "text"]`.
    fn extensions(&self) -> &[&str];

    /// Read the file at `path` and normalize it into a [`Document`].
    fn load(&self, path: &Path) -> Result<Document>;
}

/// Routes a file to the [`Loader`] registered for its extension.
///
/// Extensions are matched case-insensitively. Registering a loader that advertises an
/// already-registered extension overrides the previous entry for that extension.
#[derive(Clone, Default)]
pub struct LoaderRegistry {
    by_extension: HashMap<String, Arc<dyn Loader>>,
}

impl LoaderRegistry {
    /// Create an empty registry with no loaders.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry pre-populated with the built-in loaders.
    ///
    /// Currently this registers only [`TextLoader`]; future loaders will be added here.
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.register(Arc::new(TextLoader::new()));
        registry
    }

    /// Register `loader` for every extension it advertises.
    pub fn register(&mut self, loader: Arc<dyn Loader>) -> &mut Self {
        for ext in loader.extensions() {
            self.by_extension
                .insert(ext.to_ascii_lowercase(), Arc::clone(&loader));
        }
        self
    }

    /// The sorted list of extensions this registry can handle.
    pub fn supported_extensions(&self) -> Vec<String> {
        let mut extensions: Vec<String> = self.by_extension.keys().cloned().collect();
        extensions.sort();
        extensions
    }

    /// Whether a loader is registered for `path`'s extension.
    pub fn supports(&self, path: &Path) -> bool {
        self.loader_for(path).is_ok()
    }

    /// Look up the loader responsible for `path` based on its extension.
    ///
    /// # Errors
    /// Returns [`RagError::MissingExtension`] when `path` has no extension, or
    /// [`RagError::UnsupportedFormat`] when no loader is registered for it.
    pub fn loader_for(&self, path: &Path) -> Result<&dyn Loader> {
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| RagError::MissingExtension {
                path: path.to_path_buf(),
            })?
            .to_ascii_lowercase();

        self.by_extension
            .get(&extension)
            .map(Arc::as_ref)
            .ok_or_else(|| RagError::UnsupportedFormat {
                path: path.to_path_buf(),
                extension,
            })
    }

    /// Load `path` using whichever registered loader matches its extension.
    pub fn load(&self, path: &Path) -> Result<Document> {
        self.loader_for(path)?.load(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn defaults_support_text_extensions() {
        let registry = LoaderRegistry::with_defaults();
        assert_eq!(
            registry.supported_extensions(),
            vec!["text".to_string(), "txt".to_string()]
        );
        assert!(registry.supports(Path::new("notes.TXT")));
        assert!(!registry.supports(Path::new("notes.pdf")));
    }

    #[test]
    fn missing_extension_is_reported() {
        // `loader_for` returns `&dyn Loader` on success, which is not `Debug`, so match
        // on the result directly rather than calling `unwrap_err`.
        let registry = LoaderRegistry::with_defaults();
        assert!(matches!(
            registry.loader_for(Path::new("README")),
            Err(RagError::MissingExtension { .. })
        ));
    }

    #[test]
    fn unsupported_extension_is_reported() {
        let registry = LoaderRegistry::with_defaults();
        match registry.loader_for(Path::new("paper.pdf")) {
            Err(RagError::UnsupportedFormat { extension, path }) => {
                assert_eq!(extension, "pdf");
                assert_eq!(path, PathBuf::from("paper.pdf"));
            }
            _ => panic!("expected UnsupportedFormat error"),
        }
    }
}
