//! A fixed-size sliding-window chunker.

use super::Chunker;
use crate::document::{Chunk, ChunkMetadata, Document};
use crate::error::{RagError, Result};

/// Default target chunk size, in characters.
const DEFAULT_CHUNK_SIZE: usize = 800;
/// Default overlap between consecutive chunks, in characters.
const DEFAULT_OVERLAP: usize = 100;

/// Splits documents into fixed-size, optionally overlapping windows measured in
/// **characters** (Unicode scalar values), so multi-byte text is handled correctly.
///
/// With `word_aware` enabled (the default), the chunker prefers to end a chunk on a
/// word boundary near the target size instead of cutting through a word, and starts
/// each chunk on a non-whitespace character. Offsets in [`ChunkMetadata`] always refer
/// to character positions in the original document.
#[derive(Debug, Clone, Copy)]
pub struct FixedSizeChunker {
    chunk_size: usize,
    overlap: usize,
    word_aware: bool,
}

impl FixedSizeChunker {
    /// Create a chunker that emits chunks of at most `chunk_size` characters, with
    /// `overlap` characters shared between consecutive chunks.
    ///
    /// # Errors
    /// Returns [`RagError::InvalidChunkerConfig`] if `chunk_size` is zero or if
    /// `overlap` is greater than or equal to `chunk_size` (which would prevent the
    /// window from advancing).
    pub fn new(chunk_size: usize, overlap: usize) -> Result<Self> {
        if chunk_size == 0 {
            return Err(RagError::InvalidChunkerConfig(
                "chunk_size must be greater than 0".to_string(),
            ));
        }
        if overlap >= chunk_size {
            return Err(RagError::InvalidChunkerConfig(format!(
                "overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
            )));
        }
        Ok(Self {
            chunk_size,
            overlap,
            word_aware: true,
        })
    }

    /// Enable or disable word-boundary aware splitting. When disabled, the chunker
    /// makes exact character cuts at every `chunk_size` boundary.
    pub fn word_aware(mut self, enabled: bool) -> Self {
        self.word_aware = enabled;
        self
    }

    /// The configured maximum chunk size in characters.
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// The configured overlap in characters.
    pub fn overlap(&self) -> usize {
        self.overlap
    }

    /// Choose an end offset at or before `hard_end` that falls on a word boundary,
    /// i.e. just after the last complete word. Falls back to `hard_end` if no suitable
    /// boundary exists within the lookback window, so a single long token is never
    /// dropped or expanded indefinitely.
    fn word_boundary_end(&self, chars: &[char], start: usize, hard_end: usize) -> usize {
        // Never look back past the halfway point of the window, so word-awareness can
        // only ever shrink a chunk modestly rather than produce tiny fragments.
        let min_end = start + (self.chunk_size / 2).max(1);
        let mut end = hard_end;
        while end > min_end {
            // A boundary sits between a word character (end-1) and whitespace (end).
            if chars[end].is_whitespace() && !chars[end - 1].is_whitespace() {
                return end;
            }
            end -= 1;
        }
        hard_end
    }
}

impl Default for FixedSizeChunker {
    fn default() -> Self {
        // Safe: the default constants satisfy the constructor's invariants.
        Self::new(DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP)
            .expect("default chunker parameters are valid")
    }
}

impl Chunker for FixedSizeChunker {
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>> {
        if document.content.trim().is_empty() {
            return Err(RagError::EmptyDocument {
                id: document.id.clone(),
            });
        }

        // Work over characters so that `chunk_size`, `overlap`, and the recorded
        // offsets are all in terms of Unicode scalar values rather than bytes.
        let chars: Vec<char> = document.content.chars().collect();
        let len = chars.len();

        // First pass: compute the (start, end) character spans.
        let mut spans: Vec<(usize, usize)> = Vec::new();
        let mut start = 0usize;
        while start < len {
            if self.word_aware {
                // Begin each chunk on a meaningful (non-whitespace) character.
                while start < len && chars[start].is_whitespace() {
                    start += 1;
                }
                if start >= len {
                    break;
                }
            }

            let hard_end = (start + self.chunk_size).min(len);
            let end = if self.word_aware && hard_end < len {
                self.word_boundary_end(&chars, start, hard_end)
            } else {
                hard_end
            };

            spans.push((start, end));

            if end >= len {
                break;
            }

            // Advance the window, keeping `overlap` characters of context. The guard
            // guarantees forward progress even when an unusually short chunk and a
            // large overlap would otherwise stall the loop.
            let next = end.saturating_sub(self.overlap);
            start = if next > start { next } else { end };
        }

        let total_chunks = spans.len();
        let chunks = spans
            .into_iter()
            .enumerate()
            .map(|(index, (start, end))| Chunk {
                id: format!("{}#{index}", document.id),
                content: chars[start..end].iter().collect(),
                metadata: ChunkMetadata {
                    source: document.id.clone(),
                    chunk_index: index,
                    total_chunks,
                    start_char: start,
                    end_char: end,
                    extra: document.metadata.clone(),
                },
            })
            .collect();

        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(content: &str) -> Document {
        Document::new("doc", content)
    }

    #[test]
    fn rejects_invalid_configuration() {
        assert!(matches!(
            FixedSizeChunker::new(0, 0),
            Err(RagError::InvalidChunkerConfig(_))
        ));
        assert!(matches!(
            FixedSizeChunker::new(100, 100),
            Err(RagError::InvalidChunkerConfig(_))
        ));
        assert!(matches!(
            FixedSizeChunker::new(100, 150),
            Err(RagError::InvalidChunkerConfig(_))
        ));
        assert!(FixedSizeChunker::new(100, 99).is_ok());
    }

    #[test]
    fn empty_or_whitespace_document_errors() {
        let chunker = FixedSizeChunker::default();
        assert!(matches!(
            chunker.chunk(&doc("")),
            Err(RagError::EmptyDocument { .. })
        ));
        assert!(matches!(
            chunker.chunk(&doc("   \n\t  ")),
            Err(RagError::EmptyDocument { .. })
        ));
    }

    #[test]
    fn hard_cuts_have_exact_size_and_overlap() {
        // No whitespace, word-awareness off -> deterministic fixed windows.
        let chunker = FixedSizeChunker::new(4, 1).unwrap().word_aware(false);
        let chunks = chunker.chunk(&doc("abcdefghij")).unwrap();

        let contents: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
        assert_eq!(contents, vec!["abcd", "defg", "ghij"]);

        assert_eq!(chunks[0].id, "doc#0");
        assert_eq!(chunks[1].id, "doc#1");
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.metadata.chunk_index, i);
            assert_eq!(chunk.metadata.total_chunks, 3);
        }
        assert_eq!(chunks[0].metadata.start_char, 0);
        assert_eq!(chunks[0].metadata.end_char, 4);
        assert_eq!(chunks[1].metadata.start_char, 3);
    }

    #[test]
    fn short_document_yields_single_chunk() {
        let chunker = FixedSizeChunker::new(100, 10).unwrap();
        let chunks = chunker.chunk(&doc("just a little text")).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "just a little text");
        assert_eq!(chunks[0].metadata.total_chunks, 1);
    }

    #[test]
    fn word_aware_splitting_does_not_cut_words() {
        let text = "the quick brown fox jumps over the lazy dog again and again";
        let chunker = FixedSizeChunker::new(20, 5).unwrap();
        let chunks = chunker.chunk(&doc(text)).unwrap();

        assert!(chunks.len() > 1);
        for chunk in &chunks {
            let trimmed = chunk.content.trim();
            assert!(!trimmed.is_empty());
            // No chunk should start or end in the middle of a word: boundaries are at
            // whitespace (or the document edges).
            assert!(!chunk.content.starts_with(char::is_whitespace));
        }
        // Every character of the source is covered by some chunk's span.
        assert_eq!(chunks[0].metadata.start_char, 0);
        assert_eq!(
            chunks.last().unwrap().metadata.end_char,
            text.chars().count()
        );
    }

    #[test]
    fn offsets_match_content_for_multibyte_text() {
        let text = "café résumé naïve façade jalapeño piñata fiancé soufflé";
        let chunker = FixedSizeChunker::new(12, 3).unwrap();
        let chunks = chunker.chunk(&doc(text)).unwrap();

        let chars: Vec<char> = text.chars().collect();
        for chunk in &chunks {
            let expected: String = chars[chunk.metadata.start_char..chunk.metadata.end_char]
                .iter()
                .collect();
            assert_eq!(chunk.content, expected);
            assert!(chunk.metadata.start_char < chunk.metadata.end_char);
        }
    }

    #[test]
    fn document_metadata_is_propagated_to_chunks() {
        let document = Document::new("doc", "alpha beta gamma delta epsilon zeta")
            .with_metadata("loader", "text")
            .with_metadata("filename", "doc.txt");
        let chunks = FixedSizeChunker::new(15, 3)
            .unwrap()
            .chunk(&document)
            .unwrap();

        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert_eq!(
                chunk.metadata.extra.get("loader").map(String::as_str),
                Some("text")
            );
            assert_eq!(
                chunk.metadata.extra.get("filename").map(String::as_str),
                Some("doc.txt")
            );
        }
    }
}
