//! Core morphology and lossless segmentation engine.

#![forbid(unsafe_code)]

mod scanner;

pub use nedo_format::{ByteSpan, FormatError, LosslessDocument, SurfaceUnit};
pub use scanner::{
    scan, scan_compact, scan_lexical_spans, scan_lexical_spans_with_hints, CodeScanHints,
    CompactScanResult, HintedLexicalSpanIter, LexicalKind, LexicalSpanIter, ScanError, ScanResult,
};

/// Current morphology bundle schema.
pub const MORPH_BUNDLE_SCHEMA_VERSION: u32 = 1;

/// Creates a lossless, deliberately unanalyzed document.
///
/// This is an explicit state, not a morphology fallback disguised as success.
#[must_use]
pub const fn preserve(raw: Vec<u8>) -> LosslessDocument {
    LosslessDocument::unanalyzed(raw)
}
