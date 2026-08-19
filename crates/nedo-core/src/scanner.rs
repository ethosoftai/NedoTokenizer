//! Deterministic byte-general lexical scanning.
//!
//! The scanner never normalizes, drops, or replaces input. Every output unit is
//! a half-open byte span into the original buffer, and units cover the complete
//! document without gaps or overlaps.

use core::fmt;

use nedo_format::{ByteSpan, FormatError, LosslessDocument, SurfaceUnit};

/// Stable coarse lexical classes produced before morphological analysis.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum LexicalKind {
    /// Spaces, tabs, and non-line-breaking Unicode whitespace.
    Whitespace = 1,
    /// CR, LF, CRLF, NEL, or Unicode line/paragraph separators.
    LineBreak = 2,
    /// Letter-led word or identifier-like text.
    Word = 3,
    /// Numeric text, including internal date/decimal separators.
    Number = 4,
    /// Punctuation characters and punctuation runs.
    Punctuation = 5,
    /// Mathematical, currency, emoji, and other printable symbols.
    Symbol = 6,
    /// Valid Unicode or ASCII control characters other than line breaks.
    Control = 7,
    /// Bytes that are not part of a valid UTF-8 scalar sequence.
    Opaque = 8,
}

/// Cheap code-presence hints collected during the lexical scan.
///
/// These hints never classify spans by themselves. They only decide whether the
/// tokenizer must run its stricter line-level code detector after scanning.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CodeScanHints {
    may_contain_unmarked_code: bool,
    may_contain_inline_artifact: bool,
    has_backtick: bool,
}

impl CodeScanHints {
    /// Whether at least one high-confidence unmarked-code signal occurred.
    #[must_use]
    pub const fn may_contain_unmarked_code(self) -> bool {
        self.may_contain_unmarked_code
    }

    /// Whether a URL, e-mail, assignment, timestamp, or technical literal may occur.
    #[must_use]
    pub const fn may_contain_inline_artifact(self) -> bool {
        self.may_contain_inline_artifact
    }

    /// Whether a backtick occurred anywhere in the document.
    #[must_use]
    pub const fn has_backtick(self) -> bool {
        self.has_backtick
    }
}

/// A validated lexical scan over an exact byte document.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScanResult {
    document: LosslessDocument,
    kinds: Vec<LexicalKind>,
    code_hints: CodeScanHints,
}

impl ScanResult {
    /// Returns the exact lossless document and its contiguous units.
    #[must_use]
    pub const fn document(&self) -> &LosslessDocument {
        &self.document
    }

    /// Returns one lexical class per document unit.
    #[must_use]
    pub fn kinds(&self) -> &[LexicalKind] {
        &self.kinds
    }

    /// Returns code-presence hints collected without a second document scan.
    #[must_use]
    pub const fn code_hints(&self) -> CodeScanHints {
        self.code_hints
    }

    /// Returns the lexical class at `index`.
    ///
    /// # Errors
    ///
    /// Returns [`ScanError::MissingUnit`] when `index` is out of bounds.
    pub fn kind(&self, index: usize) -> Result<LexicalKind, ScanError> {
        self.kinds
            .get(index)
            .copied()
            .ok_or(ScanError::MissingUnit { index })
    }

    /// Returns the exact bytes for a lexical unit.
    ///
    /// # Errors
    ///
    /// Returns [`ScanError::MissingUnit`] for an invalid index, or a wrapped
    /// format error if the stored span cannot be sliced.
    pub fn unit_bytes(&self, index: usize) -> Result<&[u8], ScanError> {
        let unit = self
            .document
            .units()
            .get(index)
            .ok_or(ScanError::MissingUnit { index })?;
        self.document.slice(unit.span).map_err(ScanError::Format)
    }

    /// Consumes the scan and returns the lossless span document.
    #[must_use]
    pub fn into_document(self) -> LosslessDocument {
        self.document
    }

    /// Checks metadata cardinality and exact contiguous byte coverage.
    ///
    /// # Errors
    ///
    /// Returns an error if kinds and units differ in length, any unit begins at
    /// an unexpected offset, or the units do not cover the entire document.
    pub fn validate(&self) -> Result<(), ScanError> {
        self.document.validate().map_err(ScanError::Format)?;
        let units = self.document.units();
        if units.len() != self.kinds.len() {
            return Err(ScanError::MetadataLengthMismatch {
                unit_count: units.len(),
                kind_count: self.kinds.len(),
            });
        }

        let mut expected_start = 0_u64;
        for (index, unit) in units.iter().enumerate() {
            if unit.span.start != expected_start {
                return Err(ScanError::DiscontinuousCoverage {
                    index,
                    expected_start,
                    actual_start: unit.span.start,
                });
            }
            expected_start = unit.span.end;
        }

        let document_len =
            u64::try_from(self.document.decode().len()).map_err(|_| ScanError::LengthOverflow {
                field: "document_len",
            })?;
        if expected_start != document_len {
            return Err(ScanError::IncompleteCoverage {
                covered_until: expected_start,
                document_len,
            });
        }
        Ok(())
    }
}

/// Allocation-free iterator over exact lexical spans.
///
/// This runs the same classifier and grouping rules as [`scan`] without
/// building metadata vectors or collecting code-presence hints.
pub struct LexicalSpanIter<'a> {
    raw: &'a [u8],
    index: usize,
    failed: bool,
}

impl Iterator for LexicalSpanIter<'_> {
    type Item = Result<(ByteSpan, LexicalKind), ScanError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed || self.index >= self.raw.len() {
            return None;
        }
        let result = (|| {
            let start = self.index;
            let (kind, width) = classify_at(self.raw, self.index);
            self.index = self
                .index
                .checked_add(width)
                .ok_or(ScanError::LengthOverflow {
                    field: "scan_index",
                })?;
            self.index = match kind {
                LexicalKind::Word => consume_word(self.raw, self.index)?,
                LexicalKind::Number => consume_number(self.raw, self.index)?,
                LexicalKind::Whitespace
                | LexicalKind::LineBreak
                | LexicalKind::Punctuation
                | LexicalKind::Symbol
                | LexicalKind::Control
                | LexicalKind::Opaque => consume_same_kind(self.raw, self.index, kind)?,
            };
            let start = u64::try_from(start).map_err(|_| ScanError::LengthOverflow {
                field: "span_start",
            })?;
            let end = u64::try_from(self.index)
                .map_err(|_| ScanError::LengthOverflow { field: "span_end" })?;
            Ok((ByteSpan { start, end }, kind))
        })();
        if result.is_err() {
            self.failed = true;
        }
        Some(result)
    }
}

/// Streams exact lexical spans without allocating a scan result.
#[must_use]
pub const fn scan_lexical_spans(raw: &[u8]) -> LexicalSpanIter<'_> {
    LexicalSpanIter {
        raw,
        index: 0,
        failed: false,
    }
}

/// Allocation-free lexical iterator that also accumulates exact code-presence hints.
pub struct HintedLexicalSpanIter<'a> {
    raw: &'a [u8],
    index: usize,
    failed: bool,
    tracker: CodeHintTracker,
}

impl HintedLexicalSpanIter<'_> {
    /// Consumes the fully-drained iterator and returns its code-presence hints.
    #[must_use]
    pub fn into_hints(self) -> CodeScanHints {
        self.tracker.finish()
    }
}

impl Iterator for HintedLexicalSpanIter<'_> {
    type Item = Result<(ByteSpan, LexicalKind), ScanError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed || self.index >= self.raw.len() {
            return None;
        }
        let result = (|| {
            let start = self.index;
            let (kind, width) = classify_at(self.raw, self.index);
            self.index = self
                .index
                .checked_add(width)
                .ok_or(ScanError::LengthOverflow {
                    field: "scan_index",
                })?;
            self.index = match kind {
                LexicalKind::Word => consume_word(self.raw, self.index)?,
                LexicalKind::Number => consume_number(self.raw, self.index)?,
                LexicalKind::Whitespace
                | LexicalKind::LineBreak
                | LexicalKind::Punctuation
                | LexicalKind::Symbol
                | LexicalKind::Control
                | LexicalKind::Opaque => consume_same_kind(self.raw, self.index, kind)?,
            };
            self.tracker.observe(kind, &self.raw[start..self.index]);
            let start = u64::try_from(start).map_err(|_| ScanError::LengthOverflow {
                field: "span_start",
            })?;
            let end = u64::try_from(self.index)
                .map_err(|_| ScanError::LengthOverflow { field: "span_end" })?;
            Ok((ByteSpan { start, end }, kind))
        })();
        if result.is_err() {
            self.failed = true;
        }
        Some(result)
    }
}

/// Streams lexical spans while collecting the same code hints as [`scan_compact`].
#[must_use]
pub fn scan_lexical_spans_with_hints(raw: &[u8]) -> HintedLexicalSpanIter<'_> {
    HintedLexicalSpanIter {
        raw,
        index: 0,
        failed: false,
        tracker: CodeHintTracker::default(),
    }
}

/// Compact lexical scan used by flat production encoders.
///
/// Unlike [`ScanResult`], this representation stores spans directly and does
/// not allocate empty analysis vectors or a rich lossless document graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompactScanResult {
    raw: Vec<u8>,
    spans: Vec<ByteSpan>,
    kinds: Vec<LexicalKind>,
    code_hints: CodeScanHints,
}

impl CompactScanResult {
    /// Returns the exact input bytes.
    #[must_use]
    pub fn raw(&self) -> &[u8] {
        &self.raw
    }

    /// Consumes the compact scan and returns the exact original bytes.
    #[must_use]
    pub fn into_raw(self) -> Vec<u8> {
        self.raw
    }

    /// Returns contiguous lexical spans over the input.
    #[must_use]
    pub fn spans(&self) -> &[ByteSpan] {
        &self.spans
    }

    /// Returns one lexical kind per span.
    #[must_use]
    pub fn kinds(&self) -> &[LexicalKind] {
        &self.kinds
    }

    /// Returns code-presence hints collected during scanning.
    #[must_use]
    pub const fn code_hints(&self) -> CodeScanHints {
        self.code_hints
    }

    /// Returns exact bytes for one lexical span.
    ///
    /// # Errors
    ///
    /// Returns [`ScanError::MissingUnit`] when `index` is invalid.
    pub fn unit_bytes(&self, index: usize) -> Result<&[u8], ScanError> {
        let span = self
            .spans
            .get(index)
            .copied()
            .ok_or(ScanError::MissingUnit { index })?;
        let start = usize::try_from(span.start).map_err(|_| ScanError::LengthOverflow {
            field: "span_start",
        })?;
        let end = usize::try_from(span.end)
            .map_err(|_| ScanError::LengthOverflow { field: "span_end" })?;
        self.raw
            .get(start..end)
            .ok_or(ScanError::IncompleteCoverage {
                covered_until: span.end,
                document_len: self.raw.len() as u64,
            })
    }

    /// Validates metadata cardinality and exact contiguous byte coverage.
    ///
    /// # Errors
    ///
    /// Returns an error for discontinuous or incomplete spans.
    pub fn validate(&self) -> Result<(), ScanError> {
        if self.spans.len() != self.kinds.len() {
            return Err(ScanError::MetadataLengthMismatch {
                unit_count: self.spans.len(),
                kind_count: self.kinds.len(),
            });
        }
        let mut expected = 0_u64;
        for (index, span) in self.spans.iter().copied().enumerate() {
            if span.start != expected {
                return Err(ScanError::DiscontinuousCoverage {
                    index,
                    expected_start: expected,
                    actual_start: span.start,
                });
            }
            expected = span.end;
        }
        let document_len =
            u64::try_from(self.raw.len()).map_err(|_| ScanError::LengthOverflow {
                field: "document_len",
            })?;
        if expected != document_len {
            return Err(ScanError::IncompleteCoverage {
                covered_until: expected,
                document_len,
            });
        }
        Ok(())
    }
}

/// Scans arbitrary bytes into the compact flat-production representation.
///
/// This uses the exact same classifier, grouping rules, and code hints as
/// [`scan`] while avoiding rich per-unit allocation.
///
/// # Errors
///
/// Returns an error only when offsets cannot be represented.
pub fn scan_compact(raw: Vec<u8>) -> Result<CompactScanResult, ScanError> {
    let mut spans = Vec::new();
    let mut kinds = Vec::new();
    let mut code_tracker = CodeHintTracker::default();
    let mut index = 0_usize;

    while index < raw.len() {
        let start = index;
        let (kind, width) = classify_at(&raw, index);
        index = index.checked_add(width).ok_or(ScanError::LengthOverflow {
            field: "scan_index",
        })?;
        index = match kind {
            LexicalKind::Word => consume_word(&raw, index)?,
            LexicalKind::Number => consume_number(&raw, index)?,
            LexicalKind::Whitespace
            | LexicalKind::LineBreak
            | LexicalKind::Punctuation
            | LexicalKind::Symbol
            | LexicalKind::Control
            | LexicalKind::Opaque => consume_same_kind(&raw, index, kind)?,
        };
        code_tracker.observe(kind, &raw[start..index]);
        let start = u64::try_from(start).map_err(|_| ScanError::LengthOverflow {
            field: "span_start",
        })?;
        let end =
            u64::try_from(index).map_err(|_| ScanError::LengthOverflow { field: "span_end" })?;
        spans.push(ByteSpan { start, end });
        kinds.push(kind);
    }

    Ok(CompactScanResult {
        raw,
        spans,
        kinds,
        code_hints: code_tracker.finish(),
    })
}

/// Scans arbitrary bytes without normalization or replacement.
///
/// The result is deterministic and every byte belongs to exactly one lexical
/// unit. Invalid UTF-8 is represented by [`LexicalKind::Opaque`] spans.
///
/// # Errors
///
/// Returns an error only if offsets cannot be represented or an internal format
/// invariant is violated.
pub fn scan(raw: Vec<u8>) -> Result<ScanResult, ScanError> {
    let mut units = Vec::new();
    let mut kinds = Vec::new();
    let mut code_tracker = CodeHintTracker::default();
    let mut index = 0_usize;

    while index < raw.len() {
        let start = index;
        let (kind, width) = classify_at(&raw, index);
        index = index.checked_add(width).ok_or(ScanError::LengthOverflow {
            field: "scan_index",
        })?;

        index = match kind {
            LexicalKind::Word => consume_word(&raw, index)?,
            LexicalKind::Number => consume_number(&raw, index)?,
            LexicalKind::Whitespace
            | LexicalKind::LineBreak
            | LexicalKind::Punctuation
            | LexicalKind::Symbol
            | LexicalKind::Control
            | LexicalKind::Opaque => consume_same_kind(&raw, index, kind)?,
        };

        let start_u64 = u64::try_from(start).map_err(|_| ScanError::LengthOverflow {
            field: "span_start",
        })?;
        let end_u64 =
            u64::try_from(index).map_err(|_| ScanError::LengthOverflow { field: "span_end" })?;
        let span = ByteSpan::new(start_u64, end_u64).map_err(ScanError::Format)?;
        code_tracker.observe(kind, &raw[start..index]);
        units.push(SurfaceUnit::new(span, Vec::new()).map_err(ScanError::Format)?);
        kinds.push(kind);
    }

    let document = LosslessDocument::new(raw, units).map_err(ScanError::Format)?;
    let result = ScanResult {
        document,
        kinds,
        code_hints: code_tracker.finish(),
    };
    result.validate()?;
    Ok(result)
}

struct CodeHintTracker {
    hints: CodeScanHints,
    first_significant: bool,
    call_word: bool,
    after_member_separator: bool,
    quote_key_state: u8,
    section_state: u8,
    escaped_markup_state: u8,
    line_starts_with_hash: bool,
    previous_number: bool,
    previous_zero_number: bool,
}

impl CodeHintTracker {
    #[inline]
    fn observe(&mut self, kind: LexicalKind, bytes: &[u8]) {
        if kind == LexicalKind::LineBreak {
            self.previous_number = false;
            self.previous_zero_number = false;
            if self.section_state == 2 {
                self.hints.may_contain_unmarked_code = true;
            }
            self.reset_line();
            return;
        }
        if kind == LexicalKind::Whitespace {
            self.previous_number = false;
            self.previous_zero_number = false;
            if self.section_state == 1 {
                self.section_state = 0;
            }
            return;
        }
        if kind == LexicalKind::Punctuation && bytes.contains(&b'`') {
            self.hints.has_backtick = true;
        }
        if matches!(kind, LexicalKind::Symbol | LexicalKind::Punctuation)
            && (bytes.contains(&b'@')
                || bytes.contains(&b'=')
                || bytes.windows(3).any(|window| window == b"://"))
        {
            self.hints.may_contain_inline_artifact = true;
        }
        if kind == LexicalKind::Word
            && (matches!(bytes, b"http" | b"https" | b"www")
                || (self.previous_number && looks_like_iso_time_tail(bytes)))
        {
            self.hints.may_contain_inline_artifact = true;
        }
        if self.hints.may_contain_unmarked_code {
            self.first_significant = false;
            return;
        }

        let was_first = self.first_significant;
        let previous_number = self.previous_number;
        let previous_zero_number = self.previous_zero_number;
        self.previous_number = false;
        self.previous_zero_number = false;

        if self.quote_key_state == 2 {
            if bytes == b":" {
                self.hints.may_contain_unmarked_code = true;
                return;
            }
            self.quote_key_state = 0;
        } else if self.quote_key_state == 1
            && matches!(kind, LexicalKind::Symbol | LexicalKind::Punctuation)
        {
            if let Some(close) = bytes.iter().position(|byte| *byte == b'"') {
                let suffix = &bytes[close + 1..];
                if suffix == b":" {
                    self.hints.may_contain_unmarked_code = true;
                    return;
                }
                self.quote_key_state = u8::from(suffix.is_empty()) * 2;
            }
        }

        if self.section_state == 2 {
            self.section_state = 0;
        }

        if self.escaped_markup_state == 1 {
            if kind == LexicalKind::Word && bytes == b"lt" {
                self.escaped_markup_state = 2;
            } else {
                self.escaped_markup_state = 0;
            }
        } else if self.escaped_markup_state == 2 {
            if kind == LexicalKind::Punctuation && bytes.contains(&b';') {
                self.hints.may_contain_unmarked_code = true;
                return;
            }
            self.escaped_markup_state = 0;
        }

        match kind {
            LexicalKind::Word => {
                if previous_number && looks_like_numeric_literal_tail(bytes, previous_zero_number) {
                    self.hints.may_contain_unmarked_code = true;
                    return;
                }
                if was_first && is_code_leading_word(bytes) {
                    self.hints.may_contain_unmarked_code = true;
                    return;
                }
                self.call_word = bytes.first().is_some_and(u8::is_ascii_lowercase)
                    && (was_first || self.after_member_separator);
                self.after_member_separator = false;
            }
            LexicalKind::Symbol | LexicalKind::Punctuation => {
                if was_first
                    && (bytes.first() == Some(&b'<')
                        || matches!(bytes, b"//" | b"/*" | b"#!" | b"#["))
                {
                    self.hints.may_contain_unmarked_code = true;
                    return;
                }
                if bytes == b"=" && self.call_word {
                    self.hints.may_contain_unmarked_code = true;
                    return;
                }
                if bytes_are_strong_code_signal(bytes) {
                    self.hints.may_contain_unmarked_code = true;
                    return;
                }
                if bytes.contains(&b'(') && self.call_word {
                    self.hints.may_contain_unmarked_code = true;
                    return;
                }
                if self.line_starts_with_hash && bytes.contains(&b'!') {
                    self.hints.may_contain_unmarked_code = true;
                    return;
                }
                if self.section_state == 1 {
                    if bytes
                        .iter()
                        .all(|byte| matches!(*byte, b'.' | b':' | b'-' | b'['))
                    {
                        // Still inside a TOML-like section name.
                    } else if bytes.iter().all(|byte| *byte == b']') {
                        self.section_state = 2;
                    } else {
                        self.section_state = 0;
                    }
                }
                self.after_member_separator = matches!(bytes, b"." | b"::");
                if !matches!(bytes, b"!" | b"." | b"::") {
                    self.call_word = false;
                }
            }
            LexicalKind::Number => {
                self.previous_number = true;
                self.previous_zero_number = bytes == b"0";
                self.call_word = false;
                self.after_member_separator = false;
            }
            LexicalKind::Control | LexicalKind::Opaque => {
                self.call_word = false;
                self.after_member_separator = false;
                self.quote_key_state = 0;
                self.section_state = 0;
                self.escaped_markup_state = 0;
            }
            LexicalKind::Whitespace | LexicalKind::LineBreak => {}
        }

        if was_first {
            if bytes.first() == Some(&b'"') {
                let suffix = &bytes[1..];
                if let Some(close) = suffix.iter().position(|byte| *byte == b'"') {
                    let after = &suffix[close + 1..];
                    if after == b":" {
                        self.hints.may_contain_unmarked_code = true;
                        return;
                    }
                    self.quote_key_state = u8::from(after.is_empty()) * 2;
                } else {
                    self.quote_key_state = 1;
                }
            }
            if bytes.iter().all(|byte| *byte == b'[') {
                self.section_state = 1;
            }
            if bytes == b"&" {
                self.escaped_markup_state = 1;
            }
            self.line_starts_with_hash = bytes.first() == Some(&b'#');
        }
        self.first_significant = false;
    }

    #[inline]
    fn finish(mut self) -> CodeScanHints {
        if self.section_state == 2 {
            self.hints.may_contain_unmarked_code = true;
        }
        self.hints
    }

    #[inline]
    fn reset_line(&mut self) {
        self.first_significant = true;
        self.call_word = false;
        self.after_member_separator = false;
        self.quote_key_state = 0;
        self.section_state = 0;
        self.escaped_markup_state = 0;
        self.line_starts_with_hash = false;
        self.previous_number = false;
        self.previous_zero_number = false;
    }
}

impl Default for CodeScanHints {
    fn default() -> Self {
        Self {
            may_contain_unmarked_code: false,
            may_contain_inline_artifact: false,
            has_backtick: false,
        }
    }
}

impl Default for CodeHintTracker {
    fn default() -> Self {
        Self {
            hints: CodeScanHints::default(),
            first_significant: true,
            call_word: false,
            after_member_separator: false,
            quote_key_state: 0,
            section_state: 0,
            escaped_markup_state: 0,
            line_starts_with_hash: false,
            previous_number: false,
            previous_zero_number: false,
        }
    }
}

#[inline]
fn bytes_are_strong_code_signal(bytes: &[u8]) -> bool {
    bytes.iter().any(|byte| matches!(*byte, b'{' | b'}'))
        || matches!(
            bytes,
            b"=>"
                | b"::"
                | b"->"
                | b"+="
                | b"-="
                | b"*="
                | b"/="
                | b"%="
                | b"=="
                | b"!="
                | b"<="
                | b">="
                | b"&&"
                | b"||"
                | b"??"
                | b"?."
                | b":="
                | b"<<"
                | b">>"
        )
}

#[inline]
fn looks_like_iso_time_tail(bytes: &[u8]) -> bool {
    bytes.len() >= 3
        && matches!(bytes[0], b'T' | b't')
        && bytes[1].is_ascii_digit()
        && bytes[2].is_ascii_digit()
}

#[inline]
fn looks_like_numeric_literal_tail(bytes: &[u8], previous_zero: bool) -> bool {
    let Some((&first, rest)) = bytes.split_first() else {
        return false;
    };
    if previous_zero && matches!(first, b'x' | b'X' | b'b' | b'B' | b'o' | b'O') {
        return !rest.is_empty()
            && rest
                .iter()
                .all(|byte| byte.is_ascii_hexdigit() || *byte == b'_');
    }
    if matches!(first, b'e' | b'E') {
        let digits = rest
            .strip_prefix(b"+")
            .or_else(|| rest.strip_prefix(b"-"))
            .unwrap_or(rest);
        return !digits.is_empty()
            && digits
                .iter()
                .all(|byte| byte.is_ascii_digit() || *byte == b'_');
    }
    if matches!(first, b'f' | b'F' | b'i' | b'I' | b'u' | b'U') {
        return !rest.is_empty() && rest.iter().all(u8::is_ascii_digit);
    }
    false
}

#[inline]
fn is_code_leading_word(bytes: &[u8]) -> bool {
    matches!(
        bytes,
        b"pub"
            | b"async"
            | b"unsafe"
            | b"extern"
            | b"fn"
            | b"struct"
            | b"enum"
            | b"trait"
            | b"impl"
            | b"mod"
            | b"use"
            | b"let"
            | b"const"
            | b"static"
            | b"type"
            | b"def"
            | b"class"
            | b"import"
            | b"from"
            | b"function"
            | b"var"
            | b"package"
            | b"namespace"
            | b"interface"
            | b"export"
            | b"if"
            | b"for"
            | b"while"
            | b"match"
            | b"loop"
            | b"else"
            | b"return"
            | b"break"
            | b"continue"
            | b"pass"
            | b"raise"
            | b"try"
            | b"except"
            | b"finally"
            | b"with"
            | b"yield"
            | b"await"
            | b"SELECT"
            | b"INSERT"
            | b"UPDATE"
            | b"DELETE"
            | b"CREATE"
            | b"ALTER"
            | b"WITH"
    )
}

fn consume_same_kind(raw: &[u8], index: usize, target: LexicalKind) -> Result<usize, ScanError> {
    match target {
        LexicalKind::Whitespace => consume_whitespace_run(raw, index),
        LexicalKind::LineBreak => consume_line_break_run(raw, index),
        LexicalKind::Control => consume_control_run(raw, index),
        LexicalKind::Punctuation => consume_punctuation_run(raw, index),
        LexicalKind::Symbol => consume_symbol_run(raw, index),
        LexicalKind::Word | LexicalKind::Number | LexicalKind::Opaque => {
            consume_same_kind_generic(raw, index, target)
        }
    }
}

fn consume_same_kind_generic(
    raw: &[u8],
    mut index: usize,
    target: LexicalKind,
) -> Result<usize, ScanError> {
    while index < raw.len() {
        let (kind, width) = classify_at(raw, index);
        if kind != target {
            break;
        }
        index = index.checked_add(width).ok_or(ScanError::LengthOverflow {
            field: "scan_index",
        })?;
    }
    Ok(index)
}

fn consume_whitespace_run(raw: &[u8], mut index: usize) -> Result<usize, ScanError> {
    while index < raw.len() {
        let byte = raw[index];
        let width = if byte.is_ascii() {
            if matches!(byte, b' ' | b'\t' | 0x0b | 0x0c) {
                1
            } else {
                break;
            }
        } else {
            let (kind, width) = classify_at(raw, index);
            if kind != LexicalKind::Whitespace {
                break;
            }
            width
        };
        index = index.checked_add(width).ok_or(ScanError::LengthOverflow {
            field: "scan_index",
        })?;
    }
    Ok(index)
}

fn consume_line_break_run(raw: &[u8], mut index: usize) -> Result<usize, ScanError> {
    while index < raw.len() {
        let byte = raw[index];
        let width = if byte == b'\r' {
            1 + usize::from(raw.get(index.saturating_add(1)) == Some(&b'\n'))
        } else if byte == b'\n' {
            1
        } else if byte.is_ascii() {
            break;
        } else {
            let (kind, width) = classify_at(raw, index);
            if kind != LexicalKind::LineBreak {
                break;
            }
            width
        };
        index = index.checked_add(width).ok_or(ScanError::LengthOverflow {
            field: "scan_index",
        })?;
    }
    Ok(index)
}

fn consume_control_run(raw: &[u8], mut index: usize) -> Result<usize, ScanError> {
    while index < raw.len() {
        let byte = raw[index];
        let width = if byte.is_ascii() {
            if (byte < 0x20 && !matches!(byte, b'\r' | b'\n' | b' ' | b'\t' | 0x0b | 0x0c))
                || byte == 0x7f
            {
                1
            } else {
                break;
            }
        } else {
            let (kind, width) = classify_at(raw, index);
            if kind != LexicalKind::Control {
                break;
            }
            width
        };
        index = index.checked_add(width).ok_or(ScanError::LengthOverflow {
            field: "scan_index",
        })?;
    }
    Ok(index)
}

fn consume_punctuation_run(raw: &[u8], mut index: usize) -> Result<usize, ScanError> {
    while index < raw.len() {
        let byte = raw[index];
        let width = if byte.is_ascii() {
            if is_ascii_operator_start(byte) && ascii_operator_width(raw, index).is_some() {
                break;
            }
            if is_ascii_punctuation(byte) {
                1
            } else {
                break;
            }
        } else {
            let (kind, width) = classify_at(raw, index);
            if kind != LexicalKind::Punctuation {
                break;
            }
            width
        };
        index = index.checked_add(width).ok_or(ScanError::LengthOverflow {
            field: "scan_index",
        })?;
    }
    Ok(index)
}

fn consume_symbol_run(raw: &[u8], mut index: usize) -> Result<usize, ScanError> {
    while index < raw.len() {
        let byte = raw[index];
        let width = if byte.is_ascii() {
            if is_ascii_operator_start(byte) {
                if let Some(width) = ascii_operator_width(raw, index) {
                    width
                } else {
                    let (kind, width) = classify_ascii(raw, index, byte);
                    if kind != LexicalKind::Symbol {
                        break;
                    }
                    width
                }
            } else {
                let (kind, width) = classify_ascii(raw, index, byte);
                if kind != LexicalKind::Symbol {
                    break;
                }
                width
            }
        } else {
            let (kind, width) = classify_at(raw, index);
            if kind == LexicalKind::Symbol {
                width
            } else if scalar_at(raw, index).is_some_and(|(value, _)| value == '\u{200d}') {
                let after_joiner = index.checked_add(width).ok_or(ScanError::LengthOverflow {
                    field: "scan_index",
                })?;
                if after_joiner >= raw.len() {
                    break;
                }
                let (next_kind, next_width) = classify_at(raw, after_joiner);
                if next_kind != LexicalKind::Symbol {
                    break;
                }
                width
                    .checked_add(next_width)
                    .ok_or(ScanError::LengthOverflow {
                        field: "scan_index",
                    })?
            } else {
                break;
            }
        };
        index = index.checked_add(width).ok_or(ScanError::LengthOverflow {
            field: "scan_index",
        })?;
    }
    Ok(index)
}

fn consume_word(raw: &[u8], mut index: usize) -> Result<usize, ScanError> {
    while index < raw.len() {
        let byte = raw[index];
        if byte.is_ascii_alphanumeric() || byte == b'_' {
            index = index.checked_add(1).ok_or(ScanError::LengthOverflow {
                field: "scan_index",
            })?;
            continue;
        }
        if !byte.is_ascii() {
            let (kind, width) = classify_at(raw, index);
            if matches!(kind, LexicalKind::Word | LexicalKind::Number) {
                index = index.checked_add(width).ok_or(ScanError::LengthOverflow {
                    field: "scan_index",
                })?;
                continue;
            }
        }

        if matches!(byte, b'\'' | b'-') || !byte.is_ascii() {
            if let Some(connector_width) = word_connector_width(raw, index) {
                let after_connector =
                    index
                        .checked_add(connector_width)
                        .ok_or(ScanError::LengthOverflow {
                            field: "scan_index",
                        })?;
                if after_connector < raw.len() && is_word_or_number_at(raw, after_connector) {
                    index = after_connector;
                    continue;
                }
            }
        }
        break;
    }
    Ok(index)
}

fn consume_number(raw: &[u8], mut index: usize) -> Result<usize, ScanError> {
    while index < raw.len() {
        let byte = raw[index];
        if byte.is_ascii_digit() {
            index = index.checked_add(1).ok_or(ScanError::LengthOverflow {
                field: "scan_index",
            })?;
            continue;
        }
        if !byte.is_ascii() {
            let (kind, width) = classify_at(raw, index);
            if kind == LexicalKind::Number {
                index = index.checked_add(width).ok_or(ScanError::LengthOverflow {
                    field: "scan_index",
                })?;
                continue;
            }
        }

        if matches!(byte, b'.' | b',' | b':' | b'/' | b'-') || !byte.is_ascii() {
            if let Some(separator_width) = number_separator_width(raw, index) {
                let after_separator =
                    index
                        .checked_add(separator_width)
                        .ok_or(ScanError::LengthOverflow {
                            field: "scan_index",
                        })?;
                if after_separator < raw.len() && is_number_at(raw, after_separator) {
                    index = after_separator;
                    continue;
                }
            }
        }
        if byte == b'\'' || !byte.is_ascii() {
            if let Some(connector_width) = number_suffix_connector_width(raw, index) {
                let after_connector =
                    index
                        .checked_add(connector_width)
                        .ok_or(ScanError::LengthOverflow {
                            field: "scan_index",
                        })?;
                if after_connector < raw.len() && is_word_at(raw, after_connector) {
                    return consume_word(raw, after_connector);
                }
            }
        }
        break;
    }
    Ok(index)
}

#[inline(always)]
fn is_word_or_number_at(raw: &[u8], index: usize) -> bool {
    let byte = raw[index];
    if byte.is_ascii() {
        return byte.is_ascii_alphanumeric() || byte == b'_';
    }
    matches!(
        classify_at(raw, index).0,
        LexicalKind::Word | LexicalKind::Number
    )
}

#[inline(always)]
fn is_number_at(raw: &[u8], index: usize) -> bool {
    let byte = raw[index];
    if byte.is_ascii() {
        return byte.is_ascii_digit();
    }
    classify_at(raw, index).0 == LexicalKind::Number
}

#[inline(always)]
fn is_word_at(raw: &[u8], index: usize) -> bool {
    let byte = raw[index];
    if byte.is_ascii() {
        return byte.is_ascii_alphabetic() || byte == b'_';
    }
    classify_at(raw, index).0 == LexicalKind::Word
}

fn word_connector_width(raw: &[u8], index: usize) -> Option<usize> {
    let (value, width) = scalar_at(raw, index)?;
    matches!(value, '\'' | '\u{2019}' | '-' | '\u{2010}' | '\u{2011}').then_some(width)
}

fn number_suffix_connector_width(raw: &[u8], index: usize) -> Option<usize> {
    let (value, width) = scalar_at(raw, index)?;
    matches!(value, '\'' | '\u{2019}').then_some(width)
}

fn number_separator_width(raw: &[u8], index: usize) -> Option<usize> {
    let (value, width) = scalar_at(raw, index)?;
    matches!(value, '.' | ',' | ':' | '/' | '-' | '\u{066b}' | '\u{066c}').then_some(width)
}

fn classify_at(raw: &[u8], index: usize) -> (LexicalKind, usize) {
    let byte = raw[index];
    if byte.is_ascii() {
        if is_ascii_operator_start(byte) {
            if let Some(width) = ascii_operator_width(raw, index) {
                return (LexicalKind::Symbol, width);
            }
        }
        return classify_ascii(raw, index, byte);
    }

    let Some((value, width)) = decode_utf8_at(raw, index) else {
        return (LexicalKind::Opaque, 1);
    };

    let kind = if matches!(value, '\u{0085}' | '\u{2028}' | '\u{2029}') {
        LexicalKind::LineBreak
    } else if value.is_whitespace() {
        LexicalKind::Whitespace
    } else if value.is_control() {
        LexicalKind::Control
    } else if value.is_numeric() {
        LexicalKind::Number
    } else if value.is_alphabetic() || is_combining_mark(value) {
        LexicalKind::Word
    } else if is_unicode_punctuation(value) {
        LexicalKind::Punctuation
    } else {
        LexicalKind::Symbol
    };
    (kind, width)
}

#[inline(always)]
const fn is_ascii_operator_start(byte: u8) -> bool {
    matches!(
        byte,
        b'=' | b'!'
            | b'<'
            | b'>'
            | b'*'
            | b'/'
            | b'?'
            | b'-'
            | b':'
            | b'&'
            | b'|'
            | b'+'
            | b'%'
            | b'^'
    )
}

fn ascii_operator_width(raw: &[u8], index: usize) -> Option<usize> {
    let first = *raw.get(index)?;
    let second = *raw.get(index.checked_add(1)?)?;
    let third = raw.get(index.checked_add(2)?).copied();
    let width = match (first, second) {
        (b'=', b'=') => 2 + usize::from(third == Some(b'=')),
        (b'!', b'=') => 2 + usize::from(third == Some(b'=')),
        (b'<', b'<') | (b'>', b'>') | (b'*', b'*') | (b'/', b'/') | (b'?', b'?') => {
            2 + usize::from(third == Some(b'='))
        }
        (b'<', b'=')
        | (b'>', b'=')
        | (b'-', b'>')
        | (b'=', b'>')
        | (b':', b':')
        | (b'&', b'&')
        | (b'|', b'|')
        | (b'+', b'+')
        | (b'-', b'-')
        | (b'+', b'=')
        | (b'-', b'=')
        | (b'*', b'=')
        | (b'/', b'=')
        | (b'%', b'=')
        | (b'&', b'=')
        | (b'|', b'=')
        | (b'^', b'=')
        | (b'/', b'*')
        | (b'*', b'/')
        | (b'?', b'.') => 2,
        _ => return None,
    };
    Some(width)
}

fn classify_ascii(raw: &[u8], index: usize, byte: u8) -> (LexicalKind, usize) {
    if byte == b'\r' {
        if raw.get(index.saturating_add(1)) == Some(&b'\n') {
            return (LexicalKind::LineBreak, 2);
        }
        return (LexicalKind::LineBreak, 1);
    }
    if byte == b'\n' {
        return (LexicalKind::LineBreak, 1);
    }
    if matches!(byte, b' ' | b'\t' | 0x0b | 0x0c) {
        return (LexicalKind::Whitespace, 1);
    }
    if byte < 0x20 || byte == 0x7f {
        return (LexicalKind::Control, 1);
    }
    if byte.is_ascii_alphabetic() || byte == b'_' {
        return (LexicalKind::Word, 1);
    }
    if byte.is_ascii_digit() {
        return (LexicalKind::Number, 1);
    }
    if is_ascii_punctuation(byte) {
        return (LexicalKind::Punctuation, 1);
    }
    (LexicalKind::Symbol, 1)
}

const fn is_ascii_punctuation(byte: u8) -> bool {
    matches!(
        byte,
        b'.' | b','
            | b';'
            | b':'
            | b'!'
            | b'?'
            | b'\''
            | b'"'
            | b'('
            | b')'
            | b'['
            | b']'
            | b'{'
            | b'}'
            | b'<'
            | b'>'
            | b'-'
            | b'/'
            | b'\\'
            | b'`'
    )
}

const fn is_combining_mark(value: char) -> bool {
    matches!(
        value as u32,
        0x0300..=0x036f
            | 0x1ab0..=0x1aff
            | 0x1dc0..=0x1dff
            | 0x20d0..=0x20ff
            | 0xfe20..=0xfe2f
    )
}

const fn is_unicode_punctuation(value: char) -> bool {
    matches!(
        value as u32,
        0x2000..=0x206f
            | 0x2e00..=0x2e7f
            | 0x3000..=0x303f
            | 0xfe10..=0xfe1f
            | 0xfe30..=0xfe4f
            | 0xff01..=0xff0f
            | 0xff1a..=0xff20
            | 0xff3b..=0xff40
            | 0xff5b..=0xff65
    )
}

fn scalar_at(raw: &[u8], index: usize) -> Option<(char, usize)> {
    let byte = *raw.get(index)?;
    if byte.is_ascii() {
        return Some((char::from(byte), 1));
    }
    decode_utf8_at(raw, index)
}

fn decode_utf8_at(raw: &[u8], index: usize) -> Option<(char, usize)> {
    let first = *raw.get(index)?;
    let (width, mut codepoint) = match first {
        0xc2..=0xdf => (2, u32::from(first & 0x1f)),
        0xe0..=0xef => (3, u32::from(first & 0x0f)),
        0xf0..=0xf4 => (4, u32::from(first & 0x07)),
        _ => return None,
    };

    let second = *raw.get(index.checked_add(1)?)?;
    if !is_continuation(second) {
        return None;
    }
    if (first == 0xe0 && second < 0xa0)
        || (first == 0xed && second > 0x9f)
        || (first == 0xf0 && second < 0x90)
        || (first == 0xf4 && second > 0x8f)
    {
        return None;
    }
    codepoint = (codepoint << 6) | u32::from(second & 0x3f);

    for offset in 2..width {
        let next_index = index.checked_add(offset)?;
        let next = *raw.get(next_index)?;
        if !is_continuation(next) {
            return None;
        }
        codepoint = (codepoint << 6) | u32::from(next & 0x3f);
    }

    char::from_u32(codepoint).map(|value| (value, width))
}

const fn is_continuation(byte: u8) -> bool {
    (byte & 0xc0) == 0x80
}

/// Lexical scan failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ScanError {
    /// Underlying span-format validation failed.
    Format(FormatError),
    /// A length or offset could not be represented.
    LengthOverflow { field: &'static str },
    /// Unit and kind vectors differ in cardinality.
    MetadataLengthMismatch {
        unit_count: usize,
        kind_count: usize,
    },
    /// A lexical unit did not start where the previous one ended.
    DiscontinuousCoverage {
        index: usize,
        expected_start: u64,
        actual_start: u64,
    },
    /// Lexical units did not cover the complete document.
    IncompleteCoverage {
        covered_until: u64,
        document_len: u64,
    },
    /// A requested unit index does not exist.
    MissingUnit { index: usize },
}

impl fmt::Display for ScanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ScanError {}

#[cfg(test)]
mod tests {
    use super::{
        scan, scan_compact, scan_lexical_spans, scan_lexical_spans_with_hints, LexicalKind,
        ScanError,
    };

    #[test]
    fn preserves_code_whitespace_and_line_endings() -> Result<(), ScanError> {
        let raw = b"def f(x: int):\r\n\treturn x != 0  # yorum\n".to_vec();
        let result = scan(raw.clone())?;
        assert_eq!(result.document().decode(), raw);
        result.validate()?;

        let mut saw_crlf = false;
        let mut saw_tab = false;
        let mut saw_operator = false;
        for (index, kind) in result.kinds().iter().copied().enumerate() {
            let bytes = result.unit_bytes(index)?;
            saw_crlf |= kind == LexicalKind::LineBreak && bytes == b"\r\n";
            saw_tab |= kind == LexicalKind::Whitespace && bytes == b"\t";
            saw_operator |= kind == LexicalKind::Symbol && bytes == b"!=";
        }
        assert!(saw_crlf);
        assert!(saw_tab);
        assert!(saw_operator);
        Ok(())
    }

    #[test]
    fn keeps_turkish_apostrophe_suffix_in_one_word() -> Result<(), ScanError> {
        let result = scan("Ankara'da İstanbul’daydı".as_bytes().to_vec())?;
        assert_eq!(result.kind(0)?, LexicalKind::Word);
        assert_eq!(result.unit_bytes(0)?, b"Ankara'da");
        assert_eq!(result.kind(2)?, LexicalKind::Word);
        assert_eq!(result.unit_bytes(2)?, "İstanbul’daydı".as_bytes());
        Ok(())
    }

    #[test]
    fn groups_dates_and_decimals_as_numbers() -> Result<(), ScanError> {
        let result = scan(b"26.07.2026 3,1415".to_vec())?;
        assert_eq!(
            result.kinds(),
            [
                LexicalKind::Number,
                LexicalKind::Whitespace,
                LexicalKind::Number
            ]
        );
        assert_eq!(result.unit_bytes(0)?, b"26.07.2026");
        assert_eq!(result.unit_bytes(2)?, b"3,1415");
        Ok(())
    }

    #[test]
    fn keeps_numeric_apostrophe_suffix_in_one_unit() -> Result<(), ScanError> {
        let result = scan("2026'da 3'üncü".as_bytes().to_vec())?;
        assert_eq!(result.unit_bytes(0)?, b"2026'da");
        assert_eq!(result.kind(0)?, LexicalKind::Number);
        let ordinal = "3'üncü".as_bytes();
        assert_eq!(result.unit_bytes(2)?, ordinal);
        assert_eq!(result.kind(2)?, LexicalKind::Number);
        Ok(())
    }

    #[test]
    fn invalid_utf8_is_explicit_and_lossless() -> Result<(), ScanError> {
        let raw = vec![b'A', 0xff, 0xfe, b' ', 0xc4, 0xb0];
        let result = scan(raw.clone())?;
        assert_eq!(result.document().decode(), raw);
        assert_eq!(
            result.kinds(),
            [
                LexicalKind::Word,
                LexicalKind::Opaque,
                LexicalKind::Whitespace,
                LexicalKind::Word,
            ]
        );
        assert_eq!(result.unit_bytes(1)?, [0xff, 0xfe]);
        Ok(())
    }

    #[test]
    fn scan_is_deterministic() -> Result<(), ScanError> {
        let raw = "İyi\tgeceler! 🧪\r\nH200'lerde kod_42".as_bytes().to_vec();
        let first = scan(raw.clone())?;
        let second = scan(raw)?;
        assert_eq!(first, second);
        Ok(())
    }

    #[test]
    fn every_byte_value_is_covered_exactly() -> Result<(), ScanError> {
        let raw: Vec<u8> = (u8::MIN..=u8::MAX).collect();
        let result = scan(raw.clone())?;
        assert_eq!(result.document().decode(), raw);
        result.validate()?;
        Ok(())
    }

    #[test]
    fn recognizes_common_code_operators_as_single_symbols() -> Result<(), ScanError> {
        let result = scan(b"a != b && c::d -> e".to_vec())?;
        let expected = [
            b"!=".as_slice(),
            b"&&".as_slice(),
            b"::".as_slice(),
            b"->".as_slice(),
        ];
        let mut found = Vec::new();
        for (index, kind) in result.kinds().iter().copied().enumerate() {
            if kind == LexicalKind::Symbol {
                found.push(result.unit_bytes(index)?);
            }
        }
        assert_eq!(found, expected);
        Ok(())
    }

    #[test]
    fn code_hints_cover_late_unmarked_code_without_marking_prose() -> Result<(), ScanError> {
        let mut mixed = vec![b'a'; 800];
        mixed.extend_from_slice(b"\nfn main() {\n    return;\n}\n");
        let mixed_scan = scan(mixed)?;
        assert!(mixed_scan.code_hints().may_contain_unmarked_code());
        assert!(!mixed_scan.code_hints().has_backtick());

        let mut escaped = vec![b'a'; 800];
        escaped.extend_from_slice(b"\n&lt;SyncML version=\"1.2\"&gt;\n");
        let escaped_scan = scan(escaped)?;
        assert!(escaped_scan.code_hints().may_contain_unmarked_code());

        let prose = scan(
            "Bu yalnızca normal bir Türkçe cümledir.\nİkinci cümle de düzyazıdır."
                .as_bytes()
                .to_vec(),
        )?;
        assert!(!prose.code_hints().may_contain_unmarked_code());
        assert!(!prose.code_hints().has_backtick());
        Ok(())
    }

    #[test]
    fn deterministic_pseudorandom_bytes_remain_exact() -> Result<(), ScanError> {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for length in [0_usize, 1, 7, 31, 257, 4096] {
            let mut raw = Vec::with_capacity(length);
            for _ in 0..length {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                raw.push(state.to_le_bytes()[0]);
            }
            let result = scan(raw.clone())?;
            assert_eq!(result.document().decode(), raw);
            result.validate()?;
        }
        Ok(())
    }
    #[test]
    fn compact_scan_matches_rich_scan_exactly() -> Result<(), ScanError> {
        let mut inputs = vec![
            Vec::new(),
            b"a != b && c::d -> e".to_vec(),
            "Ankara'da İstanbul’daydı 2026'da".as_bytes().to_vec(),
            (u8::MIN..=u8::MAX).collect(),
        ];
        let mut state = 0x1234_5678_9abc_def0_u64;
        let mut random = Vec::with_capacity(8192);
        for _ in 0..8192 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            random.push(state.to_le_bytes()[0]);
        }
        inputs.push(random);
        for raw in inputs {
            let rich = scan(raw.clone())?;
            let compact = scan_compact(raw.clone())?;
            compact.validate()?;
            assert_eq!(compact.raw(), raw);
            assert_eq!(compact.kinds(), rich.kinds());
            assert_eq!(compact.code_hints(), rich.code_hints());
            assert_eq!(compact.spans().len(), rich.document().units().len());
            for index in 0..compact.spans().len() {
                assert_eq!(compact.spans()[index], rich.document().units()[index].span);
                assert_eq!(compact.unit_bytes(index)?, rich.unit_bytes(index)?);
            }
        }
        Ok(())
    }

    #[test]
    fn streamed_lexical_spans_match_compact_scan() -> Result<(), ScanError> {
        let mut inputs = vec![
            Vec::new(),
            b"a != b && c::d -> e".to_vec(),
            "Ankara'da İstanbul’daydı 2026'da".as_bytes().to_vec(),
            (u8::MIN..=u8::MAX).collect(),
        ];
        let mut state = 0x6a09_e667_f3bc_c909_u64;
        let mut random = Vec::with_capacity(16_384);
        for _ in 0..16_384 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            random.push(state.to_le_bytes()[0]);
        }
        inputs.push(random);
        for raw in inputs {
            let compact = scan_compact(raw.clone())?;
            let streamed = scan_lexical_spans(&raw).collect::<Result<Vec<_>, _>>()?;
            assert_eq!(streamed.len(), compact.spans().len());
            for (index, (span, kind)) in streamed.into_iter().enumerate() {
                assert_eq!(span, compact.spans()[index]);
                assert_eq!(kind, compact.kinds()[index]);
            }
        }
        Ok(())
    }

    #[test]
    fn hinted_stream_matches_compact_hints_and_spans() -> Result<(), ScanError> {
        let inputs = [
            b"normal Turkce metin".as_slice(),
            b"fn main() {\n    return;\n}\n".as_slice(),
            b"`inline code` ve metin".as_slice(),
            b"a != b && c::d -> e".as_slice(),
        ];
        for raw in inputs {
            let compact = scan_compact(raw.to_vec())?;
            let mut hinted = scan_lexical_spans_with_hints(raw);
            let spans = hinted.by_ref().collect::<Result<Vec<_>, _>>()?;
            assert_eq!(hinted.into_hints(), compact.code_hints());
            assert_eq!(spans.len(), compact.spans().len());
            for (index, (span, kind)) in spans.into_iter().enumerate() {
                assert_eq!(span, compact.spans()[index]);
                assert_eq!(kind, compact.kinds()[index]);
            }
        }
        Ok(())
    }
}
