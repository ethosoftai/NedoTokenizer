use std::sync::Arc;

use super::*;

const SURFACE_PROGRAM_TURKISH: u8 = 0;
const SURFACE_PROGRAM_CODE: u8 = 1;
const CODE_PROGRAM_MAX_UNITS: usize = 256;

/// Compact unit used only by the exact surface-ID production path.
///
/// Cuts live in one document-local contiguous array instead of one heap
/// allocation per unit. Rich analysis metadata remains exclusive to the
/// public rich tokenizer path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FlatSurfaceUnit {
    pub(crate) span: ByteSpan,
    pub(crate) kind: LexicalKind,
    pub(crate) mode: TokenMode,
    pub(crate) status: TokenStatus,
    cut_start: usize,
    cut_len: usize,
}

impl FlatSurfaceUnit {
    fn new(span: ByteSpan, kind: LexicalKind, mode: TokenMode, status: TokenStatus) -> Self {
        Self {
            span,
            kind,
            mode,
            status,
            cut_start: 0,
            cut_len: 0,
        }
    }

    pub(crate) fn cuts<'a>(&self, cuts: &'a [u64]) -> Result<&'a [u64], TokenizerError> {
        let end = self
            .cut_start
            .checked_add(self.cut_len)
            .ok_or(TokenizerError::LengthOverflow("flat surface cut range"))?;
        cuts.get(self.cut_start..end)
            .ok_or(TokenizerError::InvalidTrainingEncoding(
                "flat surface cut range is invalid",
            ))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SurfaceProgramUse {
    pub(crate) start_unit: usize,
    pub(crate) end_unit: usize,
    pub(crate) program: Arc<FlatSegmentProgram>,
}

pub(crate) fn split_long_surface_units(
    raw: &[u8],
    units: Vec<FlatSurfaceUnit>,
    maximum_chars: usize,
) -> Result<Vec<FlatSurfaceUnit>, TokenizerError> {
    let mut output = Vec::with_capacity(units.len());
    for unit in units {
        let should_split = matches!(unit.status, TokenStatus::Unknown | TokenStatus::Code)
            || unit.mode == TokenMode::Opaque;
        if !should_split {
            output.push(unit);
            continue;
        }
        let spans = crate::chunk_span(
            raw,
            unit.span,
            maximum_chars,
            unit.mode == TokenMode::Opaque,
        )?;
        if spans.len() == 1 {
            output.push(unit);
            continue;
        }
        for span in spans {
            output.push(FlatSurfaceUnit::new(
                span,
                unit.kind,
                unit.mode,
                unit.status,
            ));
        }
    }
    Ok(output)
}

pub(crate) fn scan_fixed_units(
    raw: &[u8],
    requested: TokenizerMode,
) -> Result<Vec<FlatSurfaceUnit>, TokenizerError> {
    let mut units = Vec::new();
    for_each_fixed_unit(raw, requested, |unit| {
        units.push(unit);
        Ok(())
    })?;
    Ok(units)
}

fn for_each_fixed_unit(
    raw: &[u8],
    requested: TokenizerMode,
    mut consume: impl FnMut(FlatSurfaceUnit) -> Result<(), TokenizerError>,
) -> Result<(), TokenizerError> {
    if requested == TokenizerMode::Auto {
        return Err(TokenizerError::InvalidConfiguration(
            "fixed lexical stream cannot run in auto mode",
        ));
    }
    if requested == TokenizerMode::Turkish {
        for lexical in nedo_core::scan_lexical_spans(raw) {
            let (span, kind) = lexical?;
            let mode = effective_mode(span, kind, &[], requested);
            for_each_turkish_piece(raw, span, kind, mode, &mut consume)?;
        }
        return Ok(());
    }
    let mut boundaries = Vec::with_capacity(8);
    for lexical in nedo_core::scan_lexical_spans(raw) {
        let (span, kind) = lexical?;
        let bytes = unit_bytes(raw, span)?;
        let mode = effective_mode(span, kind, &[], requested);
        boundaries.clear();
        boundaries.push(span.start);
        boundaries.push(span.end);
        if kind == LexicalKind::Punctuation {
            add_scalar_boundaries(bytes, span.start, &mut boundaries)?;
        } else if mode == TokenMode::Turkish
            && kind == LexicalKind::Symbol
            && bytes.iter().all(u8::is_ascii_punctuation)
        {
            add_scalar_boundaries(bytes, span.start, &mut boundaries)?;
        }
        if mode == TokenMode::Code && matches!(kind, LexicalKind::Word | LexicalKind::Number) {
            let text = std::str::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8Unit)?;
            for relative in identifier_cuts(text) {
                boundaries.push(
                    span.start
                        .checked_add(
                            u64::try_from(relative).map_err(|_| {
                                TokenizerError::LengthOverflow("identifier boundary")
                            })?,
                        )
                        .ok_or(TokenizerError::LengthOverflow("identifier boundary"))?,
                );
            }
        }
        if boundaries.len() > 2 {
            boundaries.sort_unstable();
            boundaries.dedup();
        }
        let status = initial_status(mode);
        for window in boundaries.windows(2) {
            consume(FlatSurfaceUnit::new(
                ByteSpan {
                    start: window[0],
                    end: window[1],
                },
                kind,
                mode,
                status,
            ))?;
        }
    }
    Ok(())
}

fn for_each_turkish_piece(
    raw: &[u8],
    span: ByteSpan,
    kind: LexicalKind,
    mode: TokenMode,
    consume: &mut impl FnMut(FlatSurfaceUnit) -> Result<(), TokenizerError>,
) -> Result<(), TokenizerError> {
    let bytes = unit_bytes(raw, span)?;
    let split_scalars = kind == LexicalKind::Punctuation
        || (mode == TokenMode::Turkish
            && kind == LexicalKind::Symbol
            && bytes.iter().all(u8::is_ascii_punctuation));
    let status = initial_status(mode);
    if !split_scalars {
        return consume(FlatSurfaceUnit::new(span, kind, mode, status));
    }
    let text = std::str::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8Unit)?;
    let mut starts = text.char_indices().map(|(index, _)| index).peekable();
    while let Some(relative_start) = starts.next() {
        let relative_end = starts.peek().copied().unwrap_or(bytes.len());
        let start = span
            .start
            .checked_add(
                u64::try_from(relative_start)
                    .map_err(|_| TokenizerError::LengthOverflow("structural boundary"))?,
            )
            .ok_or(TokenizerError::LengthOverflow("structural boundary"))?;
        let end = span
            .start
            .checked_add(
                u64::try_from(relative_end)
                    .map_err(|_| TokenizerError::LengthOverflow("structural boundary"))?,
            )
            .ok_or(TokenizerError::LengthOverflow("structural boundary"))?;
        consume(FlatSurfaceUnit::new(
            ByteSpan { start, end },
            kind,
            mode,
            status,
        ))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_append_cached_turkish_document(
    raw: &[u8],
    newline: bool,
    vocabulary: &SurfaceVocabulary,
    maximum_sentence_tokens: usize,
    maximum_chars: usize,
    cache: &mut FlatAnalysisCache,
    ids: &mut Vec<u16>,
    lengths: &mut Vec<u8>,
) -> Result<bool, TokenizerError> {
    if cache.surface_program_entries == 0 {
        return Ok(false);
    }
    let id_start = ids.len();
    let length_output_start = lengths.len();
    let length_start = vocabulary.begin_cached_surface_document(ids, lengths)?;
    let mut stream = CachedTurkishStream {
        raw,
        vocabulary,
        maximum_sentence_tokens,
        maximum_chars,
        cache,
        ids,
        lengths,
        segment_start: None,
        segment_end: 0,
        segment_tokens: 0,
        pending_whitespace: None,
        missed: false,
    };
    for_each_fixed_unit(raw, TokenizerMode::Turkish, |unit| stream.consume(unit))?;
    stream.finish()?;
    if stream.missed {
        stream.ids.truncate(id_start);
        stream.lengths.truncate(length_output_start);
        return Ok(false);
    }
    vocabulary.finish_cached_surface_document(
        raw.len(),
        newline,
        length_start,
        stream.ids,
        stream.lengths,
    )?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_append_cached_auto_document(
    scan: &CompactScanResult,
    newline: bool,
    vocabulary: &SurfaceVocabulary,
    maximum_sentence_tokens: usize,
    maximum_chars: usize,
    cache: &mut FlatAnalysisCache,
    ids: &mut Vec<u16>,
    lengths: &mut Vec<u8>,
) -> Result<bool, TokenizerError> {
    if cache.surface_program_entries == 0 {
        return Ok(false);
    }
    let raw = scan.raw();
    let id_start = ids.len();
    let length_output_start = lengths.len();
    let length_start = vocabulary.begin_cached_surface_document(ids, lengths)?;
    let mut stream = CachedTurkishStream {
        raw,
        vocabulary,
        maximum_sentence_tokens,
        maximum_chars,
        cache,
        ids,
        lengths,
        segment_start: None,
        segment_end: 0,
        segment_tokens: 0,
        pending_whitespace: None,
        missed: false,
    };
    for_each_compact_unit_without_code_spans(scan, TokenizerMode::Auto, |unit| {
        stream.consume(unit)
    })?;
    stream.finish()?;
    if stream.missed {
        stream.ids.truncate(id_start);
        stream.lengths.truncate(length_output_start);
        return Ok(false);
    }
    vocabulary.finish_cached_surface_document(
        raw.len(),
        newline,
        length_start,
        stream.ids,
        stream.lengths,
    )?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_append_cached_auto_raw_document(
    raw: &[u8],
    newline: bool,
    vocabulary: &SurfaceVocabulary,
    maximum_sentence_tokens: usize,
    maximum_chars: usize,
    detect_unmarked_code: bool,
    cache: &mut FlatAnalysisCache,
    ids: &mut Vec<u16>,
    lengths: &mut Vec<u8>,
) -> Result<bool, TokenizerError> {
    if cache.surface_program_entries == 0 {
        return Ok(false);
    }
    let id_start = ids.len();
    let output_length_start = lengths.len();
    let cache_hits_start = cache.hits;
    let segment_hits_start = cache.segment_program_hits;
    let length_start = vocabulary.begin_cached_surface_document(ids, lengths)?;

    let (hints, missed) = {
        let mut stream = CachedTurkishStream {
            raw,
            vocabulary,
            maximum_sentence_tokens,
            maximum_chars,
            cache,
            ids,
            lengths,
            segment_start: None,
            segment_end: 0,
            segment_tokens: 0,
            pending_whitespace: None,
            missed: false,
        };
        let mut lexical = nedo_core::scan_lexical_spans_with_hints(raw);
        for item in lexical.by_ref() {
            let (span, kind) = item?;
            let mode = effective_mode(span, kind, &[], TokenizerMode::Auto);
            for_each_turkish_piece(raw, span, kind, mode, &mut |unit| stream.consume(unit))?;
        }
        let hints = lexical.into_hints();
        stream.finish()?;
        (hints, stream.missed)
    };

    let has_code =
        hints.has_backtick() || (detect_unmarked_code && hints.may_contain_unmarked_code());
    if missed || has_code {
        ids.truncate(id_start);
        lengths.truncate(output_length_start);
        cache.hits = cache_hits_start;
        cache.segment_program_hits = segment_hits_start;
        return Ok(false);
    }
    vocabulary.finish_cached_surface_document(raw.len(), newline, length_start, ids, lengths)?;
    Ok(true)
}

struct CachedTurkishStream<'a> {
    raw: &'a [u8],
    vocabulary: &'a SurfaceVocabulary,
    maximum_sentence_tokens: usize,
    maximum_chars: usize,
    cache: &'a mut FlatAnalysisCache,
    ids: &'a mut Vec<u16>,
    lengths: &'a mut Vec<u8>,
    segment_start: Option<u64>,
    segment_end: u64,
    segment_tokens: usize,
    pending_whitespace: Option<FlatSurfaceUnit>,
    missed: bool,
}

impl CachedTurkishStream<'_> {
    fn consume(&mut self, unit: FlatSurfaceUnit) -> Result<(), TokenizerError> {
        if self.missed {
            return Ok(());
        }
        if unit.mode != TokenMode::Turkish
            || matches!(
                unit.kind,
                LexicalKind::LineBreak | LexicalKind::Control | LexicalKind::Opaque
            )
        {
            self.flush_segment()?;
            if !self.missed {
                self.encode_direct(unit)?;
            }
            return Ok(());
        }
        if unit.kind == LexicalKind::Whitespace {
            if self.segment_start.is_some() {
                if let Some(pending) = self.pending_whitespace.as_mut() {
                    if pending.span.end == unit.span.start
                        && pending.kind == unit.kind
                        && pending.mode == unit.mode
                        && pending.status == unit.status
                    {
                        pending.span.end = unit.span.end;
                    } else {
                        return Err(TokenizerError::InvalidTrainingEncoding(
                            "non-contiguous cached trailing whitespace",
                        ));
                    }
                } else {
                    self.pending_whitespace = Some(unit);
                }
            } else {
                self.encode_direct(unit)?;
            }
            return Ok(());
        }
        if self.segment_start.is_none() {
            self.segment_start = Some(unit.span.start);
        } else {
            self.pending_whitespace = None;
        }
        self.segment_end = unit.span.end;
        self.segment_tokens = self.segment_tokens.saturating_add(1);
        if is_sentence_boundary(unit.kind, self.raw, unit.span)?
            || self.segment_tokens >= self.maximum_sentence_tokens
        {
            self.flush_segment()?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), TokenizerError> {
        if !self.missed {
            self.flush_segment()?;
        }
        Ok(())
    }

    fn flush_segment(&mut self) -> Result<(), TokenizerError> {
        let Some(start) = self.segment_start.take() else {
            return Ok(());
        };
        let exact = unit_bytes(
            self.raw,
            ByteSpan {
                start,
                end: self.segment_end,
            },
        )?;
        if !append_borrowed_surface_program(
            self.cache,
            exact,
            SURFACE_PROGRAM_TURKISH,
            self.ids,
            self.lengths,
        )? {
            self.missed = true;
            return Ok(());
        }
        self.cache.segment_program_hits = self.cache.segment_program_hits.saturating_add(1);
        self.cache.hits = self
            .cache
            .hits
            .saturating_add(u64::try_from(self.segment_tokens).unwrap_or(u64::MAX));
        self.segment_end = 0;
        self.segment_tokens = 0;
        if let Some(trailing) = self.pending_whitespace.take() {
            self.encode_direct(trailing)?;
        }
        Ok(())
    }

    fn encode_direct(&mut self, unit: FlatSurfaceUnit) -> Result<(), TokenizerError> {
        self.vocabulary.encode_flat_unit_direct(
            self.raw,
            &unit,
            self.maximum_chars,
            self.ids,
            self.lengths,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn append_cached_code_document(
    raw: &[u8],
    newline: bool,
    vocabulary: &SurfaceVocabulary,
    maximum_chars: usize,
    cache: &mut FlatAnalysisCache,
    ids: &mut Vec<u16>,
    lengths: &mut Vec<u8>,
) -> Result<(), TokenizerError> {
    if cache.surface_program_entries != 0 {
        let id_start = ids.len();
        let output_length_start = lengths.len();
        let length_start = vocabulary.begin_cached_surface_document(ids, lengths)?;
        if try_append_cached_code_lines(raw, cache, ids, lengths)? {
            vocabulary.finish_cached_surface_document(
                raw.len(),
                newline,
                length_start,
                ids,
                lengths,
            )?;
            return Ok(());
        }
        ids.truncate(id_start);
        lengths.truncate(output_length_start);
    }

    let length_start = vocabulary.begin_cached_surface_document(ids, lengths)?;
    let mut stream = CachedCodeStream {
        raw,
        vocabulary,
        maximum_chars,
        cache,
        ids,
        lengths,
        units: Vec::with_capacity(64),
        chunk_start: 0,
        chunk_end: 0,
    };
    for_each_fixed_unit(raw, TokenizerMode::Code, |unit| stream.consume(unit))?;
    stream.flush()?;
    vocabulary.finish_cached_surface_document(
        raw.len(),
        newline,
        length_start,
        stream.ids,
        stream.lengths,
    )
}

fn try_append_cached_code_lines(
    raw: &[u8],
    cache: &mut FlatAnalysisCache,
    ids: &mut Vec<u16>,
    lengths: &mut Vec<u8>,
) -> Result<bool, TokenizerError> {
    let mut start = 0_usize;
    let mut hit_lines = 0_u64;
    while start < raw.len() {
        let end = next_code_line_end(raw, start);
        let exact = &raw[start..end];
        if !append_borrowed_surface_program(cache, exact, SURFACE_PROGRAM_CODE, ids, lengths)? {
            return Ok(false);
        }
        hit_lines = hit_lines.saturating_add(1);
        start = end;
    }
    cache.segment_program_hits = cache.segment_program_hits.saturating_add(hit_lines);
    cache.hits = cache.hits.saturating_add(hit_lines);
    Ok(true)
}

#[inline]
fn next_code_line_end(raw: &[u8], start: usize) -> usize {
    let mut index = start;
    while index < raw.len() {
        match raw[index] {
            b'\n' => return index + 1,
            b'\r' => {
                return index + 1 + usize::from(raw.get(index + 1) == Some(&b'\n'));
            }
            0xc2 if raw.get(index + 1) == Some(&0x85) => return index + 2,
            0xe2 if raw.get(index + 1) == Some(&0x80)
                && matches!(raw.get(index + 2), Some(0xa8 | 0xa9)) =>
            {
                return index + 3;
            }
            _ => index += 1,
        }
    }
    raw.len()
}

struct CachedCodeStream<'a> {
    raw: &'a [u8],
    vocabulary: &'a SurfaceVocabulary,
    maximum_chars: usize,
    cache: &'a mut FlatAnalysisCache,
    ids: &'a mut Vec<u16>,
    lengths: &'a mut Vec<u8>,
    units: Vec<FlatSurfaceUnit>,
    chunk_start: u64,
    chunk_end: u64,
}

impl CachedCodeStream<'_> {
    fn consume(&mut self, unit: FlatSurfaceUnit) -> Result<(), TokenizerError> {
        if self.units.is_empty() {
            self.chunk_start = unit.span.start;
        }
        self.chunk_end = unit.span.end;
        let boundary = unit.kind == LexicalKind::LineBreak
            || unit.mode == TokenMode::Opaque
            || self.units.len().saturating_add(1) >= CODE_PROGRAM_MAX_UNITS;
        self.units.push(unit);
        if boundary {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), TokenizerError> {
        if self.units.is_empty() {
            return Ok(());
        }
        let exact = unit_bytes(
            self.raw,
            ByteSpan {
                start: self.chunk_start,
                end: self.chunk_end,
            },
        )?;
        if let Some(program) = cached_surface_program(self.cache, exact, SURFACE_PROGRAM_CODE) {
            if program.surface_ids.len() != program.surface_lengths.len() {
                return Err(TokenizerError::InvalidTrainingEncoding(
                    "cached code surface program cardinalities differ",
                ));
            }
            self.cache.segment_program_hits = self.cache.segment_program_hits.saturating_add(1);
            self.cache.hits = self
                .cache
                .hits
                .saturating_add(u64::try_from(self.units.len()).unwrap_or(u64::MAX));
            self.ids.extend_from_slice(&program.surface_ids);
            self.lengths.extend_from_slice(&program.surface_lengths);
        } else {
            let mut program_ids = Vec::new();
            let mut program_lengths = Vec::new();
            self.vocabulary.encode_flat_range_into(
                self.raw,
                &self.units,
                &[],
                0..self.units.len(),
                self.maximum_chars,
                false,
                &mut program_ids,
                &mut program_lengths,
            )?;
            self.ids.extend_from_slice(&program_ids);
            self.lengths.extend_from_slice(&program_lengths);
            let fingerprint = surface_program_fingerprint(exact, SURFACE_PROGRAM_CODE);
            let program = FlatSegmentProgram {
                tokens: Box::new([]),
                relative_cuts: Box::new([]),
                surface_ids: program_ids.into_boxed_slice(),
                surface_lengths: program_lengths.into_boxed_slice(),
            };
            let _ = insert_surface_program(
                self.cache,
                SURFACE_PROGRAM_CODE,
                fingerprint,
                exact,
                program,
            );
        }
        self.units.clear();
        self.chunk_start = self.chunk_end;
        Ok(())
    }
}

pub(crate) fn split_units(
    lexical_scan: &CompactScanResult,
    code_spans: &[ByteSpan],
    requested: TokenizerMode,
) -> Result<Vec<FlatSurfaceUnit>, TokenizerError> {
    let spans = lexical_scan.spans();
    let kinds = lexical_scan.kinds();
    if spans.len() != kinds.len() {
        return Err(TokenizerError::Scan(ScanError::MetadataLengthMismatch {
            unit_count: spans.len(),
            kind_count: kinds.len(),
        }));
    }
    if code_spans.is_empty() {
        return split_units_without_code_spans(lexical_scan, requested);
    }

    let mut boundaries = Vec::new();
    for (index, (span, kind)) in spans.iter().copied().zip(kinds.iter().copied()).enumerate() {
        boundaries.push(span.start);
        boundaries.push(span.end);
        let bytes = lexical_scan.unit_bytes(index)?;
        if kind == LexicalKind::Punctuation {
            add_scalar_boundaries(bytes, span.start, &mut boundaries)?;
        }
        let mode = effective_mode(span, kind, code_spans, requested);
        if mode == TokenMode::Turkish
            && kind == LexicalKind::Symbol
            && bytes.iter().all(u8::is_ascii_punctuation)
        {
            add_scalar_boundaries(bytes, span.start, &mut boundaries)?;
        }
        if mode == TokenMode::Code && matches!(kind, LexicalKind::Word | LexicalKind::Number) {
            let text = std::str::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8Unit)?;
            for relative in identifier_cuts(text) {
                boundaries.push(
                    span.start
                        .checked_add(
                            u64::try_from(relative).map_err(|_| {
                                TokenizerError::LengthOverflow("identifier boundary")
                            })?,
                        )
                        .ok_or(TokenizerError::LengthOverflow("identifier boundary"))?,
                );
            }
        }
    }
    for span in code_spans {
        boundaries.push(span.start);
        boundaries.push(span.end);
    }
    boundaries.sort_unstable();
    boundaries.dedup();

    let mut units = Vec::with_capacity(boundaries.len().saturating_sub(1));
    let mut source_index = 0_usize;
    for window in boundaries.windows(2) {
        let span = ByteSpan {
            start: window[0],
            end: window[1],
        };
        while source_index + 1 < spans.len() && spans[source_index].end <= span.start {
            source_index += 1;
        }
        let source =
            spans
                .get(source_index)
                .copied()
                .ok_or(TokenizerError::InvalidTrainingEncoding(
                    "flat surface source unit is absent",
                ))?;
        if span.start < source.start || span.end > source.end {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "flat surface split crosses a lexical unit",
            ));
        }
        let kind = kinds[source_index];
        let mode = effective_mode(span, kind, code_spans, requested);
        units.push(FlatSurfaceUnit::new(span, kind, mode, initial_status(mode)));
    }
    Ok(units)
}

fn split_units_without_code_spans(
    lexical_scan: &CompactScanResult,
    requested: TokenizerMode,
) -> Result<Vec<FlatSurfaceUnit>, TokenizerError> {
    let mut units = Vec::with_capacity(lexical_scan.spans().len());
    for_each_compact_unit_without_code_spans(lexical_scan, requested, |unit| {
        units.push(unit);
        Ok(())
    })?;
    Ok(units)
}

fn for_each_compact_unit_without_code_spans(
    lexical_scan: &CompactScanResult,
    requested: TokenizerMode,
    mut consume: impl FnMut(FlatSurfaceUnit) -> Result<(), TokenizerError>,
) -> Result<(), TokenizerError> {
    let spans = lexical_scan.spans();
    let kinds = lexical_scan.kinds();
    let mut boundaries = Vec::with_capacity(8);
    for (index, (span, kind)) in spans.iter().copied().zip(kinds.iter().copied()).enumerate() {
        let mode = effective_mode(span, kind, &[], requested);
        boundaries.clear();
        boundaries.push(span.start);
        boundaries.push(span.end);
        let bytes = lexical_scan.unit_bytes(index)?;
        if kind == LexicalKind::Punctuation {
            add_scalar_boundaries(bytes, span.start, &mut boundaries)?;
        } else if mode == TokenMode::Turkish
            && kind == LexicalKind::Symbol
            && bytes.iter().all(u8::is_ascii_punctuation)
        {
            add_scalar_boundaries(bytes, span.start, &mut boundaries)?;
        }
        if mode == TokenMode::Code && matches!(kind, LexicalKind::Word | LexicalKind::Number) {
            let text = std::str::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8Unit)?;
            for relative in identifier_cuts(text) {
                boundaries.push(
                    span.start
                        .checked_add(
                            u64::try_from(relative).map_err(|_| {
                                TokenizerError::LengthOverflow("identifier boundary")
                            })?,
                        )
                        .ok_or(TokenizerError::LengthOverflow("identifier boundary"))?,
                );
            }
        }
        if boundaries.len() > 2 {
            boundaries.sort_unstable();
            boundaries.dedup();
        }
        let status = initial_status(mode);
        for window in boundaries.windows(2) {
            consume(FlatSurfaceUnit::new(
                ByteSpan {
                    start: window[0],
                    end: window[1],
                },
                kind,
                mode,
                status,
            ))?;
        }
    }
    Ok(())
}

const fn initial_status(mode: TokenMode) -> TokenStatus {
    match mode {
        TokenMode::Code => TokenStatus::Code,
        TokenMode::Opaque => TokenStatus::Opaque,
        TokenMode::Turkish => TokenStatus::Structural,
    }
}

pub(crate) fn apply_contextual_analysis(
    tokenizer: &Tokenizer<'_>,
    raw: &[u8],
    units: &mut [FlatSurfaceUnit],
    cuts: &mut Vec<u64>,
    vocabulary: &SurfaceVocabulary,
    use_morphology: bool,
    maximum_chars: usize,
    cache: &mut FlatAnalysisCache,
) -> Result<Vec<SurfaceProgramUse>, TokenizerError> {
    let mut segment = Vec::new();
    let mut programs = Vec::new();
    for index in 0..units.len() {
        let unit = units[index];
        if unit.mode != TokenMode::Turkish
            || matches!(
                unit.kind,
                LexicalKind::LineBreak | LexicalKind::Control | LexicalKind::Opaque
            )
        {
            flush_segment(
                tokenizer,
                raw,
                units,
                cuts,
                &mut segment,
                &mut programs,
                vocabulary,
                use_morphology,
                maximum_chars,
                cache,
            )?;
            continue;
        }
        if unit.kind == LexicalKind::Whitespace {
            continue;
        }
        segment.push(index);
        let boundary = is_sentence_boundary(unit.kind, raw, unit.span)?;
        if boundary || segment.len() >= tokenizer.config.max_sentence_tokens {
            flush_segment(
                tokenizer,
                raw,
                units,
                cuts,
                &mut segment,
                &mut programs,
                vocabulary,
                use_morphology,
                maximum_chars,
                cache,
            )?;
        }
    }
    flush_segment(
        tokenizer,
        raw,
        units,
        cuts,
        &mut segment,
        &mut programs,
        vocabulary,
        use_morphology,
        maximum_chars,
        cache,
    )?;
    Ok(programs)
}

#[allow(clippy::too_many_arguments)]
fn flush_segment(
    tokenizer: &Tokenizer<'_>,
    raw: &[u8],
    units: &mut [FlatSurfaceUnit],
    cuts: &mut Vec<u64>,
    indices: &mut Vec<usize>,
    programs: &mut Vec<SurfaceProgramUse>,
    vocabulary: &SurfaceVocabulary,
    use_morphology: bool,
    maximum_chars: usize,
    cache: &mut FlatAnalysisCache,
) -> Result<(), TokenizerError> {
    if indices.is_empty() {
        return Ok(());
    }
    let exact = segment_exact_bytes(raw, units, indices)?;
    let fingerprint = surface_program_fingerprint(exact, SURFACE_PROGRAM_TURKISH);
    if let Some(program) = cached_surface_program(cache, exact, SURFACE_PROGRAM_TURKISH) {
        cache.segment_program_hits = cache.segment_program_hits.saturating_add(1);
        cache.hits = cache
            .hits
            .saturating_add(u64::try_from(indices.len()).unwrap_or(u64::MAX));
        programs.push(program_use(indices, program)?);
        indices.clear();
        return Ok(());
    }
    cache.segment_program_misses = cache.segment_program_misses.saturating_add(1);

    #[cfg(feature = "compiled-surface-table")]
    if let Some(table) = tokenizer.compiled_surface_analysis_table.as_ref() {
        let mut sources = Vec::with_capacity(indices.len());
        for index in indices.iter().copied() {
            let token = unit_str(raw, units[index].span)?;
            if let Some(set) = table.get(token) {
                sources.push(FlatAnalysisSource::Compiled(set));
            } else {
                sources.push(FlatAnalysisSource::Live(
                    cache.analyze(&tokenizer.morphology, token)?,
                ));
            }
        }
        let sets = sources
            .iter()
            .map(FlatAnalysisSource::set)
            .collect::<Vec<_>>();
        apply_sets(tokenizer, raw, units, cuts, indices, &sets, cache)?;
    } else {
        let mut owned_sets = Vec::with_capacity(indices.len());
        for index in indices.iter().copied() {
            let token = unit_str(raw, units[index].span)?;
            owned_sets.push(cache.analyze(&tokenizer.morphology, token)?);
        }
        let sets = owned_sets.iter().map(AsRef::as_ref).collect::<Vec<_>>();
        apply_sets(tokenizer, raw, units, cuts, indices, &sets, cache)?;
    }
    #[cfg(not(feature = "compiled-surface-table"))]
    {
        let mut owned_sets = Vec::with_capacity(indices.len());
        for index in indices.iter().copied() {
            let token = unit_str(raw, units[index].span)?;
            owned_sets.push(cache.analyze(&tokenizer.morphology, token)?);
        }
        let sets = owned_sets.iter().map(AsRef::as_ref).collect::<Vec<_>>();
        apply_sets(tokenizer, raw, units, cuts, indices, &sets, cache)?;
    }
    // Surface tokenization is left-context sensitive: one ASCII space immediately
    // before the first word may merge into that word's root. The cached sentence
    // program starts at the first non-whitespace unit, so caching final IDs here
    // would silently drop that bridge. Cache morphology analysis only; final IDs
    // are always produced from the real document unit stream below.
    let _ = (programs, vocabulary, use_morphology, maximum_chars, fingerprint, exact);
    indices.clear();
    Ok(())
}

fn program_use(
    indices: &[usize],
    program: Arc<FlatSegmentProgram>,
) -> Result<SurfaceProgramUse, TokenizerError> {
    let start_unit = indices
        .first()
        .copied()
        .ok_or(TokenizerError::InvalidTrainingEncoding(
            "surface segment has no first unit",
        ))?;
    let end_unit = indices
        .last()
        .copied()
        .and_then(|index| index.checked_add(1))
        .ok_or(TokenizerError::LengthOverflow("surface segment unit end"))?;
    Ok(SurfaceProgramUse {
        start_unit,
        end_unit,
        program,
    })
}

fn insert_surface_program(
    cache: &mut FlatAnalysisCache,
    kind: u8,
    fingerprint: u64,
    exact: &[u8],
    program: FlatSegmentProgram,
) -> Arc<FlatSegmentProgram> {
    let program = Arc::new(program);
    if cache.surface_program_entries >= SEGMENT_PROGRAM_CACHE_ENTRIES {
        return program;
    }
    let bytes = exact
        .len()
        .saturating_add(std::mem::size_of::<FlatSurfaceProgramEntry>())
        .saturating_add(
            program
                .surface_ids
                .len()
                .saturating_mul(std::mem::size_of::<u16>()),
        )
        .saturating_add(program.surface_lengths.len());
    let Some(slot) = vacant_surface_program_slot(cache, fingerprint) else {
        return program;
    };
    cache.surface_program_bytes = cache
        .surface_program_bytes
        .saturating_add(u64::try_from(bytes).unwrap_or(u64::MAX));
    cache.surface_programs.slots[slot] = Some(FlatSurfaceProgramEntry {
        fingerprint,
        kind,
        exact: exact.to_vec().into_boxed_slice(),
        program: Arc::clone(&program),
    });
    cache.surface_program_entries = cache.surface_program_entries.saturating_add(1);
    program
}

#[inline]
fn vacant_surface_program_slot(cache: &FlatAnalysisCache, fingerprint: u64) -> Option<usize> {
    let mask = cache.surface_programs.slots.len().checked_sub(1)?;
    debug_assert!(cache.surface_programs.slots.len().is_power_of_two());
    let mut slot = (fingerprint as usize) & mask;
    for _ in 0..cache.surface_programs.slots.len() {
        if cache.surface_programs.slots[slot].is_none() {
            return Some(slot);
        }
        slot = (slot + 1) & mask;
    }
    None
}

fn segment_exact_bytes<'a>(
    raw: &'a [u8],
    units: &[FlatSurfaceUnit],
    indices: &[usize],
) -> Result<&'a [u8], TokenizerError> {
    let first = indices.first().and_then(|index| units.get(*index)).ok_or(
        TokenizerError::InvalidTrainingEncoding("flat surface segment has no first unit"),
    )?;
    let last = indices.last().and_then(|index| units.get(*index)).ok_or(
        TokenizerError::InvalidTrainingEncoding("flat surface segment has no last unit"),
    )?;
    unit_bytes(
        raw,
        ByteSpan {
            start: first.span.start,
            end: last.span.end,
        },
    )
}

fn append_borrowed_surface_program(
    cache: &FlatAnalysisCache,
    exact: &[u8],
    kind: u8,
    ids: &mut Vec<u16>,
    lengths: &mut Vec<u8>,
) -> Result<bool, TokenizerError> {
    let fingerprint = surface_program_fingerprint(exact, kind);
    let Some(entry) = find_surface_program(cache, fingerprint, exact, kind) else {
        return Ok(false);
    };
    if entry.program.surface_ids.len() != entry.program.surface_lengths.len() {
        return Err(TokenizerError::InvalidTrainingEncoding(
            "cached surface program cardinalities differ",
        ));
    }
    ids.extend_from_slice(&entry.program.surface_ids);
    lengths.extend_from_slice(&entry.program.surface_lengths);
    Ok(true)
}

fn cached_surface_program(
    cache: &FlatAnalysisCache,
    exact: &[u8],
    kind: u8,
) -> Option<Arc<FlatSegmentProgram>> {
    let fingerprint = surface_program_fingerprint(exact, kind);
    find_surface_program(cache, fingerprint, exact, kind).map(|entry| Arc::clone(&entry.program))
}

#[inline]
fn find_surface_program<'a>(
    cache: &'a FlatAnalysisCache,
    fingerprint: u64,
    exact: &[u8],
    kind: u8,
) -> Option<&'a FlatSurfaceProgramEntry> {
    let mask = cache.surface_programs.slots.len().checked_sub(1)?;
    debug_assert!(cache.surface_programs.slots.len().is_power_of_two());
    let mut slot = (fingerprint as usize) & mask;
    for _ in 0..cache.surface_programs.slots.len() {
        let entry = cache.surface_programs.slots[slot].as_ref()?;
        if entry.fingerprint == fingerprint && entry.kind == kind && entry.exact.as_ref() == exact {
            return Some(entry);
        }
        slot = (slot + 1) & mask;
    }
    None
}

#[inline(always)]
fn surface_program_fingerprint(bytes: &[u8], kind: u8) -> u64 {
    const MIX1: u64 = 0x9e37_79b1_85eb_ca87;
    const MIX2: u64 = 0xc2b2_ae3d_27d4_eb4f;
    let mut hash = (bytes.len() as u64).wrapping_mul(MIX1)
        ^ MIX2
        ^ u64::from(kind).wrapping_mul(0xd6e8_feb8_6659_fd93);
    if bytes.len() <= 32 {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            hash = mix_surface_fingerprint(hash, load_surface_word(chunk));
        }
        let remainder = chunks.remainder();
        if !remainder.is_empty() {
            hash = mix_surface_fingerprint(hash, load_surface_tail(remainder));
        }
    } else {
        let last = bytes.len() - 8;
        let middle = bytes.len() / 2 - 4;
        let quarter = bytes.len() / 4;
        let three_quarters = bytes.len() * 3 / 4 - 4;
        for offset in [0, quarter, middle, three_quarters, last] {
            hash = mix_surface_fingerprint(hash, load_surface_word(&bytes[offset..offset + 8]));
        }
    }
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^= hash >> 33;
    hash
}
#[inline(always)]
fn mix_surface_fingerprint(hash: u64, word: u64) -> u64 {
    const MIX1: u64 = 0x9e37_79b1_85eb_ca87;
    const MIX2: u64 = 0xc2b2_ae3d_27d4_eb4f;
    (hash ^ word.wrapping_mul(MIX1))
        .rotate_left(27)
        .wrapping_mul(MIX2)
        .wrapping_add(MIX1)
}
#[inline(always)]
fn load_surface_word(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(
        bytes
            .try_into()
            .expect("eight-byte surface fingerprint word"),
    )
}
#[inline(always)]
fn load_surface_tail(bytes: &[u8]) -> u64 {
    let mut word = 0_u64;
    for (shift, byte) in bytes.iter().copied().enumerate() {
        word |= u64::from(byte) << (shift * 8);
    }
    word
}

pub(crate) fn try_append_cached_document_program(
    raw: &[u8],
    newline: bool,
    cache: &mut FlatAnalysisCache,
    ids: &mut Vec<u16>,
    lengths: &mut Vec<u8>,
) -> Result<bool, TokenizerError> {
    if cache.document_program_capacity == 0 || cache.document_program_entries == 0 {
        if cache.document_program_capacity > 0 {
            cache.document_program_misses = cache.document_program_misses.saturating_add(1);
        }
        return Ok(false);
    }
    let fingerprint = document_program_fingerprint(raw, newline);
    let Some(entry) = find_document_program(cache, fingerprint, raw, newline) else {
        cache.document_program_misses = cache.document_program_misses.saturating_add(1);
        return Ok(false);
    };
    if entry.ids.len() != entry.lengths.len() {
        return Err(TokenizerError::InvalidTrainingEncoding(
            "cached document program cardinalities differ",
        ));
    }
    ids.extend_from_slice(&entry.ids);
    lengths.extend_from_slice(&entry.lengths);
    cache.document_program_hits = cache.document_program_hits.saturating_add(1);
    Ok(true)
}

pub(crate) fn insert_document_program(
    cache: &mut FlatAnalysisCache,
    exact: Box<[u8]>,
    newline: bool,
    ids: &[u16],
    lengths: &[u8],
) -> Result<(), TokenizerError> {
    if ids.len() != lengths.len() {
        return Err(TokenizerError::InvalidTrainingEncoding(
            "document program cardinalities differ",
        ));
    }
    if cache.document_program_capacity == 0
        || cache.document_program_entries >= cache.document_program_capacity
    {
        return Ok(());
    }
    let fingerprint = document_program_fingerprint(&exact, newline);
    if find_document_program(cache, fingerprint, &exact, newline).is_some() {
        return Ok(());
    }
    let Some(slot) = vacant_document_program_slot(cache, fingerprint) else {
        return Ok(());
    };
    let bytes = exact
        .len()
        .saturating_add(ids.len().saturating_mul(std::mem::size_of::<u16>()))
        .saturating_add(lengths.len())
        .saturating_add(std::mem::size_of::<FlatDocumentProgramEntry>());
    cache.document_programs.slots[slot] = Some(FlatDocumentProgramEntry {
        fingerprint,
        newline,
        exact,
        ids: ids.to_vec().into_boxed_slice(),
        lengths: lengths.to_vec().into_boxed_slice(),
    });
    cache.document_program_entries = cache.document_program_entries.saturating_add(1);
    cache.document_program_bytes = cache
        .document_program_bytes
        .saturating_add(u64::try_from(bytes).unwrap_or(u64::MAX));
    Ok(())
}

#[inline]
fn find_document_program<'a>(
    cache: &'a FlatAnalysisCache,
    fingerprint: u64,
    exact: &[u8],
    newline: bool,
) -> Option<&'a FlatDocumentProgramEntry> {
    let mask = cache.document_programs.slots.len().checked_sub(1)?;
    debug_assert!(cache.document_programs.slots.len().is_power_of_two());
    let mut slot = (fingerprint as usize) & mask;
    for _ in 0..cache.document_programs.slots.len() {
        let entry = cache.document_programs.slots[slot].as_ref()?;
        if entry.fingerprint == fingerprint
            && entry.newline == newline
            && entry.exact.as_ref() == exact
        {
            return Some(entry);
        }
        slot = (slot + 1) & mask;
    }
    None
}

#[inline]
fn vacant_document_program_slot(cache: &FlatAnalysisCache, fingerprint: u64) -> Option<usize> {
    let mask = cache.document_programs.slots.len().checked_sub(1)?;
    debug_assert!(cache.document_programs.slots.len().is_power_of_two());
    let mut slot = (fingerprint as usize) & mask;
    for _ in 0..cache.document_programs.slots.len() {
        if cache.document_programs.slots[slot].is_none() {
            return Some(slot);
        }
        slot = (slot + 1) & mask;
    }
    None
}

#[inline(always)]
fn document_program_fingerprint(raw: &[u8], newline: bool) -> u64 {
    surface_program_fingerprint(raw, 0xd0 | u8::from(newline))
}

fn apply_sets(
    tokenizer: &Tokenizer<'_>,
    raw: &[u8],
    units: &mut [FlatSurfaceUnit],
    cuts: &mut Vec<u64>,
    indices: &[usize],
    sets: &[&FlatAnalysisSet],
    cache: &mut FlatAnalysisCache,
) -> Result<(), TokenizerError> {
    let output_invariant = sets.iter().all(|set| set.output_invariant);

    if tokenizer.config.contextual_disambiguation && !output_invariant {
        let ambiguity = sets
            .iter()
            .map(|set| set.ambiguity.as_ref())
            .collect::<Vec<_>>();
        let scoring_codes = sets
            .iter()
            .map(|set| set.scoring_codes.as_ref())
            .collect::<Vec<_>>();
        let selected = tokenizer.disambiguator.disambiguate_indices_scored_causal(
            &ambiguity,
            &scoring_codes,
            &mut cache.disambiguation_scores,
        )?;
        if selected.len() != indices.len() {
            return Err(TokenizerError::ContextLengthMismatch);
        }
        for ((&index, set), candidate_index) in indices.iter().zip(sets).zip(selected) {
            apply_candidate(raw, &mut units[index], cuts, set, candidate_index)?;
        }
    } else {
        for (&index, set) in indices.iter().zip(sets) {
            apply_candidate(raw, &mut units[index], cuts, set, 0)?;
        }
    }
    Ok(())
}

fn apply_candidate(
    raw: &[u8],
    unit: &mut FlatSurfaceUnit,
    cuts: &mut Vec<u64>,
    set: &FlatAnalysisSet,
    candidate_index: usize,
) -> Result<(), TokenizerError> {
    let relative =
        set.relative_cuts
            .get(candidate_index)
            .ok_or(TokenizerError::InvalidTrainingEncoding(
                "flat surface disambiguator selected an out-of-range candidate",
            ))?;
    let unknown =
        *set.unknown
            .get(candidate_index)
            .ok_or(TokenizerError::InvalidTrainingEncoding(
                "flat surface candidate status is out of range",
            ))?;
    apply_output(raw, unit, cuts, relative, unknown)
}

fn apply_output(
    raw: &[u8],
    unit: &mut FlatSurfaceUnit,
    cuts: &mut Vec<u64>,
    relative: &[u32],
    unknown: bool,
) -> Result<(), TokenizerError> {
    if matches!(unit.kind, LexicalKind::Punctuation | LexicalKind::Symbol) {
        unit.status = TokenStatus::Structural;
        unit.cut_start = cuts.len();
        unit.cut_len = 0;
        return Ok(());
    }
    unit.status = if unknown {
        TokenStatus::Unknown
    } else {
        TokenStatus::Morphological
    };
    let start = cuts.len();
    for cut in relative {
        cuts.push(
            unit.span
                .start
                .checked_add(u64::from(*cut))
                .ok_or(TokenizerError::LengthOverflow("flat surface absolute cut"))?,
        );
    }
    if unit.kind == LexicalKind::Number {
        cuts.extend(numeric_micro_cuts(raw, unit.span)?);
        cuts[start..].sort_unstable();
        let mut write = start;
        for read in start..cuts.len() {
            if write == start || cuts[read] != cuts[write - 1] {
                cuts[write] = cuts[read];
                write += 1;
            }
        }
        cuts.truncate(write);
    }
    unit.cut_start = start;
    unit.cut_len = cuts.len() - start;
    Ok(())
}

fn capture_segment_program(
    raw: &[u8],
    units: &[FlatSurfaceUnit],
    cuts: &[u64],
    indices: &[usize],
    vocabulary: &SurfaceVocabulary,
    use_morphology: bool,
    maximum_chars: usize,
) -> Result<FlatSegmentProgram, TokenizerError> {
    let start_unit = indices
        .first()
        .copied()
        .ok_or(TokenizerError::InvalidTrainingEncoding(
            "surface segment has no first unit",
        ))?;
    let end_unit = indices
        .last()
        .copied()
        .and_then(|index| index.checked_add(1))
        .ok_or(TokenizerError::LengthOverflow("surface segment unit end"))?;
    let mut surface_ids = Vec::new();
    let mut surface_lengths = Vec::new();
    vocabulary.encode_flat_range_into(
        raw,
        units,
        cuts,
        start_unit..end_unit,
        maximum_chars,
        use_morphology,
        &mut surface_ids,
        &mut surface_lengths,
    )?;
    Ok(FlatSegmentProgram {
        tokens: Box::new([]),
        relative_cuts: Box::new([]),
        surface_ids: surface_ids.into_boxed_slice(),
        surface_lengths: surface_lengths.into_boxed_slice(),
    })
}

#[cfg(test)]
mod tests {
    use super::{next_code_line_end, scan_fixed_units, split_units, FlatSurfaceUnit};
    use crate::{TokenizerError, TokenizerMode};
    use nedo_core::scan_compact;

    #[test]
    fn flat_surface_unit_stays_compact() {
        assert!(std::mem::size_of::<FlatSurfaceUnit>() <= 40);
    }

    #[test]
    fn fixed_stream_matches_compact_split_exactly() -> Result<(), TokenizerError> {
        let mut inputs = vec![
            Vec::new(),
            b"a != b && HTTPServer42::run()".to_vec(),
            "Ankara'da İstanbul’daydı 2026'da...".as_bytes().to_vec(),
            (u8::MIN..=u8::MAX).collect(),
        ];
        let mut state = 0xbb67_ae85_84ca_a73b_u64;
        let mut random = Vec::with_capacity(8192);
        for _ in 0..8192 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            random.push(state.to_le_bytes()[0]);
        }
        inputs.push(random);
        for raw in inputs {
            for mode in [TokenizerMode::Turkish, TokenizerMode::Code] {
                let compact = scan_compact(raw.clone())?;
                let expected = split_units(&compact, &[], mode)?;
                let actual = scan_fixed_units(&raw, mode)?;
                assert_eq!(actual, expected);
            }
        }
        Ok(())
    }
    #[test]
    fn code_line_end_handles_every_scanner_line_break() {
        let raw = b"a\r\nb\rc\nd\xc2\x85e\xe2\x80\xa8f\xe2\x80\xa9g";
        let mut starts = Vec::new();
        let mut start = 0_usize;
        while start < raw.len() {
            starts.push(start);
            start = next_code_line_end(raw, start);
        }
        assert_eq!(starts, vec![0, 3, 5, 7, 10, 14, 18]);
        assert_eq!(next_code_line_end(b"", 0), 0);
    }
}
