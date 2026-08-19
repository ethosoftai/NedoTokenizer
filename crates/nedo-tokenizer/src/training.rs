//! Canonical, checked conversion from rich tokenizer output to flat training streams.

use crate::{
    CharacterVocabulary, TokenMode, TokenizedDocument, TokenizedUnit, TokenizerError, BOS_ID,
    BYTE_BASE_ID, CODE_END_ID, CODE_START_ID, EOS_ID, MORPHEME_BOUNDARY_ID, UNIT_BOUNDARY_ID,
};

/// Structural controls emitted by the rich-reference training encoder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TrainingEncodingOptions {
    /// Emit one lexical-unit boundary before every rich unit.
    pub emit_unit_boundaries: bool,
    /// Emit selected morpheme cuts inside each unit.
    pub emit_morpheme_boundaries: bool,
    /// Emit code-mode entry and exit controls.
    pub emit_code_boundaries: bool,
}

impl Default for TrainingEncodingOptions {
    fn default() -> Self {
        Self {
            emit_unit_boundaries: true,
            emit_morpheme_boundaries: true,
            emit_code_boundaries: true,
        }
    }
}

/// One document encoded for the flat training pack format.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainingEncoding {
    /// Stable token IDs, checked to fit the on-disk `u16` representation.
    pub ids: Vec<u16>,
    /// Exact source-byte contribution of every token.
    pub lengths: Vec<u8>,
}

/// Concatenated rich-reference encodings and document token offsets.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainingBatch {
    /// Concatenated token IDs.
    pub ids: Vec<u16>,
    /// Concatenated byte lengths.
    pub lengths: Vec<u8>,
    /// Token offsets with one leading zero and one final end offset.
    pub document_offsets: Vec<u64>,
}

impl CharacterVocabulary {
    /// Encodes one byte document without tokenizer structural boundaries.
    ///
    /// # Errors
    ///
    /// Returns an error for ID/length overflow or inconsistent byte accounting.
    pub fn encode_character_training_document(
        &self,
        raw: &[u8],
        newline: bool,
    ) -> Result<TrainingEncoding, TokenizerError> {
        let capacity = raw
            .len()
            .checked_add(3)
            .ok_or(TokenizerError::LengthOverflow(
                "character training capacity",
            ))?;
        let mut output = TrainingEncoding {
            ids: Vec::with_capacity(capacity),
            lengths: Vec::with_capacity(capacity),
        };
        push_token(&mut output, BOS_ID, 0)?;
        SurfaceEncoder::new(self, raw, &mut output).encode(0, raw.len(), &[], false)?;
        if newline {
            push_token(&mut output, BYTE_BASE_ID + u32::from(b'\n'), 1)?;
        }
        push_token(&mut output, EOS_ID, 0)?;
        validate_byte_accounting(&output, raw.len(), newline)?;
        Ok(output)
    }

    /// Converts one validated rich tokenized document into the canonical training stream.
    ///
    /// The result is the exact reference that future metadata-free/flat tokenizer paths
    /// must match for IDs, lengths, token count, and byte accounting.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid spans/cuts, overflow, or byte-accounting mismatch.
    pub fn encode_training_document(
        &self,
        document: &TokenizedDocument,
        newline: bool,
        options: TrainingEncodingOptions,
    ) -> Result<TrainingEncoding, TokenizerError> {
        document.validate()?;
        self.encode_training_units(document.raw(), document.units(), newline, options)
    }

    pub(crate) fn encode_training_units(
        &self,
        raw: &[u8],
        units: &[TokenizedUnit],
        newline: bool,
        options: TrainingEncodingOptions,
    ) -> Result<TrainingEncoding, TokenizerError> {
        validate_flat_units(raw, units)?;
        let structural_capacity =
            units
                .len()
                .checked_mul(2)
                .ok_or(TokenizerError::LengthOverflow(
                    "training structural capacity",
                ))?;
        let capacity = raw
            .len()
            .checked_add(structural_capacity)
            .and_then(|value| value.checked_add(3))
            .ok_or(TokenizerError::LengthOverflow("training encoding capacity"))?;
        let mut output = TrainingEncoding {
            ids: Vec::with_capacity(capacity),
            lengths: Vec::with_capacity(capacity),
        };
        push_token(&mut output, BOS_ID, 0)?;
        let mut in_code = false;
        for unit in units {
            if options.emit_code_boundaries {
                if unit.mode == TokenMode::Code && !in_code {
                    push_token(&mut output, CODE_START_ID, 0)?;
                    in_code = true;
                } else if unit.mode != TokenMode::Code && in_code {
                    push_token(&mut output, CODE_END_ID, 0)?;
                    in_code = false;
                }
            }
            if options.emit_unit_boundaries {
                push_token(&mut output, UNIT_BOUNDARY_ID, 0)?;
            }
            let start = usize::try_from(unit.span.start)
                .map_err(|_| TokenizerError::LengthOverflow("training unit start"))?;
            let end = usize::try_from(unit.span.end)
                .map_err(|_| TokenizerError::LengthOverflow("training unit end"))?;
            let cuts = if options.emit_morpheme_boundaries {
                unit.cuts.as_slice()
            } else {
                &[]
            };
            SurfaceEncoder::new(self, raw, &mut output).encode(
                start,
                end,
                cuts,
                options.emit_morpheme_boundaries,
            )?;
        }
        if options.emit_code_boundaries && in_code {
            push_token(&mut output, CODE_END_ID, 0)?;
        }
        if newline {
            push_token(&mut output, BYTE_BASE_ID + u32::from(b'\n'), 1)?;
        }
        push_token(&mut output, EOS_ID, 0)?;
        validate_byte_accounting(&output, raw.len(), newline)?;
        Ok(output)
    }

    /// Encodes rich documents and constructs exact token offsets in input order.
    ///
    /// # Errors
    ///
    /// Returns an error if newline cardinality differs, a document fails encoding,
    /// or concatenated token offsets overflow `u64`.
    pub fn encode_training_batch(
        &self,
        documents: &[TokenizedDocument],
        newline_flags: &[bool],
        options: TrainingEncodingOptions,
    ) -> Result<TrainingBatch, TokenizerError> {
        if documents.len() != newline_flags.len() {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "document and newline counts differ",
            ));
        }
        let estimated_tokens = documents.iter().try_fold(0_usize, |total, document| {
            let document_tokens = document
                .raw()
                .len()
                .checked_add(1)
                .ok_or(TokenizerError::LengthOverflow("training document capacity"))?;
            total
                .checked_add(document_tokens)
                .ok_or(TokenizerError::LengthOverflow("training batch capacity"))
        })?;
        let mut batch = TrainingBatch {
            ids: Vec::with_capacity(estimated_tokens),
            lengths: Vec::with_capacity(estimated_tokens),
            document_offsets: Vec::with_capacity(
                documents
                    .len()
                    .checked_add(1)
                    .ok_or(TokenizerError::LengthOverflow("training offset capacity"))?,
            ),
        };
        batch.document_offsets.push(0);
        for (document, newline) in documents.iter().zip(newline_flags) {
            let encoded = self.encode_training_document(document, *newline, options)?;
            batch.ids.extend_from_slice(&encoded.ids);
            batch.lengths.extend_from_slice(&encoded.lengths);
            batch.document_offsets.push(
                u64::try_from(batch.ids.len())
                    .map_err(|_| TokenizerError::LengthOverflow("training token offset"))?,
            );
        }
        Ok(batch)
    }
}

/// Encodes arbitrary bytes directly with fixed byte-fallback IDs.
///
/// # Errors
///
/// Returns an error for token ID overflow or inconsistent byte accounting.
pub fn encode_byte_training_document(
    raw: &[u8],
    newline: bool,
) -> Result<TrainingEncoding, TokenizerError> {
    let capacity = raw
        .len()
        .checked_add(3)
        .ok_or(TokenizerError::LengthOverflow("byte training capacity"))?;
    let mut output = TrainingEncoding {
        ids: Vec::with_capacity(capacity),
        lengths: Vec::with_capacity(capacity),
    };
    push_token(&mut output, BOS_ID, 0)?;
    for byte in raw {
        push_token(&mut output, BYTE_BASE_ID + u32::from(*byte), 1)?;
    }
    if newline {
        push_token(&mut output, BYTE_BASE_ID + u32::from(b'\n'), 1)?;
    }
    push_token(&mut output, EOS_ID, 0)?;
    validate_byte_accounting(&output, raw.len(), newline)?;
    Ok(output)
}

fn validate_flat_units(raw: &[u8], units: &[TokenizedUnit]) -> Result<(), TokenizerError> {
    let document_len = u64::try_from(raw.len())
        .map_err(|_| TokenizerError::LengthOverflow("flat training document length"))?;
    let mut expected = 0_u64;
    for (index, unit) in units.iter().enumerate() {
        if unit.span.start != expected
            || unit.span.end <= unit.span.start
            || unit.span.end > document_len
        {
            return Err(TokenizerError::InvalidUnitCoverage {
                index,
                expected,
                start: unit.span.start,
                end: unit.span.end,
            });
        }
        let mut previous = unit.span.start;
        for cut in &unit.cuts {
            if *cut <= previous || *cut >= unit.span.end {
                return Err(TokenizerError::InvalidTrainingEncoding(
                    "flat training cut is outside its unit or not increasing",
                ));
            }
            previous = *cut;
        }
        expected = unit.span.end;
    }
    if expected != document_len {
        return Err(TokenizerError::IncompleteDocument {
            covered: expected,
            document_len,
        });
    }
    Ok(())
}

fn validate_byte_accounting(
    output: &TrainingEncoding,
    raw_bytes: usize,
    newline: bool,
) -> Result<(), TokenizerError> {
    if output.ids.len() != output.lengths.len() {
        return Err(TokenizerError::InvalidTrainingEncoding(
            "ID and byte-length cardinalities differ",
        ));
    }
    let expected = raw_bytes
        .checked_add(usize::from(newline))
        .ok_or(TokenizerError::LengthOverflow("training expected bytes"))?;
    let actual = output.lengths.iter().try_fold(0_usize, |total, length| {
        total
            .checked_add(usize::from(*length))
            .ok_or(TokenizerError::LengthOverflow("training byte accounting"))
    })?;
    if actual != expected {
        return Err(TokenizerError::InvalidTrainingEncoding(
            "training byte accounting differs from source bytes",
        ));
    }
    Ok(())
}

fn push_token(output: &mut TrainingEncoding, id: u32, length: u8) -> Result<(), TokenizerError> {
    output
        .ids
        .push(u16::try_from(id).map_err(|_| TokenizerError::LengthOverflow("training token ID"))?);
    output.lengths.push(length);
    Ok(())
}

struct SurfaceEncoder<'a> {
    vocabulary: &'a CharacterVocabulary,
    raw: &'a [u8],
    output: &'a mut TrainingEncoding,
}

impl<'a> SurfaceEncoder<'a> {
    const fn new(
        vocabulary: &'a CharacterVocabulary,
        raw: &'a [u8],
        output: &'a mut TrainingEncoding,
    ) -> Self {
        Self {
            vocabulary,
            raw,
            output,
        }
    }

    fn encode(
        &mut self,
        start: usize,
        end: usize,
        cuts: &[u64],
        emit_cuts: bool,
    ) -> Result<(), TokenizerError> {
        if start > end || end > self.raw.len() {
            return Err(TokenizerError::UnitOutsideDocument);
        }
        let mut position = start;
        let mut cut_index = 0_usize;
        while position < end {
            if emit_cuts {
                let absolute = u64::try_from(position)
                    .map_err(|_| TokenizerError::LengthOverflow("training byte position"))?;
                while cuts.get(cut_index) == Some(&absolute) {
                    push_token(self.output, MORPHEME_BOUNDARY_ID, 0)?;
                    cut_index += 1;
                }
            }
            if let Some((value, width)) = decode_scalar(self.raw, position) {
                let scalar_end = position
                    .checked_add(width)
                    .ok_or(TokenizerError::LengthOverflow("training scalar end"))?;
                if scalar_end <= end {
                    if let Some(id) = self.vocabulary.id_for_char(value) {
                        let length = u8::try_from(width)
                            .map_err(|_| TokenizerError::LengthOverflow("UTF-8 scalar width"))?;
                        push_token(self.output, id, length)?;
                    } else {
                        for byte in &self.raw[position..scalar_end] {
                            push_token(self.output, BYTE_BASE_ID + u32::from(*byte), 1)?;
                        }
                    }
                    position = scalar_end;
                    continue;
                }
            }
            push_token(self.output, BYTE_BASE_ID + u32::from(self.raw[position]), 1)?;
            position += 1;
        }
        if emit_cuts && cut_index != cuts.len() {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "morpheme cut is not an encoded scalar boundary",
            ));
        }
        Ok(())
    }
}

fn decode_scalar(raw: &[u8], position: usize) -> Option<(char, usize)> {
    let tail = raw.get(position..)?;
    let width = match *tail.first()? {
        0x00..=0x7f => 1,
        0xc2..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf4 => 4,
        _ => return None,
    };
    let bytes = tail.get(..width)?;
    let value = std::str::from_utf8(bytes).ok()?.chars().next()?;
    Some((value, width))
}

#[cfg(test)]
mod tests {
    use nedo_core::LexicalKind;
    use nedo_format::ByteSpan;

    use super::{encode_byte_training_document, CharacterVocabulary, TrainingEncodingOptions};
    use crate::{
        TokenMode, TokenStatus, TokenizedDocument, TokenizedUnit, BOS_ID, BYTE_BASE_ID,
        CHAR_BASE_ID, CODE_END_ID, CODE_START_ID, EOS_ID, MORPHEME_BOUNDARY_ID, UNIT_BOUNDARY_ID,
    };

    fn id(value: u32) -> Result<u16, crate::TokenizerError> {
        u16::try_from(value).map_err(|_| crate::TokenizerError::LengthOverflow("test token ID"))
    }

    fn unit(
        start: u64,
        end: u64,
        kind: LexicalKind,
        mode: TokenMode,
        status: TokenStatus,
        cuts: Vec<u64>,
    ) -> TokenizedUnit {
        TokenizedUnit {
            span: ByteSpan { start, end },
            kind,
            mode,
            status,
            group_id: None,
            cuts,
            analysis: None,
        }
    }

    #[test]
    fn rich_reference_emits_exact_boundaries_and_lengths() -> Result<(), crate::TokenizerError> {
        let vocabulary = CharacterVocabulary::from_sorted(vec!['a', 'b', 'c'])?;
        let document = TokenizedDocument::new(
            b"abc".to_vec(),
            vec![unit(
                0,
                3,
                LexicalKind::Word,
                TokenMode::Turkish,
                TokenStatus::Structural,
                vec![1],
            )],
        )?;
        let encoded = vocabulary.encode_training_document(
            &document,
            true,
            TrainingEncodingOptions::default(),
        )?;
        assert_eq!(
            encoded.ids,
            vec![
                id(BOS_ID)?,
                id(UNIT_BOUNDARY_ID)?,
                id(CHAR_BASE_ID)?,
                id(MORPHEME_BOUNDARY_ID)?,
                id(CHAR_BASE_ID + 1)?,
                id(CHAR_BASE_ID + 2)?,
                id(BYTE_BASE_ID + u32::from(b'\n'))?,
                id(EOS_ID)?,
            ]
        );
        assert_eq!(encoded.lengths, vec![0, 0, 1, 0, 1, 1, 1, 0]);
        Ok(())
    }

    #[test]
    fn code_transitions_are_exact_and_closed() -> Result<(), crate::TokenizerError> {
        let vocabulary = CharacterVocabulary::from_sorted(vec!['a', 'b', 'c'])?;
        let document = TokenizedDocument::new(
            b"abc".to_vec(),
            vec![
                unit(
                    0,
                    1,
                    LexicalKind::Word,
                    TokenMode::Code,
                    TokenStatus::Code,
                    vec![],
                ),
                unit(
                    1,
                    2,
                    LexicalKind::Word,
                    TokenMode::Turkish,
                    TokenStatus::Structural,
                    vec![],
                ),
                unit(
                    2,
                    3,
                    LexicalKind::Word,
                    TokenMode::Code,
                    TokenStatus::Code,
                    vec![],
                ),
            ],
        )?;
        let encoded = vocabulary.encode_training_document(
            &document,
            false,
            TrainingEncodingOptions::default(),
        )?;
        assert_eq!(
            encoded.ids,
            vec![
                id(BOS_ID)?,
                id(CODE_START_ID)?,
                id(UNIT_BOUNDARY_ID)?,
                id(CHAR_BASE_ID)?,
                id(CODE_END_ID)?,
                id(UNIT_BOUNDARY_ID)?,
                id(CHAR_BASE_ID + 1)?,
                id(CODE_START_ID)?,
                id(UNIT_BOUNDARY_ID)?,
                id(CHAR_BASE_ID + 2)?,
                id(CODE_END_ID)?,
                id(EOS_ID)?,
            ]
        );
        assert_eq!(encoded.lengths, vec![0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]);
        Ok(())
    }

    #[test]
    fn learned_unicode_and_byte_fallback_preserve_widths() -> Result<(), crate::TokenizerError> {
        let vocabulary = CharacterVocabulary::from_sorted(vec!['ç'])?;
        let raw = "ç🙂".as_bytes().to_vec();
        let document = TokenizedDocument::new(
            raw,
            vec![unit(
                0,
                6,
                LexicalKind::Word,
                TokenMode::Turkish,
                TokenStatus::Structural,
                vec![],
            )],
        )?;
        let encoded = vocabulary.encode_training_document(
            &document,
            false,
            TrainingEncodingOptions::default(),
        )?;
        assert_eq!(encoded.ids[0], id(BOS_ID)?);
        assert_eq!(encoded.ids[1], id(UNIT_BOUNDARY_ID)?);
        assert_eq!(encoded.ids[2], id(CHAR_BASE_ID)?);
        assert_eq!(encoded.lengths, vec![0, 0, 2, 1, 1, 1, 1, 0]);
        assert_eq!(encoded.ids.last().copied(), Some(id(EOS_ID)?));
        Ok(())
    }

    #[test]
    fn invalid_utf8_is_byte_exact() -> Result<(), crate::TokenizerError> {
        let vocabulary = CharacterVocabulary::from_sorted(vec!['a'])?;
        let raw = vec![0xff, b'a'];
        let document = TokenizedDocument::new(
            raw,
            vec![unit(
                0,
                2,
                LexicalKind::Opaque,
                TokenMode::Opaque,
                TokenStatus::Opaque,
                vec![],
            )],
        )?;
        let encoded = vocabulary.encode_training_document(
            &document,
            false,
            TrainingEncodingOptions::default(),
        )?;
        assert_eq!(encoded.ids[0], id(BOS_ID)?);
        assert_eq!(encoded.ids[1], id(UNIT_BOUNDARY_ID)?);
        assert_eq!(encoded.ids[2], id(BYTE_BASE_ID + 0xff)?);
        assert_eq!(encoded.ids[3], id(CHAR_BASE_ID)?);
        assert_eq!(encoded.lengths, vec![0, 0, 1, 1, 0]);
        Ok(())
    }

    #[test]
    fn ablation_options_remove_only_requested_controls() -> Result<(), crate::TokenizerError> {
        let vocabulary = CharacterVocabulary::from_sorted(vec!['a', 'b'])?;
        let document = TokenizedDocument::new(
            b"ab".to_vec(),
            vec![unit(
                0,
                2,
                LexicalKind::Word,
                TokenMode::Code,
                TokenStatus::Code,
                vec![1],
            )],
        )?;
        let encoded = vocabulary.encode_training_document(
            &document,
            false,
            TrainingEncodingOptions {
                emit_unit_boundaries: false,
                emit_morpheme_boundaries: false,
                emit_code_boundaries: false,
            },
        )?;
        assert_eq!(
            encoded.ids,
            vec![
                id(BOS_ID)?,
                id(CHAR_BASE_ID)?,
                id(CHAR_BASE_ID + 1)?,
                id(EOS_ID)?,
            ]
        );
        assert_eq!(encoded.lengths, vec![0, 1, 1, 0]);
        Ok(())
    }

    #[test]
    fn batch_offsets_match_concatenated_documents() -> Result<(), crate::TokenizerError> {
        let vocabulary = CharacterVocabulary::from_sorted(vec!['a', 'b'])?;
        let first = TokenizedDocument::new(
            b"a".to_vec(),
            vec![unit(
                0,
                1,
                LexicalKind::Word,
                TokenMode::Turkish,
                TokenStatus::Structural,
                vec![],
            )],
        )?;
        let second = TokenizedDocument::new(
            b"b".to_vec(),
            vec![unit(
                0,
                1,
                LexicalKind::Word,
                TokenMode::Turkish,
                TokenStatus::Structural,
                vec![],
            )],
        )?;
        let batch = vocabulary.encode_training_batch(
            &[first, second],
            &[false, true],
            TrainingEncodingOptions::default(),
        )?;
        assert_eq!(batch.document_offsets, vec![0, 4, 9]);
        assert_eq!(batch.ids.len(), batch.lengths.len());
        assert_eq!(
            batch
                .lengths
                .iter()
                .map(|value| usize::from(*value))
                .sum::<usize>(),
            3
        );
        Ok(())
    }

    #[test]
    fn byte_encoder_preserves_every_input_byte() -> Result<(), crate::TokenizerError> {
        let encoded = encode_byte_training_document(&[0x00, 0xff], true)?;
        assert_eq!(
            encoded.ids,
            vec![
                id(BOS_ID)?,
                id(BYTE_BASE_ID)?,
                id(BYTE_BASE_ID + 0xff)?,
                id(BYTE_BASE_ID + u32::from(b'\n'))?,
                id(EOS_ID)?,
            ]
        );
        assert_eq!(encoded.lengths, vec![0, 1, 1, 1, 0]);
        Ok(())
    }
}
