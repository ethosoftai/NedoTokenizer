//! `NedoFormer` inner-encoder input contract.
//!
//! The inner Mamba sees original character/byte IDs plus explicit unit and
//! morpheme-boundary markers.  `segment_offsets` describe independent recurrent
//! segments: state must reset before every segment.  Orthographic whitespace
//! outside a phonological group is retained in the character stream but is not
//! pooled into an outer-model vector.  Clitic whitespace remains inside the
//! shared phonological segment because the tokenizer's `group_id` joins it to
//! the preceding host.

use super::{
    ByteSpan, CharacterVocabulary, LexicalKind, NedoFormerLatticeDocument,
    NedoFormerLatticeSidecar, NedoFormerSamplingPolicy, TokenMode, TokenizedDocument,
    TokenizerError, BYTE_BASE_ID, CODE_END_ID, CODE_START_ID, MORPHEME_BOUNDARY_ID,
    UNIT_BOUNDARY_ID,
};

/// One immutable inner-Mamba input stream plus recurrent reset/pooling metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NedoFormerInputEncoding {
    /// Character/byte/special IDs. No document BOS/EOS is inserted here: each
    /// recurrent segment is explicitly delimited by `segment_offsets`.
    pub ids: Vec<u32>,
    /// Prefix offsets into `ids`; length is number of recurrent segments + 1.
    /// Mamba state resets before every segment `i`, at `segment_offsets[i]`.
    pub segment_offsets: Vec<u32>,
    /// Segment indices whose terminal state becomes an outer-model vector.
    /// Pure inter-word whitespace/line-break segments are intentionally absent.
    pub pooled_segments: Vec<u32>,
    /// Exact raw byte span corresponding to each pooled segment.
    pub pool_spans: Vec<ByteSpan>,
    /// Processing mode for each pooled segment.
    pub pool_modes: Vec<TokenMode>,
    /// Phonological group ID for each pooled segment; code/opaque segments may be `None`.
    pub pool_group_ids: Vec<Option<u32>>,
}

#[derive(Clone, Debug)]
struct InputUnit {
    span: ByteSpan,
    kind: LexicalKind,
    mode: TokenMode,
    group_id: Option<u32>,
    cuts: Vec<u64>,
}

impl CharacterVocabulary {
    /// Encodes a selected rich document for the `NedoFormer` inner Mamba.
    ///
    /// # Errors
    ///
    /// Returns an error when byte offsets cannot be represented, a unit slice is
    /// invalid, or generated stream offsets exceed `u32`.
    pub fn encode_nedoformer_input(
        &self,
        document: &TokenizedDocument,
    ) -> Result<NedoFormerInputEncoding, TokenizerError> {
        let units = document
            .units()
            .iter()
            .map(|unit| InputUnit {
                span: unit.span,
                kind: unit.kind,
                mode: unit.mode,
                group_id: unit.group_id,
                cuts: unit.cuts.clone(),
            })
            .collect::<Vec<_>>();
        encode_units(self, document.raw(), &units)
    }
}

impl NedoFormerLatticeDocument {
    /// Samples one lattice path and immediately emits the inner-Mamba input stream.
    ///
    /// # Errors
    ///
    /// Returns any sampling, document-validation, or input-encoding error.
    pub fn sample_input_encoding(
        &self,
        vocabulary: &CharacterVocabulary,
        policy: NedoFormerSamplingPolicy,
        seed: u64,
    ) -> Result<NedoFormerInputEncoding, TokenizerError> {
        let document = self.sample(policy, seed)?;
        vocabulary.encode_nedoformer_input(&document)
    }
}

impl NedoFormerLatticeSidecar {
    /// Samples a metadata-only sidecar and emits exactly the same inner-Mamba
    /// input contract as the corresponding rich lattice path.
    ///
    /// # Errors
    ///
    /// Returns any sampling, span, or input-encoding error.
    pub fn sample_input_encoding(
        &self,
        vocabulary: &CharacterVocabulary,
        policy: NedoFormerSamplingPolicy,
        seed: u64,
    ) -> Result<NedoFormerInputEncoding, TokenizerError> {
        let sampled = self.sample_lossless(policy, seed)?;
        if sampled.units().len() != self.units().len() {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "NedoFormer sidecar sampled unit cardinality differs",
            ));
        }
        let units = self
            .units()
            .iter()
            .zip(sampled.units())
            .map(|(metadata, surface)| InputUnit {
                span: surface.span,
                kind: metadata.kind,
                mode: metadata.mode,
                group_id: metadata.group_id,
                cuts: surface.cuts.clone(),
            })
            .collect::<Vec<_>>();
        encode_units(vocabulary, self.raw(), &units)
    }
}

fn encode_units(
    vocabulary: &CharacterVocabulary,
    raw: &[u8],
    units: &[InputUnit],
) -> Result<NedoFormerInputEncoding, TokenizerError> {
    let mut ids = Vec::new();
    let mut segment_offsets = vec![0_u32];
    let mut pooled_segments = Vec::new();
    let mut pool_spans = Vec::new();
    let mut pool_modes = Vec::new();
    let mut pool_group_ids = Vec::new();
    let mut index = 0_usize;

    while index < units.len() {
        let first = &units[index];
        let group = first.group_id;
        let mut end = index + 1;
        if let Some(group_id) = group {
            while end < units.len() && units[end].group_id == Some(group_id) {
                end += 1;
            }
        }
        let segment = &units[index..end];
        let mode = first.mode;
        if group.is_some() && segment.iter().any(|unit| unit.mode != mode) {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "NedoFormer phonological group crosses processing modes",
            ));
        }
        let pool = segment
            .iter()
            .any(|unit| !matches!(unit.kind, LexicalKind::Whitespace | LexicalKind::LineBreak));
        let segment_start = first.span.start;
        let segment_end = segment.last().map(|unit| unit.span.end).ok_or(
            TokenizerError::InvalidTrainingEncoding("NedoFormer input segment is empty"),
        )?;

        if mode == TokenMode::Code {
            ids.push(CODE_START_ID);
        }
        for unit in segment {
            ids.push(UNIT_BOUNDARY_ID);
            encode_unit(vocabulary, raw, unit, &mut ids)?;
        }
        if mode == TokenMode::Code {
            ids.push(CODE_END_ID);
        }

        let id_end = u32::try_from(ids.len())
            .map_err(|_| TokenizerError::LengthOverflow("NedoFormer input ID stream"))?;
        segment_offsets.push(id_end);
        if pool {
            let segment_index = u32::try_from(segment_offsets.len().saturating_sub(2))
                .map_err(|_| TokenizerError::LengthOverflow("NedoFormer input segment index"))?;
            pooled_segments.push(segment_index);
            pool_spans.push(ByteSpan {
                start: segment_start,
                end: segment_end,
            });
            pool_modes.push(mode);
            pool_group_ids.push(group);
        }
        index = end;
    }

    if pool_spans.len() != pooled_segments.len()
        || pool_modes.len() != pooled_segments.len()
        || pool_group_ids.len() != pooled_segments.len()
    {
        return Err(TokenizerError::InvalidTrainingEncoding(
            "NedoFormer input pooling metadata cardinality differs",
        ));
    }
    Ok(NedoFormerInputEncoding {
        ids,
        segment_offsets,
        pooled_segments,
        pool_spans,
        pool_modes,
        pool_group_ids,
    })
}

fn encode_unit(
    vocabulary: &CharacterVocabulary,
    raw: &[u8],
    unit: &InputUnit,
    ids: &mut Vec<u32>,
) -> Result<(), TokenizerError> {
    let start = usize::try_from(unit.span.start)
        .map_err(|_| TokenizerError::LengthOverflow("NedoFormer input unit start"))?;
    let end = usize::try_from(unit.span.end)
        .map_err(|_| TokenizerError::LengthOverflow("NedoFormer input unit end"))?;
    let bytes = raw
        .get(start..end)
        .ok_or(TokenizerError::UnitOutsideDocument)?;
    let mut cuts = unit.cuts.iter().copied().peekable();

    if let Ok(text) = std::str::from_utf8(bytes) {
        for (relative, value) in text.char_indices() {
            let absolute = unit
                .span
                .start
                .checked_add(u64::try_from(relative).map_err(|_| {
                    TokenizerError::LengthOverflow("NedoFormer input character offset")
                })?)
                .ok_or(TokenizerError::LengthOverflow(
                    "NedoFormer input character offset",
                ))?;
            emit_cuts(&mut cuts, absolute, ids);
            if let Some(id) = vocabulary.id_for_char(value) {
                ids.push(id);
            } else {
                let width = value.len_utf8();
                let local = bytes.get(relative..relative + width).ok_or(
                    TokenizerError::InvalidTrainingEncoding(
                        "NedoFormer UTF-8 character slice escaped unit",
                    ),
                )?;
                ids.extend(local.iter().map(|byte| BYTE_BASE_ID + u32::from(*byte)));
            }
        }
    } else {
        for (relative, byte) in bytes.iter().copied().enumerate() {
            let absolute =
                unit.span
                    .start
                    .checked_add(u64::try_from(relative).map_err(|_| {
                        TokenizerError::LengthOverflow("NedoFormer input byte offset")
                    })?)
                    .ok_or(TokenizerError::LengthOverflow(
                        "NedoFormer input byte offset",
                    ))?;
            emit_cuts(&mut cuts, absolute, ids);
            ids.push(BYTE_BASE_ID + u32::from(byte));
        }
    }
    if cuts.next().is_some() {
        return Err(TokenizerError::InvalidTrainingEncoding(
            "NedoFormer input cut was not aligned to a character/byte boundary",
        ));
    }
    Ok(())
}

fn emit_cuts<I>(cuts: &mut std::iter::Peekable<I>, absolute: u64, ids: &mut Vec<u32>)
where
    I: Iterator<Item = u64>,
{
    while cuts.peek().is_some_and(|cut| *cut == absolute) {
        ids.push(MORPHEME_BOUNDARY_ID);
        cuts.next();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NedoFormerVocabulary, Tokenizer, TokenizerConfig};

    #[test]
    fn clitic_group_is_one_recurrent_segment_and_sidecar_matches_rich_sampling(
    ) -> Result<(), TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let raw = "geliyor mu? koyun çocuklarımızdan\n```rust\nfoo_bar42\n```"
            .as_bytes()
            .to_vec();
        let lattice = tokenizer.nedoformer_lattice(raw.clone())?;
        let selected = lattice.selected_document()?;
        let characters = CharacterVocabulary::train(std::slice::from_ref(&selected), 500);
        let best = characters.encode_nedoformer_input(&selected)?;
        assert_eq!(best.segment_offsets.first(), Some(&0));
        assert_eq!(
            best.segment_offsets.last().copied(),
            u32::try_from(best.ids.len()).ok()
        );
        assert!(best.pool_spans.iter().any(|span| &raw
            [usize::try_from(span.start).unwrap()..usize::try_from(span.end).unwrap()]
            == "geliyor mu".as_bytes()));

        let sidecar_bytes = lattice.to_sidecar_bytes()?;
        let sidecar = NedoFormerLatticeSidecar::from_bytes(raw.clone(), &sidecar_bytes)?;
        for policy in [
            NedoFormerSamplingPolicy::Best,
            NedoFormerSamplingPolicy::Uniform,
            NedoFormerSamplingPolicy::ContextWeighted { temperature: 0.9 },
        ] {
            let rich = lattice.sample_input_encoding(&characters, policy, 2026)?;
            let compact = sidecar.sample_input_encoding(&characters, policy, 2026)?;
            assert_eq!(rich, compact);
        }

        // Output-vocabulary training remains a separate decoder-side contract.
        let generation = NedoFormerVocabulary::train(std::slice::from_ref(&selected), 64, 500, 64)?;
        assert!(!generation.is_empty());
        Ok(())
    }

    #[test]
    fn byte_fallback_stream_never_loses_invalid_utf8() -> Result<(), TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let raw = b"abc\xffdef".to_vec();
        let lattice = tokenizer.nedoformer_lattice(raw)?;
        let selected = lattice.selected_document()?;
        let characters = CharacterVocabulary::train(&[], 0);
        let encoded = characters.encode_nedoformer_input(&selected)?;
        assert!(encoded
            .ids
            .iter()
            .any(|id| *id >= BYTE_BASE_ID && *id < crate::CHAR_BASE_ID));
        Ok(())
    }
}
