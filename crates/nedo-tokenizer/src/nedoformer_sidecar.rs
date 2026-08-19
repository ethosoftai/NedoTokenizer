//! Compact metadata-only `NedoFormer` segmentation sidecar.
//!
//! Raw corpus bytes are not serialized. The sidecar pins their length and SHA-256,
//! then stores contiguous unit lengths, packed routing/group metadata, and exact cut
//! alternatives. Common single-candidate units do not store redundant scores or
//! selected flags; weighted scores are retained only where sampling can differ.

use sha2::{Digest, Sha256};

use super::{
    ByteSpan, LexicalKind, NedoFormerLatticeDocument, NedoFormerSamplingPolicy, TokenMode,
    TokenizerError, MODEL_SHA256, MORPHOLOGY_SHA256,
};

/// Stable metadata-only sidecar codec revision.
pub const NEDOFORMER_SIDECAR_SCHEMA_VERSION: u32 = 2;

const SIDECAR_MAGIC: &[u8; 8] = b"NDFSID02";
const HEADER_BYTES: usize = 8 + 4 + 32 + 32;
const KIND_MASK: u8 = 0b0000_0111;
const MODE_MASK: u8 = 0b0001_1000;
const GROUP_MASK: u8 = 0b0110_0000;
const AMBIGUOUS_FLAG: u8 = 0b1000_0000;
const GROUP_NONE: u8 = 0;
const GROUP_NEW: u8 = 1;
const GROUP_SAME: u8 = 2;

/// One compact cut-class candidate in the metadata-only sidecar.
#[derive(Clone, Debug, PartialEq)]
pub struct NedoFormerSidecarCandidate {
    /// Strictly increasing cuts in original-document byte offsets.
    pub cuts: Vec<u64>,
    /// Conditional perceptron log-score; meaningful when the unit has >1 candidate.
    pub conditional_log_score: f32,
    /// Whether the deterministic Viterbi path selects this class.
    pub selected: bool,
}

/// One compact surface unit in a `NedoFormer` sidecar.
#[derive(Clone, Debug, PartialEq)]
pub struct NedoFormerSidecarUnit {
    /// Exact byte span in the external raw document.
    pub span: ByteSpan,
    /// Scanner class.
    pub kind: LexicalKind,
    /// Turkish/code/opaque processing mode.
    pub mode: TokenMode,
    /// Inner-model phonological group identifier.
    pub group_id: Option<u32>,
    /// Distinct segmentation candidates in the same order as the rich lattice.
    pub candidates: Vec<NedoFormerSidecarCandidate>,
}

/// Validated metadata sidecar paired with externally supplied raw bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct NedoFormerLatticeSidecar {
    raw: Vec<u8>,
    units: Vec<NedoFormerSidecarUnit>,
}

impl NedoFormerLatticeDocument {
    /// Serializes a metadata-only sidecar without copying corpus text or analysis strings.
    ///
    /// The codec intentionally drops candidate status/analysis multiplicity because
    /// neither affects sampled cuts nor the inner-Mamba input. Full diagnostic metadata
    /// remains available in the self-contained rich lattice codec.
    ///
    /// # Errors
    ///
    /// Returns an error if grouping is non-canonical, counts overflow, or the rich
    /// selected path is invalid.
    pub fn to_sidecar_bytes(&self) -> Result<Vec<u8>, TokenizerError> {
        let _ = self.selected_document()?;
        let mut payload = Vec::with_capacity(self.raw().len().saturating_add(64));
        write_var_u64(
            &mut payload,
            u64::try_from(self.raw().len())
                .map_err(|_| TokenizerError::LengthOverflow("NedoFormer sidecar raw length"))?,
        );
        payload.extend_from_slice(&Sha256::digest(self.raw()));
        write_var_u64(
            &mut payload,
            u64::try_from(self.units().len())
                .map_err(|_| TokenizerError::LengthOverflow("NedoFormer sidecar unit count"))?,
        );

        let mut expected_start = 0_u64;
        let mut previous_group = None;
        let mut next_group = 0_u32;
        for lattice in self.units() {
            let unit = &lattice.selected_unit;
            if unit.span.start != expected_start || unit.span.end <= unit.span.start {
                return Err(TokenizerError::InvalidTrainingEncoding(
                    "NedoFormer sidecar unit coverage is non-canonical",
                ));
            }
            expected_start = unit.span.end;
            write_var_u64(&mut payload, unit.span.len());

            let (group_state, new_previous) =
                encode_group_state(unit.group_id, previous_group, &mut next_group)?;
            previous_group = new_previous;
            let ambiguous = lattice.candidates.len() > 1;
            let flags = pack_unit_flags(unit.kind, unit.mode, group_state, ambiguous)?;
            payload.push(flags);

            if ambiguous {
                write_var_u64(
                    &mut payload,
                    u64::try_from(lattice.candidates.len()).map_err(|_| {
                        TokenizerError::LengthOverflow("NedoFormer sidecar candidate count")
                    })?,
                );
                let selected_index = lattice
                    .candidates
                    .iter()
                    .position(|candidate| candidate.selected)
                    .ok_or(TokenizerError::InvalidTrainingEncoding(
                        "NedoFormer sidecar selected candidate is absent",
                    ))?;
                if lattice
                    .candidates
                    .iter()
                    .filter(|candidate| candidate.selected)
                    .count()
                    != 1
                {
                    return Err(TokenizerError::InvalidTrainingEncoding(
                        "NedoFormer sidecar requires exactly one selected candidate",
                    ));
                }
                write_var_u64(
                    &mut payload,
                    u64::try_from(selected_index).map_err(|_| {
                        TokenizerError::LengthOverflow("NedoFormer sidecar selected index")
                    })?,
                );
                for candidate in &lattice.candidates {
                    if !candidate.conditional_log_score.is_finite() {
                        return Err(TokenizerError::InvalidTrainingEncoding(
                            "NedoFormer sidecar score is non-finite",
                        ));
                    }
                    payload.extend_from_slice(
                        &candidate.conditional_log_score.to_bits().to_le_bytes(),
                    );
                    write_relative_cuts(&mut payload, unit.span, &candidate.cuts)?;
                }
            } else {
                let candidate =
                    lattice
                        .candidates
                        .first()
                        .ok_or(TokenizerError::InvalidTrainingEncoding(
                            "NedoFormer sidecar unit has no candidate",
                        ))?;
                if !candidate.selected {
                    return Err(TokenizerError::InvalidTrainingEncoding(
                        "NedoFormer single sidecar candidate is not selected",
                    ));
                }
                write_relative_cuts(&mut payload, unit.span, &candidate.cuts)?;
            }
        }
        if expected_start
            != u64::try_from(self.raw().len())
                .map_err(|_| TokenizerError::LengthOverflow("NedoFormer sidecar raw length"))?
        {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "NedoFormer sidecar units do not cover raw document",
            ));
        }

        let mut output = Vec::with_capacity(HEADER_BYTES + payload.len());
        output.extend_from_slice(SIDECAR_MAGIC);
        output.extend_from_slice(&NEDOFORMER_SIDECAR_SCHEMA_VERSION.to_le_bytes());
        output.extend_from_slice(&sidecar_asset_identity());
        output.extend_from_slice(&Sha256::digest(&payload));
        output.extend_from_slice(&payload);
        Ok(output)
    }
}

pub fn encode_sidecar_units(
    raw: &[u8],
    units: &[NedoFormerSidecarUnit],
) -> Result<Vec<u8>, TokenizerError> {
    let mut payload = Vec::with_capacity(raw.len().saturating_add(64));
    write_var_u64(
        &mut payload,
        u64::try_from(raw.len())
            .map_err(|_| TokenizerError::LengthOverflow("NedoFormer sidecar raw length"))?,
    );
    payload.extend_from_slice(&Sha256::digest(raw));
    write_var_u64(
        &mut payload,
        u64::try_from(units.len())
            .map_err(|_| TokenizerError::LengthOverflow("NedoFormer sidecar unit count"))?,
    );

    let mut expected_start = 0_u64;
    let mut previous_group = None;
    let mut next_group = 0_u32;
    for unit in units {
        if unit.span.start != expected_start || unit.span.end <= unit.span.start {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "NedoFormer sidecar unit coverage is non-canonical",
            ));
        }
        expected_start = unit.span.end;
        write_var_u64(&mut payload, unit.span.len());
        let (group_state, new_previous) =
            encode_group_state(unit.group_id, previous_group, &mut next_group)?;
        previous_group = new_previous;
        write_sidecar_unit_candidates(&mut payload, unit, group_state)?;
    }
    let raw_len = u64::try_from(raw.len())
        .map_err(|_| TokenizerError::LengthOverflow("NedoFormer sidecar raw length"))?;
    if expected_start != raw_len {
        return Err(TokenizerError::InvalidTrainingEncoding(
            "NedoFormer sidecar units do not cover raw document",
        ));
    }
    let mut output = Vec::with_capacity(HEADER_BYTES + payload.len());
    output.extend_from_slice(SIDECAR_MAGIC);
    output.extend_from_slice(&NEDOFORMER_SIDECAR_SCHEMA_VERSION.to_le_bytes());
    output.extend_from_slice(&sidecar_asset_identity());
    output.extend_from_slice(&Sha256::digest(&payload));
    output.extend_from_slice(&payload);
    Ok(output)
}

fn write_sidecar_unit_candidates(
    payload: &mut Vec<u8>,
    unit: &NedoFormerSidecarUnit,
    group_state: u8,
) -> Result<(), TokenizerError> {
    let ambiguous = unit.candidates.len() > 1;
    payload.push(pack_unit_flags(
        unit.kind,
        unit.mode,
        group_state,
        ambiguous,
    )?);
    if !ambiguous {
        let candidate = unit
            .candidates
            .first()
            .ok_or(TokenizerError::InvalidTrainingEncoding(
                "NedoFormer sidecar unit has no candidate",
            ))?;
        if !candidate.selected {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "NedoFormer single sidecar candidate is not selected",
            ));
        }
        return write_relative_cuts(payload, unit.span, &candidate.cuts);
    }

    write_var_u64(
        payload,
        u64::try_from(unit.candidates.len())
            .map_err(|_| TokenizerError::LengthOverflow("NedoFormer sidecar candidate count"))?,
    );
    let selected_index = unit
        .candidates
        .iter()
        .position(|candidate| candidate.selected)
        .ok_or(TokenizerError::InvalidTrainingEncoding(
            "NedoFormer sidecar selected candidate is absent",
        ))?;
    if unit
        .candidates
        .iter()
        .filter(|candidate| candidate.selected)
        .count()
        != 1
    {
        return Err(TokenizerError::InvalidTrainingEncoding(
            "NedoFormer sidecar requires exactly one selected candidate",
        ));
    }
    write_var_u64(
        payload,
        u64::try_from(selected_index)
            .map_err(|_| TokenizerError::LengthOverflow("NedoFormer sidecar selected index"))?,
    );
    for candidate in &unit.candidates {
        if !candidate.conditional_log_score.is_finite() {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "NedoFormer sidecar score is non-finite",
            ));
        }
        payload.extend_from_slice(&candidate.conditional_log_score.to_bits().to_le_bytes());
        write_relative_cuts(payload, unit.span, &candidate.cuts)?;
    }
    Ok(())
}

impl NedoFormerLatticeSidecar {
    /// Loads a compact metadata sidecar against the exact external raw document.
    ///
    /// # Errors
    ///
    /// Returns an error for asset/checksum/raw-identity/span/group/candidate failures.
    #[allow(clippy::too_many_lines)] // Stable binary validation is intentionally explicit.
    pub fn from_bytes(raw: Vec<u8>, input: &[u8]) -> Result<Self, TokenizerError> {
        if input.len() < HEADER_BYTES || input.get(..8) != Some(SIDECAR_MAGIC) {
            return Err(TokenizerError::BadCodecMagic);
        }
        let mut header = Reader::new(&input[8..]);
        let schema = header.u32()?;
        if schema != NEDOFORMER_SIDECAR_SCHEMA_VERSION {
            return Err(TokenizerError::UnsupportedCodecVersion(schema));
        }
        let expected_asset: [u8; 32] = header
            .bytes(32)?
            .try_into()
            .map_err(|_| TokenizerError::TruncatedCodec)?;
        if expected_asset != sidecar_asset_identity() {
            return Err(TokenizerError::AssetIdentityMismatch);
        }
        let expected_payload: [u8; 32] = header
            .bytes(32)?
            .try_into()
            .map_err(|_| TokenizerError::TruncatedCodec)?;
        let payload = header.remaining_bytes();
        let actual_payload: [u8; 32] = Sha256::digest(payload).into();
        if actual_payload != expected_payload {
            return Err(TokenizerError::CodecChecksumMismatch);
        }

        let mut reader = Reader::new(payload);
        let expected_raw_len = reader.usize_var("NedoFormer sidecar raw length")?;
        if expected_raw_len != raw.len() {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "NedoFormer sidecar raw length differs",
            ));
        }
        let expected_raw_sha: [u8; 32] = reader
            .bytes(32)?
            .try_into()
            .map_err(|_| TokenizerError::TruncatedCodec)?;
        let actual_raw_sha: [u8; 32] = Sha256::digest(&raw).into();
        if expected_raw_sha != actual_raw_sha {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "NedoFormer sidecar raw SHA-256 differs",
            ));
        }

        let unit_count = reader.usize_var("NedoFormer sidecar unit count")?;
        if unit_count > raw.len().saturating_add(1) {
            return Err(TokenizerError::ImpossibleCodecCount(
                "NedoFormer sidecar unit count",
            ));
        }
        let mut units = Vec::with_capacity(unit_count);
        let raw_len_u64 = u64::try_from(raw.len())
            .map_err(|_| TokenizerError::LengthOverflow("NedoFormer sidecar raw length"))?;
        let mut expected_start = 0_u64;
        let mut previous_group = None;
        let mut next_group = 0_u32;
        for _ in 0..unit_count {
            let length = reader.var_u64("NedoFormer sidecar unit length")?;
            if length == 0 {
                return Err(TokenizerError::InvalidTrainingEncoding(
                    "NedoFormer sidecar unit length is zero",
                ));
            }
            let end = expected_start
                .checked_add(length)
                .ok_or(TokenizerError::LengthOverflow(
                    "NedoFormer sidecar unit end",
                ))?;
            if end > raw_len_u64 {
                return Err(TokenizerError::InvalidTrainingEncoding(
                    "NedoFormer sidecar unit escapes raw document",
                ));
            }
            let span = ByteSpan {
                start: expected_start,
                end,
            };
            expected_start = end;

            let flags = reader.u8()?;
            let kind = unpack_kind(flags)?;
            let mode = unpack_mode(flags)?;
            let group_state = (flags & GROUP_MASK) >> 5;
            let group_id = decode_group_state(group_state, &mut previous_group, &mut next_group)?;
            let ambiguous = flags & AMBIGUOUS_FLAG != 0;

            let candidates = if ambiguous {
                let candidate_count = reader.usize_var("NedoFormer sidecar candidate count")?;
                if !(2..=256).contains(&candidate_count) {
                    return Err(TokenizerError::ImpossibleCodecCount(
                        "NedoFormer sidecar candidate count",
                    ));
                }
                let selected_index = reader.usize_var("NedoFormer sidecar selected index")?;
                if selected_index >= candidate_count {
                    return Err(TokenizerError::InvalidTrainingEncoding(
                        "NedoFormer sidecar selected index is outside candidates",
                    ));
                }
                let mut candidates = Vec::with_capacity(candidate_count);
                for candidate_index in 0..candidate_count {
                    let score = f32::from_bits(reader.u32()?);
                    if !score.is_finite() {
                        return Err(TokenizerError::InvalidTrainingEncoding(
                            "NedoFormer sidecar score is non-finite",
                        ));
                    }
                    let cuts = read_relative_cuts(&mut reader, span)?;
                    candidates.push(NedoFormerSidecarCandidate {
                        cuts,
                        conditional_log_score: score,
                        selected: candidate_index == selected_index,
                    });
                }
                candidates
            } else {
                vec![NedoFormerSidecarCandidate {
                    cuts: read_relative_cuts(&mut reader, span)?,
                    conditional_log_score: 0.0,
                    selected: true,
                }]
            };
            units.push(NedoFormerSidecarUnit {
                span,
                kind,
                mode,
                group_id,
                candidates,
            });
        }
        if expected_start != raw_len_u64 {
            return Err(TokenizerError::IncompleteDocument {
                covered: expected_start,
                document_len: raw_len_u64,
            });
        }
        if reader.remaining() != 0 {
            return Err(TokenizerError::TrailingCodecBytes(reader.remaining()));
        }
        Ok(Self { raw, units })
    }

    /// External raw bytes paired with this validated sidecar.
    #[must_use]
    pub fn raw(&self) -> &[u8] {
        &self.raw
    }

    /// Compact lattice units.
    #[must_use]
    pub fn units(&self) -> &[NedoFormerSidecarUnit] {
        &self.units
    }

    /// Samples only byte-offset cuts and returns a lossless document view.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid sampling temperature or metadata invariants.
    pub fn sample_lossless(
        &self,
        policy: NedoFormerSamplingPolicy,
        seed: u64,
    ) -> Result<nedo_format::LosslessDocument, TokenizerError> {
        if let NedoFormerSamplingPolicy::ContextWeighted { temperature } = policy {
            if !temperature.is_finite() || temperature <= 0.0 {
                return Err(TokenizerError::InvalidConfiguration(
                    "NedoFormer sampling temperature must be positive and finite",
                ));
            }
        }
        let mut rng = SplitMix64::new(seed);
        let mut surface_units = Vec::with_capacity(self.units.len());
        for unit in &self.units {
            let index = choose_candidate(&unit.candidates, policy, &mut rng)?;
            let candidate =
                unit.candidates
                    .get(index)
                    .ok_or(TokenizerError::InvalidTrainingEncoding(
                        "NedoFormer sidecar sampled candidate is outside unit",
                    ))?;
            surface_units.push(nedo_format::SurfaceUnit::new(
                unit.span,
                candidate.cuts.clone(),
            )?);
        }
        Ok(nedo_format::LosslessDocument::new(
            self.raw.clone(),
            surface_units,
        )?)
    }
}

fn choose_candidate(
    candidates: &[NedoFormerSidecarCandidate],
    policy: NedoFormerSamplingPolicy,
    rng: &mut SplitMix64,
) -> Result<usize, TokenizerError> {
    if candidates.is_empty() {
        return Err(TokenizerError::InvalidTrainingEncoding(
            "NedoFormer sidecar unit has no candidates",
        ));
    }
    if candidates.len() == 1 {
        return Ok(0);
    }
    match policy {
        NedoFormerSamplingPolicy::Best => candidates
            .iter()
            .position(|candidate| candidate.selected)
            .ok_or(TokenizerError::InvalidTrainingEncoding(
                "NedoFormer sidecar selected candidate is absent",
            )),
        NedoFormerSamplingPolicy::Uniform => {
            let length = u64::try_from(candidates.len())
                .map_err(|_| TokenizerError::LengthOverflow("NedoFormer sidecar candidates"))?;
            usize::try_from(rng.next_u64() % length)
                .map_err(|_| TokenizerError::LengthOverflow("NedoFormer sidecar sample index"))
        }
        NedoFormerSamplingPolicy::ContextWeighted { temperature } => {
            let maximum = candidates
                .iter()
                .map(|candidate| candidate.conditional_log_score / temperature)
                .fold(f32::NEG_INFINITY, f32::max);
            let weights = candidates
                .iter()
                .map(|candidate| ((candidate.conditional_log_score / temperature) - maximum).exp())
                .collect::<Vec<_>>();
            let total = weights.iter().copied().sum::<f32>();
            if !total.is_finite() || total <= 0.0 {
                return Err(TokenizerError::InvalidTrainingEncoding(
                    "NedoFormer sidecar sampling weights are invalid",
                ));
            }
            let target = rng.next_unit_f32() * total;
            let mut cumulative = 0.0_f32;
            for (index, weight) in weights.iter().copied().enumerate() {
                cumulative += weight;
                if target < cumulative {
                    return Ok(index);
                }
            }
            Ok(candidates.len() - 1)
        }
    }
}

fn sidecar_asset_identity() -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"NEDOFORMER-SIDECAR-ASSETS\0");
    hash.update(MORPHOLOGY_SHA256.as_bytes());
    hash.update(MODEL_SHA256.as_bytes());
    hash.finalize().into()
}

fn encode_group_state(
    group_id: Option<u32>,
    previous_group: Option<u32>,
    next_group: &mut u32,
) -> Result<(u8, Option<u32>), TokenizerError> {
    match group_id {
        None => Ok((GROUP_NONE, None)),
        Some(group) if previous_group == Some(group) => Ok((GROUP_SAME, Some(group))),
        Some(group) => {
            if group != *next_group {
                return Err(TokenizerError::InvalidTrainingEncoding(
                    "NedoFormer sidecar group IDs are non-canonical",
                ));
            }
            *next_group = next_group
                .checked_add(1)
                .ok_or(TokenizerError::LengthOverflow(
                    "NedoFormer sidecar group ID",
                ))?;
            Ok((GROUP_NEW, Some(group)))
        }
    }
}

fn decode_group_state(
    state: u8,
    previous_group: &mut Option<u32>,
    next_group: &mut u32,
) -> Result<Option<u32>, TokenizerError> {
    let group = match state {
        GROUP_NONE => None,
        GROUP_NEW => {
            let value = *next_group;
            *next_group = next_group
                .checked_add(1)
                .ok_or(TokenizerError::LengthOverflow(
                    "NedoFormer sidecar group ID",
                ))?;
            Some(value)
        }
        GROUP_SAME => Some(
            previous_group.ok_or(TokenizerError::InvalidTrainingEncoding(
                "NedoFormer sidecar repeated group has no predecessor",
            ))?,
        ),
        _ => {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "NedoFormer sidecar group state is reserved",
            ));
        }
    };
    *previous_group = group;
    Ok(group)
}

fn pack_unit_flags(
    kind: LexicalKind,
    mode: TokenMode,
    group_state: u8,
    ambiguous: bool,
) -> Result<u8, TokenizerError> {
    let kind_bits = (kind as u8)
        .checked_sub(1)
        .ok_or(TokenizerError::InvalidTrainingEncoding(
            "NedoFormer sidecar lexical kind cannot be packed",
        ))?;
    let mode_bits = (mode as u8)
        .checked_sub(1)
        .ok_or(TokenizerError::InvalidTrainingEncoding(
            "NedoFormer sidecar token mode cannot be packed",
        ))?;
    if kind_bits > KIND_MASK || mode_bits > 2 || group_state > GROUP_SAME {
        return Err(TokenizerError::InvalidTrainingEncoding(
            "NedoFormer sidecar packed metadata is outside schema",
        ));
    }
    Ok(kind_bits
        | (mode_bits << 3)
        | (group_state << 5)
        | if ambiguous { AMBIGUOUS_FLAG } else { 0 })
}

const fn unpack_kind(flags: u8) -> Result<LexicalKind, TokenizerError> {
    lexical_kind((flags & KIND_MASK) + 1)
}

fn unpack_mode(flags: u8) -> Result<TokenMode, TokenizerError> {
    TokenMode::try_from(((flags & MODE_MASK) >> 3) + 1)
}

fn write_relative_cuts(
    output: &mut Vec<u8>,
    span: ByteSpan,
    cuts: &[u64],
) -> Result<(), TokenizerError> {
    nedo_format::SurfaceUnit::new(span, cuts.to_vec())?;
    write_var_u64(
        output,
        u64::try_from(cuts.len())
            .map_err(|_| TokenizerError::LengthOverflow("NedoFormer sidecar cut count"))?,
    );
    for &cut in cuts {
        let relative =
            cut.checked_sub(span.start)
                .ok_or(TokenizerError::InvalidTrainingEncoding(
                    "NedoFormer sidecar cut precedes unit",
                ))?;
        write_var_u64(output, relative);
    }
    Ok(())
}

fn read_relative_cuts(reader: &mut Reader<'_>, span: ByteSpan) -> Result<Vec<u64>, TokenizerError> {
    let cut_count = reader.usize_var("NedoFormer sidecar cut count")?;
    let unit_len = usize::try_from(span.len())
        .map_err(|_| TokenizerError::LengthOverflow("NedoFormer sidecar unit length"))?;
    if cut_count > unit_len.saturating_sub(1) {
        return Err(TokenizerError::ImpossibleCodecCount(
            "NedoFormer sidecar cut count",
        ));
    }
    let mut cuts = Vec::with_capacity(cut_count);
    for _ in 0..cut_count {
        let relative = reader.var_u64("NedoFormer sidecar relative cut")?;
        let cut = span
            .start
            .checked_add(relative)
            .ok_or(TokenizerError::LengthOverflow("NedoFormer sidecar cut"))?;
        cuts.push(cut);
    }
    nedo_format::SurfaceUnit::new(span, cuts.clone())?;
    Ok(cuts)
}

fn write_var_u64(output: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = u8::try_from(value & 0x7f).unwrap_or_default();
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        output.push(byte);
        if value == 0 {
            break;
        }
    }
}

const fn varint_len(mut value: u64) -> usize {
    let mut length = 1_usize;
    while value >= 0x80 {
        value >>= 7;
        length += 1;
    }
    length
}

const fn lexical_kind(value: u8) -> Result<LexicalKind, TokenizerError> {
    match value {
        1 => Ok(LexicalKind::Whitespace),
        2 => Ok(LexicalKind::LineBreak),
        3 => Ok(LexicalKind::Word),
        4 => Ok(LexicalKind::Number),
        5 => Ok(LexicalKind::Punctuation),
        6 => Ok(LexicalKind::Symbol),
        7 => Ok(LexicalKind::Control),
        8 => Ok(LexicalKind::Opaque),
        _ => Err(TokenizerError::InvalidCodecEnum(
            "NedoFormer sidecar lexical kind",
            value,
        )),
    }
}

struct Reader<'a> {
    input: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    const fn new(input: &'a [u8]) -> Self {
        Self { input, position: 0 }
    }

    const fn remaining(&self) -> usize {
        self.input.len().saturating_sub(self.position)
    }

    fn remaining_bytes(&mut self) -> &'a [u8] {
        let output = &self.input[self.position..];
        self.position = self.input.len();
        output
    }

    fn bytes(&mut self, count: usize) -> Result<&'a [u8], TokenizerError> {
        let end = self
            .position
            .checked_add(count)
            .ok_or(TokenizerError::LengthOverflow("NedoFormer sidecar reader"))?;
        let output = self
            .input
            .get(self.position..end)
            .ok_or(TokenizerError::TruncatedCodec)?;
        self.position = end;
        Ok(output)
    }

    fn u8(&mut self) -> Result<u8, TokenizerError> {
        Ok(*self
            .bytes(1)?
            .first()
            .ok_or(TokenizerError::TruncatedCodec)?)
    }

    fn u32(&mut self) -> Result<u32, TokenizerError> {
        Ok(u32::from_le_bytes(
            self.bytes(4)?
                .try_into()
                .map_err(|_| TokenizerError::TruncatedCodec)?,
        ))
    }

    fn var_u64(&mut self, field: &'static str) -> Result<u64, TokenizerError> {
        let start = self.position;
        let mut value = 0_u64;
        let mut shift = 0_u32;
        loop {
            if shift >= 70 {
                return Err(TokenizerError::LengthOverflow(field));
            }
            let byte = self.u8()?;
            if shift == 63 && byte > 1 {
                return Err(TokenizerError::LengthOverflow(field));
            }
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        let consumed = self.position.saturating_sub(start);
        if consumed != varint_len(value) {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "NedoFormer sidecar varint is non-canonical",
            ));
        }
        Ok(value)
    }

    fn usize_var(&mut self, field: &'static str) -> Result<usize, TokenizerError> {
        usize::try_from(self.var_u64(field)?).map_err(|_| TokenizerError::LengthOverflow(field))
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    const fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn next_unit_f32(&mut self) -> f32 {
        let raw = self.next_u64() >> 41;
        let mantissa = u32::try_from(raw).unwrap_or_default();
        f32::from_bits(0x3f80_0000 | mantissa) - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::{NedoFormerLatticeSidecar, NEDOFORMER_SIDECAR_SCHEMA_VERSION};
    use crate::{NedoFormerSamplingPolicy, Tokenizer, TokenizerConfig};

    #[test]
    fn sidecar_omits_raw_text_and_preserves_sampled_cuts() -> Result<(), crate::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let raw = b"koyun cocuklarimizdan UNIQUE_SIDE_CAR_8731".to_vec();
        let lattice = tokenizer.nedoformer_lattice(raw.clone())?;
        let bytes = lattice.to_sidecar_bytes()?;
        assert_eq!(&bytes[..8], b"NDFSID02");
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            NEDOFORMER_SIDECAR_SCHEMA_VERSION
        );
        assert!(!bytes.windows(raw.len()).any(|window| window == raw));
        assert!(!bytes
            .windows(b"UNIQUE_SIDE_CAR_8731".len())
            .any(|window| window == b"UNIQUE_SIDE_CAR_8731"));
        let sidecar = NedoFormerLatticeSidecar::from_bytes(raw.clone(), &bytes)?;
        for policy in [
            NedoFormerSamplingPolicy::Best,
            NedoFormerSamplingPolicy::Uniform,
            NedoFormerSamplingPolicy::ContextWeighted { temperature: 0.9 },
        ] {
            for seed in [0_u64, 1, 91, 2026] {
                let rich = lattice.sample(policy, seed)?.lossless_document()?;
                let sampled = sidecar.sample_lossless(policy, seed)?;
                assert_eq!(sampled, rich);
                assert_eq!(sampled.decode(), raw.as_slice());
            }
        }

        let mut wrong_raw = raw;
        wrong_raw[0] ^= 1;
        assert!(NedoFormerLatticeSidecar::from_bytes(wrong_raw, &bytes).is_err());
        Ok(())
    }

    #[test]
    fn compact_sidecar_stays_bounded_on_representative_text() -> Result<(), crate::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let sentence =
            "Koyun çocuklarımızdan geliyor mu? Ankara'da 23.07.2026 saat 14:30'da buluşalım. ";
        let raw = sentence.repeat(64).into_bytes();
        let lattice = tokenizer.nedoformer_lattice(raw.clone())?;
        let sidecar = lattice.to_sidecar_bytes()?;
        assert!(sidecar.len() < raw.len().saturating_mul(2));
        let decoded = NedoFormerLatticeSidecar::from_bytes(raw.clone(), &sidecar)?
            .sample_lossless(NedoFormerSamplingPolicy::Best, 0)?;
        assert_eq!(decoded.decode(), raw.as_slice());
        Ok(())
    }
}
