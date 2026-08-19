//! Exact precompiled surface-to-candidate analysis table.

use core::fmt;
use std::{ops::Range, sync::Arc};

use nedo_morph_bundle::AmbiguityWordData;
use sha2::{Digest, Sha256};

use crate::{
    AmbiguityScoringInterner, FlatAnalysisSet, TokenizerError, MODEL_SHA256, MORPHOLOGY_SHA256,
};

const MAGIC: &[u8; 8] = b"NEDOFUL1";
const VERSION: u32 = 1;
const IDENTITY_BYTES: usize = 64;
const DIGEST_BYTES: usize = 32;
const HEADER_BYTES: usize = 8 + 4 + IDENTITY_BYTES + IDENTITY_BYTES + 4 + DIGEST_BYTES;
const EMPTY_SLOT: usize = usize::MAX;

/// One exact precompiled candidate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledSurfaceCandidateEntry {
    /// Relative tokenizer byte cuts.
    pub cuts: Vec<u32>,
    /// Explicit unknown-analysis status.
    pub unknown: bool,
    /// Exact canonical analysis identity.
    pub canonical: String,
    /// Exact perceptron lemma.
    pub lemma: String,
    /// Exact ordered inflectional groups.
    pub igs: Vec<String>,
    /// Java-compatible analysis hash.
    pub java_hash: i32,
}

/// One exact surface and all surface-valid candidates in production order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledSurfaceAnalysisEntry {
    /// Exact UTF-8 surface bytes.
    pub surface: Vec<u8>,
    /// Surface-valid candidates in production order.
    pub candidates: Vec<CompiledSurfaceCandidateEntry>,
}

#[derive(Clone, Debug)]
struct RuntimeEntry {
    surface: Range<usize>,
    set_index: usize,
    hash: u64,
}

/// Parsed exact surface analysis table with preinterned scoring identities.
#[derive(Debug)]
pub struct CompiledSurfaceAnalysisTable {
    digest: [u8; DIGEST_BYTES],
    surface_bytes: Vec<u8>,
    entries: Vec<RuntimeEntry>,
    sets: Vec<FlatAnalysisSet>,
    slots: Vec<usize>,
    scoring_interner: Arc<AmbiguityScoringInterner>,
}

impl CompiledSurfaceAnalysisTable {
    /// Parses, checksum-validates, and preinterns one full candidate table.
    ///
    /// Exact equivalence with the active morphology is separately revalidated by
    /// [`crate::Tokenizer::with_verified_compiled_surface_analysis_table`].
    ///
    /// # Errors
    ///
    /// Returns an error for corruption, identity mismatch, malformed candidates,
    /// duplicate surfaces, invalid UTF-8, or representation overflow.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CompiledSurfaceAnalysisTableError> {
        if bytes.len() < HEADER_BYTES {
            return Err(CompiledSurfaceAnalysisTableError::Truncated);
        }
        let mut cursor = 0_usize;
        if take(bytes, &mut cursor, MAGIC.len())? != MAGIC {
            return Err(CompiledSurfaceAnalysisTableError::BadMagic);
        }
        let version = read_u32(bytes, &mut cursor)?;
        if version != VERSION {
            return Err(CompiledSurfaceAnalysisTableError::UnsupportedVersion(
                version,
            ));
        }
        let morphology = take(bytes, &mut cursor, IDENTITY_BYTES)?;
        let model = take(bytes, &mut cursor, IDENTITY_BYTES)?;
        if morphology != MORPHOLOGY_SHA256.as_bytes() || model != MODEL_SHA256.as_bytes() {
            return Err(CompiledSurfaceAnalysisTableError::AssetIdentityMismatch);
        }
        let entry_count = usize::try_from(read_u32(bytes, &mut cursor)?)
            .map_err(|_| CompiledSurfaceAnalysisTableError::LengthOverflow)?;
        let expected_digest = take(bytes, &mut cursor, DIGEST_BYTES)?;
        let payload = bytes
            .get(cursor..)
            .ok_or(CompiledSurfaceAnalysisTableError::Truncated)?;
        let actual_digest: [u8; DIGEST_BYTES] = Sha256::digest(payload).into();
        if expected_digest != actual_digest {
            return Err(CompiledSurfaceAnalysisTableError::ChecksumMismatch);
        }

        let mut payload_cursor = 0_usize;
        let mut surface_bytes = Vec::new();
        let mut entries = Vec::with_capacity(entry_count);
        let mut sets = Vec::with_capacity(entry_count);
        let mut scoring_interner = AmbiguityScoringInterner::default();
        for entry_index in 0..entry_count {
            let surface_len = usize::try_from(read_u32(payload, &mut payload_cursor)?)
                .map_err(|_| CompiledSurfaceAnalysisTableError::LengthOverflow)?;
            let candidate_count = usize::try_from(read_u32(payload, &mut payload_cursor)?)
                .map_err(|_| CompiledSurfaceAnalysisTableError::LengthOverflow)?;
            if candidate_count == 0 {
                return Err(CompiledSurfaceAnalysisTableError::EmptyCandidateSet);
            }
            let surface = take(payload, &mut payload_cursor, surface_len)?;
            validate_surface(surface)?;
            let surface_start = surface_bytes.len();
            surface_bytes.extend_from_slice(surface);
            let surface_end = surface_bytes.len();

            let mut relative_cuts = Vec::with_capacity(candidate_count);
            let mut ambiguity = Vec::with_capacity(candidate_count);
            let mut unknown = Vec::with_capacity(candidate_count);
            for _ in 0..candidate_count {
                let cut_count = usize::try_from(read_u32(payload, &mut payload_cursor)?)
                    .map_err(|_| CompiledSurfaceAnalysisTableError::LengthOverflow)?;
                let candidate_unknown = match take(payload, &mut payload_cursor, 1)?[0] {
                    0 => false,
                    1 => true,
                    value => {
                        return Err(CompiledSurfaceAnalysisTableError::InvalidBoolean(value));
                    }
                };
                if take(payload, &mut payload_cursor, 3)? != [0, 0, 0] {
                    return Err(CompiledSurfaceAnalysisTableError::NonzeroReservedBytes);
                }
                let java_hash = read_i32(payload, &mut payload_cursor)?;
                let canonical_len = usize::try_from(read_u32(payload, &mut payload_cursor)?)
                    .map_err(|_| CompiledSurfaceAnalysisTableError::LengthOverflow)?;
                let lemma_len = usize::try_from(read_u32(payload, &mut payload_cursor)?)
                    .map_err(|_| CompiledSurfaceAnalysisTableError::LengthOverflow)?;
                let ig_count = usize::try_from(read_u32(payload, &mut payload_cursor)?)
                    .map_err(|_| CompiledSurfaceAnalysisTableError::LengthOverflow)?;
                let mut cuts = Vec::with_capacity(cut_count);
                for _ in 0..cut_count {
                    cuts.push(read_u32(payload, &mut payload_cursor)?);
                }
                validate_cuts(surface, &cuts)?;
                let canonical = read_string(payload, &mut payload_cursor, canonical_len)?;
                let lemma = read_string(payload, &mut payload_cursor, lemma_len)?;
                let mut igs = Vec::with_capacity(ig_count);
                for _ in 0..ig_count {
                    let length = usize::try_from(read_u32(payload, &mut payload_cursor)?)
                        .map_err(|_| CompiledSurfaceAnalysisTableError::LengthOverflow)?;
                    igs.push(read_string(payload, &mut payload_cursor, length)?);
                }
                relative_cuts.push(cuts);
                ambiguity.push(AmbiguityWordData {
                    canonical,
                    lemma,
                    igs,
                    java_hash,
                });
                unknown.push(candidate_unknown);
            }
            let scoring_codes = ambiguity
                .iter()
                .map(|word| scoring_interner.code(word))
                .collect::<Result<Vec<_>, TokenizerError>>()
                .map_err(|_| CompiledSurfaceAnalysisTableError::ScoringIdentityOverflow)?;
            let output_invariant =
                crate::flat_candidates_output_invariant(&relative_cuts, &unknown);
            sets.push(FlatAnalysisSet {
                relative_cuts: relative_cuts.into_boxed_slice(),
                ambiguity: ambiguity.into_boxed_slice(),
                scoring_codes: scoring_codes.into_boxed_slice(),
                unknown: unknown.into_boxed_slice(),
                output_invariant,
            });
            entries.push(RuntimeEntry {
                surface: surface_start..surface_end,
                set_index: entry_index,
                hash: surface_hash(surface),
            });
        }
        if payload_cursor != payload.len() {
            return Err(CompiledSurfaceAnalysisTableError::TrailingBytes(
                payload.len().saturating_sub(payload_cursor),
            ));
        }
        let slots_len = entries
            .len()
            .saturating_mul(2)
            .max(8)
            .checked_next_power_of_two()
            .ok_or(CompiledSurfaceAnalysisTableError::LengthOverflow)?;
        let mut slots = vec![EMPTY_SLOT; slots_len];
        let modulo = slots_len - 1;
        for (index, entry) in entries.iter().enumerate() {
            let surface = surface_bytes
                .get(entry.surface.clone())
                .ok_or(CompiledSurfaceAnalysisTableError::LengthOverflow)?;
            let mut slot = hash_slot(entry.hash, modulo);
            loop {
                let stored = slots[slot];
                if stored == EMPTY_SLOT {
                    slots[slot] = index;
                    break;
                }
                let existing = entries
                    .get(stored)
                    .ok_or(CompiledSurfaceAnalysisTableError::LengthOverflow)?;
                if surface_bytes.get(existing.surface.clone()) == Some(surface) {
                    return Err(CompiledSurfaceAnalysisTableError::DuplicateSurface);
                }
                slot = (slot + 1) & modulo;
            }
        }
        Ok(Self {
            digest: actual_digest,
            surface_bytes,
            entries,
            sets,
            slots,
            scoring_interner: Arc::new(scoring_interner),
        })
    }

    /// Number of exact surfaces in the table.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no surfaces are present.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn get(&self, surface: &str) -> Option<&FlatAnalysisSet> {
        if self.slots.is_empty() {
            return None;
        }
        let bytes = surface.as_bytes();
        let hash = surface_hash(bytes);
        let modulo = self.slots.len() - 1;
        let mut slot = hash_slot(hash, modulo);
        loop {
            let index = self.slots[slot];
            if index == EMPTY_SLOT {
                return None;
            }
            let entry = self.entries.get(index)?;
            if entry.hash == hash && self.surface_bytes.get(entry.surface.clone()) == Some(bytes) {
                return self.sets.get(entry.set_index);
            }
            slot = (slot + 1) & modulo;
        }
    }

    pub(crate) fn scoring_interner(&self) -> &Arc<AmbiguityScoringInterner> {
        &self.scoring_interner
    }

    pub(crate) const fn digest(&self) -> [u8; DIGEST_BYTES] {
        self.digest
    }

    pub(crate) fn entries(&self) -> impl Iterator<Item = (&[u8], &FlatAnalysisSet)> {
        self.entries.iter().filter_map(|entry| {
            Some((
                self.surface_bytes.get(entry.surface.clone())?,
                self.sets.get(entry.set_index)?,
            ))
        })
    }
}

/// Encodes a deterministic full surface-analysis table.
///
/// # Errors
///
/// Returns an error for duplicate surfaces, empty candidates, invalid UTF-8,
/// malformed cuts, or representation overflow.
pub fn encode_compiled_surface_analysis_table(
    entries: &[CompiledSurfaceAnalysisEntry],
) -> Result<Vec<u8>, CompiledSurfaceAnalysisTableError> {
    let entry_count = u32::try_from(entries.len())
        .map_err(|_| CompiledSurfaceAnalysisTableError::LengthOverflow)?;
    let mut ordered = entries.to_vec();
    ordered.sort_unstable_by(|left, right| left.surface.cmp(&right.surface));
    let mut payload = Vec::new();
    let mut previous_surface: Option<&[u8]> = None;
    for entry in &ordered {
        validate_surface(&entry.surface)?;
        if previous_surface == Some(entry.surface.as_slice()) {
            return Err(CompiledSurfaceAnalysisTableError::DuplicateSurface);
        }
        if entry.candidates.is_empty() {
            return Err(CompiledSurfaceAnalysisTableError::EmptyCandidateSet);
        }
        payload.extend_from_slice(
            &u32::try_from(entry.surface.len())
                .map_err(|_| CompiledSurfaceAnalysisTableError::LengthOverflow)?
                .to_le_bytes(),
        );
        payload.extend_from_slice(
            &u32::try_from(entry.candidates.len())
                .map_err(|_| CompiledSurfaceAnalysisTableError::LengthOverflow)?
                .to_le_bytes(),
        );
        payload.extend_from_slice(&entry.surface);
        for candidate in &entry.candidates {
            validate_cuts(&entry.surface, &candidate.cuts)?;
            payload.extend_from_slice(
                &u32::try_from(candidate.cuts.len())
                    .map_err(|_| CompiledSurfaceAnalysisTableError::LengthOverflow)?
                    .to_le_bytes(),
            );
            payload.push(u8::from(candidate.unknown));
            payload.extend_from_slice(&[0, 0, 0]);
            payload.extend_from_slice(&candidate.java_hash.to_le_bytes());
            payload.extend_from_slice(
                &u32::try_from(candidate.canonical.len())
                    .map_err(|_| CompiledSurfaceAnalysisTableError::LengthOverflow)?
                    .to_le_bytes(),
            );
            payload.extend_from_slice(
                &u32::try_from(candidate.lemma.len())
                    .map_err(|_| CompiledSurfaceAnalysisTableError::LengthOverflow)?
                    .to_le_bytes(),
            );
            payload.extend_from_slice(
                &u32::try_from(candidate.igs.len())
                    .map_err(|_| CompiledSurfaceAnalysisTableError::LengthOverflow)?
                    .to_le_bytes(),
            );
            for cut in &candidate.cuts {
                payload.extend_from_slice(&cut.to_le_bytes());
            }
            payload.extend_from_slice(candidate.canonical.as_bytes());
            payload.extend_from_slice(candidate.lemma.as_bytes());
            for ig in &candidate.igs {
                payload.extend_from_slice(
                    &u32::try_from(ig.len())
                        .map_err(|_| CompiledSurfaceAnalysisTableError::LengthOverflow)?
                        .to_le_bytes(),
                );
                payload.extend_from_slice(ig.as_bytes());
            }
        }
        previous_surface = Some(&entry.surface);
    }
    let digest: [u8; DIGEST_BYTES] = Sha256::digest(&payload).into();
    let mut output = Vec::with_capacity(HEADER_BYTES.saturating_add(payload.len()));
    output.extend_from_slice(MAGIC);
    output.extend_from_slice(&VERSION.to_le_bytes());
    output.extend_from_slice(MORPHOLOGY_SHA256.as_bytes());
    output.extend_from_slice(MODEL_SHA256.as_bytes());
    output.extend_from_slice(&entry_count.to_le_bytes());
    output.extend_from_slice(&digest);
    output.extend_from_slice(&payload);
    Ok(output)
}

fn validate_surface(surface: &[u8]) -> Result<(), CompiledSurfaceAnalysisTableError> {
    if surface.is_empty() {
        return Err(CompiledSurfaceAnalysisTableError::EmptySurface);
    }
    std::str::from_utf8(surface).map_err(|_| CompiledSurfaceAnalysisTableError::InvalidUtf8)?;
    Ok(())
}

fn validate_cuts(surface: &[u8], cuts: &[u32]) -> Result<(), CompiledSurfaceAnalysisTableError> {
    let mut previous = 0_u32;
    for (index, cut) in cuts.iter().copied().enumerate() {
        if cut == 0
            || usize::try_from(cut).map_or(true, |value| value >= surface.len())
            || (index > 0 && cut <= previous)
        {
            return Err(CompiledSurfaceAnalysisTableError::InvalidCuts);
        }
        previous = cut;
    }
    Ok(())
}

fn read_string(
    bytes: &[u8],
    cursor: &mut usize,
    length: usize,
) -> Result<String, CompiledSurfaceAnalysisTableError> {
    let raw = take(bytes, cursor, length)?;
    let value =
        std::str::from_utf8(raw).map_err(|_| CompiledSurfaceAnalysisTableError::InvalidUtf8)?;
    Ok(value.to_owned())
}

fn take<'a>(
    bytes: &'a [u8],
    cursor: &mut usize,
    length: usize,
) -> Result<&'a [u8], CompiledSurfaceAnalysisTableError> {
    let end = cursor
        .checked_add(length)
        .ok_or(CompiledSurfaceAnalysisTableError::LengthOverflow)?;
    let value = bytes
        .get(*cursor..end)
        .ok_or(CompiledSurfaceAnalysisTableError::Truncated)?;
    *cursor = end;
    Ok(value)
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, CompiledSurfaceAnalysisTableError> {
    let raw: [u8; 4] = take(bytes, cursor, 4)?
        .try_into()
        .map_err(|_| CompiledSurfaceAnalysisTableError::Truncated)?;
    Ok(u32::from_le_bytes(raw))
}

fn read_i32(bytes: &[u8], cursor: &mut usize) -> Result<i32, CompiledSurfaceAnalysisTableError> {
    let raw: [u8; 4] = take(bytes, cursor, 4)?
        .try_into()
        .map_err(|_| CompiledSurfaceAnalysisTableError::Truncated)?;
    Ok(i32::from_le_bytes(raw))
}

#[inline(always)]
fn surface_hash(bytes: &[u8]) -> u64 {
    let mut value = 0xcbf2_9ce4_8422_2325_u64 ^ (bytes.len() as u64).rotate_left(17);
    for byte in bytes {
        value ^= u64::from(*byte);
        value = value.wrapping_mul(0x0000_0100_0000_01b3);
    }
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[inline(always)]
fn hash_slot(hash: u64, modulo: usize) -> usize {
    usize::try_from(hash).map_or(0, |value| value & modulo)
}

/// Full compiled surface-table parse/build failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompiledSurfaceAnalysisTableError {
    /// Input ended before declared data was available.
    Truncated,
    /// Magic header differs.
    BadMagic,
    /// Schema version is unsupported.
    UnsupportedVersion(u32),
    /// Morphology/model identity differs.
    AssetIdentityMismatch,
    /// Payload checksum differs.
    ChecksumMismatch,
    /// Boolean byte is non-canonical.
    InvalidBoolean(u8),
    /// Reserved bytes are nonzero.
    NonzeroReservedBytes,
    /// Surface is empty.
    EmptySurface,
    /// Candidate set is empty.
    EmptyCandidateSet,
    /// Surface or feature string is invalid UTF-8.
    InvalidUtf8,
    /// Cuts are malformed.
    InvalidCuts,
    /// Surface key occurs more than once.
    DuplicateSurface,
    /// Length cannot be represented.
    LengthOverflow,
    /// Preinterned scoring identity cannot be represented.
    ScoringIdentityOverflow,
    /// Undeclared trailing bytes remain.
    TrailingBytes(usize),
}

impl fmt::Display for CompiledSurfaceAnalysisTableError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for CompiledSurfaceAnalysisTableError {}

#[cfg(test)]
mod tests {
    use super::{
        encode_compiled_surface_analysis_table, CompiledSurfaceAnalysisEntry,
        CompiledSurfaceAnalysisTable, CompiledSurfaceAnalysisTableError,
        CompiledSurfaceCandidateEntry,
    };

    #[test]
    fn round_trip_preserves_candidate_order_and_features() {
        let bytes = encode_compiled_surface_analysis_table(&[CompiledSurfaceAnalysisEntry {
            surface: b"koyun".to_vec(),
            candidates: vec![
                CompiledSurfaceCandidateEntry {
                    cuts: Vec::new(),
                    unknown: false,
                    canonical: "koyun_Noun".to_owned(),
                    lemma: "koyun".to_owned(),
                    igs: vec!["Noun+A3sg".to_owned()],
                    java_hash: 7,
                },
                CompiledSurfaceCandidateEntry {
                    cuts: vec![3],
                    unknown: false,
                    canonical: "koy_Verb+un".to_owned(),
                    lemma: "koymak".to_owned(),
                    igs: vec!["Verb".to_owned(), "Imp+A2pl".to_owned()],
                    java_hash: 9,
                },
            ],
        }])
        .expect("test full table must encode");
        let table =
            CompiledSurfaceAnalysisTable::from_bytes(&bytes).expect("test full table must parse");
        let set = table.get("koyun").expect("surface must exist");
        assert_eq!(set.relative_cuts.as_ref(), [Vec::new(), vec![3]]);
        assert_eq!(set.ambiguity[0].canonical, "koyun_Noun");
        assert_eq!(set.ambiguity[1].java_hash, 9);
        assert_eq!(set.scoring_codes.len(), 2);
    }

    #[test]
    fn checksum_rejects_payload_corruption() {
        let mut bytes = encode_compiled_surface_analysis_table(&[CompiledSurfaceAnalysisEntry {
            surface: b"ev".to_vec(),
            candidates: vec![CompiledSurfaceCandidateEntry {
                cuts: Vec::new(),
                unknown: false,
                canonical: "ev_Noun".to_owned(),
                lemma: "ev".to_owned(),
                igs: vec!["Noun+A3sg".to_owned()],
                java_hash: 1,
            }],
        }])
        .expect("test full table must encode");
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        assert_eq!(
            CompiledSurfaceAnalysisTable::from_bytes(&bytes)
                .expect_err("payload corruption must fail"),
            CompiledSurfaceAnalysisTableError::ChecksumMismatch
        );
    }
}
