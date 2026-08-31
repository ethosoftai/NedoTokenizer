//! Surface-piece vocabulary for morphology-aware, boundary-free LM streams.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Range;

use nedo_format::ByteSpan;
use sha2::{Digest, Sha256};

use crate::{
    flat_surface::{FlatSurfaceUnit, SurfaceProgramUse},
    LexicalKind, TokenizedDocument, TokenizedUnit, TokenizerError, TrainingEncoding,
};

/// Padding ID for surface-piece models.
pub const SURFACE_PAD_ID: u32 = 0;
/// Beginning-of-document ID for surface-piece models.
pub const SURFACE_BOS_ID: u32 = 1;
/// End-of-document ID for surface-piece models.
pub const SURFACE_EOS_ID: u32 = 2;
/// First exact byte fallback ID.
pub const SURFACE_BYTE_BASE_ID: u32 = 3;
/// First learned surface-piece ID.
pub const SURFACE_ENTRY_BASE_ID: u32 = SURFACE_BYTE_BASE_ID + 256;

const SURFACE_GREEDY_MAGIC: &[u8; 8] = b"NDSRF001";
const SURFACE_BPE_MAGIC: &[u8; 8] = b"NDSRF002";
const SURFACE_LEXICAL_BPE_MAGIC: &[u8; 8] = b"NDSRF003";
const MAX_LEARNED_PIECE_BYTES: usize = 96;
const NO_TRIE_NODE: u32 = u32::MAX;
const TRIE_LINEAR_EDGE_LIMIT: usize = 8;
const TRIE_PACKED_NODE_LIMIT: u32 = 1 << 24;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TrieNode {
    first_edge: u32,
    terminal_id: u16,
    edge_count: u16,
}

type TrieEdge = u32;

#[inline(always)]
const fn pack_trie_edge(byte: u8, next: u32) -> TrieEdge {
    (next << 8) | byte as u32
}

#[inline(always)]
const fn trie_edge_byte(edge: TrieEdge) -> u8 {
    edge as u8
}

#[inline(always)]
const fn trie_edge_next(edge: TrieEdge) -> u32 {
    edge >> 8
}

#[derive(Default)]
struct TrieBuilderNode {
    terminal_id: u16,
    children: BTreeMap<u8, usize>,
}

/// Surface segmentation algorithm encoded by the vocabulary asset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SurfaceVocabularyKind {
    /// Legacy v0.2 longest-prefix trie segmentation.
    GreedyLongest,
    /// GPT/Tiktoken-style byte-pair encoding with morphology cuts as hard boundaries.
    ByteBpe,
    /// GPT/Tiktoken-style byte-pair encoding at lexical/scanner boundaries only.
    LexicalByteBpe,
}

/// Bounded deterministic frequency trainer for learned surface pieces.
#[derive(Clone, Debug)]
pub struct SurfaceVocabularyTrainer {
    counts: HashMap<Vec<u8>, u64>,
    max_candidates: usize,
}

impl SurfaceVocabularyTrainer {
    /// Creates a trainer with a deterministic heavy-candidate cap.
    ///
    /// # Errors
    ///
    /// Returns an error when the cap is too small to build a useful vocabulary.
    pub fn new(max_candidates: usize) -> Result<Self, TokenizerError> {
        if max_candidates < 256 {
            return Err(TokenizerError::InvalidConfiguration(
                "surface vocabulary candidate cap must be at least 256",
            ));
        }
        Ok(Self {
            counts: HashMap::new(),
            max_candidates,
        })
    }

    /// Observes exact unit and morphology-cut surface pieces from one document.
    ///
    /// # Errors
    ///
    /// Returns an error when stored spans cannot address the source bytes.
    pub fn observe(&mut self, document: &TokenizedDocument) -> Result<(), TokenizerError> {
        self.observe_with_policy(document, true)
    }

    /// Observes only tokenizer units, without morphology-derived candidate pieces.
    ///
    /// This is the controlled no-morphology ablation: the source documents,
    /// tokenizer units, vocabulary budget, and byte fallback remain unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when stored spans cannot address the source bytes.
    pub fn observe_without_morphology(
        &mut self,
        document: &TokenizedDocument,
    ) -> Result<(), TokenizerError> {
        self.observe_with_policy(document, false)
    }

    fn observe_with_policy(
        &mut self,
        document: &TokenizedDocument,
        use_morphology: bool,
    ) -> Result<(), TokenizerError> {
        let units = document.units();
        for (unit_index, unit) in units.iter().enumerate() {
            let start = usize::try_from(unit.span.start)
                .map_err(|_| TokenizerError::LengthOverflow("surface unit start"))?;
            let end = usize::try_from(unit.span.end)
                .map_err(|_| TokenizerError::LengthOverflow("surface unit end"))?;
            let unit_bytes = document
                .raw()
                .get(start..end)
                .ok_or(TokenizerError::UnitOutsideDocument)?;
            self.observe_candidate(unit_bytes, 1);
            self.observe_non_ascii_scalars(unit_bytes);

            if use_morphology {
                let mut left = start;
                for boundary in unit
                    .cuts
                    .iter()
                    .copied()
                    .chain(std::iter::once(unit.span.end))
                {
                    let right = usize::try_from(boundary)
                        .map_err(|_| TokenizerError::LengthOverflow("surface piece end"))?;
                    let piece = document
                        .raw()
                        .get(left..right)
                        .ok_or(TokenizerError::UnitOutsideDocument)?;
                    self.observe_candidate(piece, 4);
                    left = right;
                }
            }

            if let Some(following) = units.get(unit_index.saturating_add(1)) {
                if let Some(range) = prefix_bridge_range(
                    document.raw(),
                    unit.span,
                    unit.kind,
                    following.span,
                    following.kind,
                    &following.cuts,
                    use_morphology,
                )? {
                    let piece = document
                        .raw()
                        .get(range)
                        .ok_or(TokenizerError::UnitOutsideDocument)?;
                    self.observe_candidate(piece, if use_morphology { 4 } else { 1 });
                }
            }
        }
        if self.counts.len() > self.max_candidates.saturating_mul(2) {
            self.prune();
        }
        Ok(())
    }

    /// Finalizes the highest-frequency learned pieces inside a total vocabulary budget.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested vocabulary cannot include fixed byte fallback IDs.
    pub fn finish(mut self, vocabulary_size: usize) -> Result<SurfaceVocabulary, TokenizerError> {
        let fixed = usize::try_from(SURFACE_ENTRY_BASE_ID)
            .map_err(|_| TokenizerError::LengthOverflow("surface fixed vocabulary"))?;
        if vocabulary_size <= fixed || vocabulary_size > usize::from(u16::MAX) + 1 {
            return Err(TokenizerError::InvalidConfiguration(
                "surface vocabulary size must be in 260..=65536",
            ));
        }
        self.prune();
        let learned = vocabulary_size - fixed;
        let mut ranked = self.counts.into_iter().collect::<Vec<_>>();
        ranked.sort_unstable_by(|left, right| {
            right
                .1
                .cmp(&left.1)
                .then_with(|| left.0.as_slice().cmp(right.0.as_slice()))
        });
        ranked.truncate(learned);
        SurfaceVocabulary::from_ranked(ranked.into_iter().map(|entry| entry.0).collect())
    }

    /// Number of currently retained candidate strings.
    #[must_use]
    pub fn candidate_count(&self) -> usize {
        self.counts.len()
    }

    fn observe_candidate(&mut self, value: &[u8], weight: u64) {
        if value.is_empty() || value.len() > MAX_LEARNED_PIECE_BYTES {
            return;
        }
        let count = self.counts.entry(value.to_vec()).or_insert(0);
        *count = count.saturating_add(weight);
    }

    fn observe_non_ascii_scalars(&mut self, value: &[u8]) {
        let Ok(text) = std::str::from_utf8(value) else {
            return;
        };
        for character in text.chars().filter(|character| !character.is_ascii()) {
            let mut encoded = [0_u8; 4];
            let scalar = character.encode_utf8(&mut encoded);
            self.observe_candidate(scalar.as_bytes(), 1);
        }
    }

    fn prune(&mut self) {
        if self.counts.len() <= self.max_candidates {
            return;
        }
        let mut ranked = self.counts.drain().collect::<Vec<_>>();
        ranked.sort_unstable_by(|left, right| {
            right
                .1
                .cmp(&left.1)
                .then_with(|| left.0.as_slice().cmp(right.0.as_slice()))
        });
        ranked.truncate(self.max_candidates);
        self.counts.extend(ranked);
    }
}

fn prefix_bridge_range(
    raw: &[u8],
    whitespace_span: ByteSpan,
    whitespace_kind: LexicalKind,
    following_span: ByteSpan,
    following_kind: LexicalKind,
    following_cuts: &[u64],
    use_morphology: bool,
) -> Result<Option<Range<usize>>, TokenizerError> {
    if whitespace_kind != LexicalKind::Whitespace
        || !matches!(following_kind, LexicalKind::Word | LexicalKind::Number)
        || whitespace_span.end != following_span.start
    {
        return Ok(None);
    }
    let whitespace_start = usize::try_from(whitespace_span.start)
        .map_err(|_| TokenizerError::LengthOverflow("surface prefix whitespace start"))?;
    let whitespace_end = usize::try_from(whitespace_span.end)
        .map_err(|_| TokenizerError::LengthOverflow("surface prefix whitespace end"))?;
    if raw.get(whitespace_start..whitespace_end) != Some(b" ".as_slice()) {
        return Ok(None);
    }
    let following_end = if use_morphology {
        following_cuts.first().copied().unwrap_or(following_span.end)
    } else {
        following_span.end
    };
    let following_end = usize::try_from(following_end)
        .map_err(|_| TokenizerError::LengthOverflow("surface prefix bridge end"))?;
    if following_end <= whitespace_end {
        return Ok(None);
    }
    Ok(Some(whitespace_start..following_end))
}

fn flat_unit_requires_split(
    raw: &[u8],
    unit: &FlatSurfaceUnit,
    maximum_chars: usize,
) -> Result<bool, TokenizerError> {
    let bytes = crate::unit_bytes(raw, unit.span)?;
    let should_split =
        unit.status == crate::TokenStatus::Code || unit.mode == crate::TokenMode::Opaque;
    if !should_split {
        return Ok(false);
    }
    if unit.mode == crate::TokenMode::Opaque {
        return Ok(bytes.len() > maximum_chars);
    }
    if bytes.len() <= maximum_chars {
        return Ok(false);
    }
    let text = std::str::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8Unit)?;
    Ok(text.chars().nth(maximum_chars).is_some())
}

/// Returns the exact byte spans inside which surface BPE merges are allowed.
///
/// Turkish morphology cuts stay hard boundaries. One ordinary ASCII inter-word
/// space may prefix the following word/number's first morphology segment. All
/// other tokenizer-unit boundaries remain hard. The spans are contiguous and
/// cover the original document exactly.
///
/// # Errors
///
/// Returns an error if document spans or offsets are invalid.
pub fn surface_bpe_segments(
    document: &TokenizedDocument,
    use_morphology: bool,
) -> Result<Vec<ByteSpan>, TokenizerError> {
    document.validate()?;
    surface_bpe_segments_from_units(document.raw(), document.units(), use_morphology)
}

fn surface_bpe_segments_from_units(
    raw: &[u8],
    units: &[TokenizedUnit],
    use_morphology: bool,
) -> Result<Vec<ByteSpan>, TokenizerError> {
    validate_units(raw, units)?;
    let mut segments = Vec::with_capacity(units.len().saturating_mul(2));
    let mut unit_index = 0_usize;
    while unit_index < units.len() {
        let unit = &units[unit_index];
        if let Some(following) = units.get(unit_index.saturating_add(1)) {
            if let Some(bridge) = prefix_bridge_range(
                raw,
                unit.span,
                unit.kind,
                following.span,
                following.kind,
                &following.cuts,
                use_morphology,
            )? {
                segments.push(ByteSpan {
                    start: u64::try_from(bridge.start).map_err(|_| {
                        TokenizerError::LengthOverflow("surface BPE segment start")
                    })?,
                    end: u64::try_from(bridge.end).map_err(|_| {
                        TokenizerError::LengthOverflow("surface BPE segment end")
                    })?,
                });
                if use_morphology {
                    let mut left = u64::try_from(bridge.end).map_err(|_| {
                        TokenizerError::LengthOverflow("surface BPE bridged remainder")
                    })?;
                    for boundary in following
                        .cuts
                        .iter()
                        .copied()
                        .chain(std::iter::once(following.span.end))
                    {
                        if boundary > left {
                            segments.push(ByteSpan {
                                start: left,
                                end: boundary,
                            });
                            left = boundary;
                        }
                    }
                }
                unit_index = unit_index.saturating_add(2);
                continue;
            }
        }

        if use_morphology {
            let mut left = unit.span.start;
            for boundary in unit
                .cuts
                .iter()
                .copied()
                .chain(std::iter::once(unit.span.end))
            {
                if boundary > left {
                    segments.push(ByteSpan {
                        start: left,
                        end: boundary,
                    });
                    left = boundary;
                }
            }
        } else {
            segments.push(unit.span);
        }
        unit_index = unit_index.saturating_add(1);
    }

    let mut expected = 0_u64;
    for (index, segment) in segments.iter().enumerate() {
        if segment.start != expected || segment.end <= segment.start {
            return Err(TokenizerError::InvalidUnitCoverage {
                index,
                expected,
                start: segment.start,
                end: segment.end,
            });
        }
        expected = segment.end;
    }
    let document_end = u64::try_from(raw.len())
        .map_err(|_| TokenizerError::LengthOverflow("surface BPE document length"))?;
    if expected != document_end {
        return Err(TokenizerError::InvalidUnitCoverage {
            index: segments.len(),
            expected,
            start: expected,
            end: document_end,
        });
    }
    Ok(segments)
}

/// Learned exact surface pieces plus complete one-byte fallback.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SurfaceVocabulary {
    kind: SurfaceVocabularyKind,
    entries: Vec<Vec<u8>>,
    lookup: HashMap<Vec<u8>, u32>,
    trie_root: [u32; 256],
    trie_nodes: Vec<TrieNode>,
    trie_edges: Vec<TrieEdge>,
}

impl SurfaceVocabulary {
    /// Builds a vocabulary from entries already ordered by desired token ID.
    ///
    /// # Errors
    ///
    /// Returns an error for empty, duplicate, oversized, or overflowing entries.
    pub fn from_ranked(entries: Vec<Vec<u8>>) -> Result<Self, TokenizerError> {
        Self::from_entries(entries, SurfaceVocabularyKind::GreedyLongest)
    }

    /// Builds a byte-BPE vocabulary whose entry order is the merge priority.
    ///
    /// Every learned entry must be a valid merge of two tokens that are already
    /// available as a raw byte or an earlier learned entry. This invariant is
    /// the same rank-order contract relied on by Tiktoken-style BPE encoders.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid merge order, duplicate/oversized entries, or overflow.
    pub fn from_bpe_ranked(entries: Vec<Vec<u8>>) -> Result<Self, TokenizerError> {
        validate_bpe_merge_order(&entries)?;
        Self::from_entries(entries, SurfaceVocabularyKind::ByteBpe)
    }

    /// Builds a lexical-boundary byte-BPE vocabulary. Morphology remains available
    /// as analysis metadata but does not force final LM token boundaries.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid merge order, duplicate/oversized entries, or overflow.
    pub fn from_lexical_bpe_ranked(entries: Vec<Vec<u8>>) -> Result<Self, TokenizerError> {
        validate_bpe_merge_order(&entries)?;
        Self::from_entries(entries, SurfaceVocabularyKind::LexicalByteBpe)
    }

    fn from_entries(
        entries: Vec<Vec<u8>>,
        kind: SurfaceVocabularyKind,
    ) -> Result<Self, TokenizerError> {
        if entries.len()
            > usize::from(u16::MAX)
                .saturating_add(1)
                .saturating_sub(usize::try_from(SURFACE_ENTRY_BASE_ID).unwrap_or(259))
        {
            return Err(TokenizerError::InvalidVocabulary(
                "surface vocabulary exceeds uint16 IDs",
            ));
        }
        let mut unique = HashSet::with_capacity(entries.len());
        if entries.iter().any(|entry| {
            entry.is_empty()
                || entry.len() > MAX_LEARNED_PIECE_BYTES
                || !unique.insert(entry.clone())
        }) {
            return Err(TokenizerError::InvalidVocabulary(
                "surface entries must be unique non-empty bounded byte strings",
            ));
        }
        let mut lookup = HashMap::with_capacity(entries.len());
        let mut builders = vec![TrieBuilderNode::default()];
        for (index, entry) in entries.iter().enumerate() {
            let index = u32::try_from(index)
                .map_err(|_| TokenizerError::LengthOverflow("surface entry index"))?;
            let id = SURFACE_ENTRY_BASE_ID
                .checked_add(index)
                .ok_or(TokenizerError::LengthOverflow("surface token ID"))?;
            lookup.insert(entry.clone(), id);
            let mut node = 0_usize;
            for byte in entry {
                let next = builders[node].children.get(byte).copied();
                node = if let Some(next) = next {
                    next
                } else {
                    let next = builders.len();
                    builders.push(TrieBuilderNode::default());
                    builders[node].children.insert(*byte, next);
                    next
                };
            }
            builders[node].terminal_id = u16::try_from(id)
                .map_err(|_| TokenizerError::LengthOverflow("surface trie token ID"))?;
        }
        let mut trie_root = [NO_TRIE_NODE; 256];
        for (byte, next) in &builders[0].children {
            trie_root[usize::from(*byte)] = u32::try_from(*next)
                .map_err(|_| TokenizerError::LengthOverflow("surface trie root"))?;
        }
        let edge_count = builders.iter().try_fold(0_usize, |total, node| {
            total
                .checked_add(node.children.len())
                .ok_or(TokenizerError::LengthOverflow("surface trie edges"))
        })?;
        let mut trie_nodes = Vec::with_capacity(builders.len());
        let mut trie_edges = Vec::with_capacity(edge_count);
        for node in &builders {
            let first_edge = u32::try_from(trie_edges.len())
                .map_err(|_| TokenizerError::LengthOverflow("surface trie edge offset"))?;
            let edge_count = u16::try_from(node.children.len())
                .map_err(|_| TokenizerError::LengthOverflow("surface trie branch count"))?;
            for (byte, next) in &node.children {
                let next = u32::try_from(*next)
                    .map_err(|_| TokenizerError::LengthOverflow("surface trie node"))?;
                if next >= TRIE_PACKED_NODE_LIMIT {
                    return Err(TokenizerError::LengthOverflow("surface packed trie node"));
                }
                trie_edges.push(pack_trie_edge(*byte, next));
            }
            trie_nodes.push(TrieNode {
                first_edge,
                terminal_id: node.terminal_id,
                edge_count,
            });
        }
        Ok(Self {
            kind,
            entries,
            lookup,
            trie_root,
            trie_nodes,
            trie_edges,
        })
    }

    /// Segmentation algorithm encoded by this asset.
    #[must_use]
    pub const fn kind(&self) -> SurfaceVocabularyKind {
        self.kind
    }

    /// Total embedding vocabulary size including specials and byte fallback.
    #[must_use]
    pub fn len(&self) -> usize {
        usize::try_from(SURFACE_ENTRY_BASE_ID).unwrap_or(259) + self.entries.len()
    }

    /// Whether no learned surface piece is present.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Ordered learned byte strings, excluding fixed IDs.
    #[must_use]
    pub fn entries(&self) -> &[Vec<u8>] {
        &self.entries
    }

    /// Looks up an exact learned surface piece.
    #[must_use]
    pub fn id_for_piece(&self, value: &[u8]) -> Option<u32> {
        self.lookup.get(value).copied()
    }

    /// Returns bytes represented by one learned or fallback token ID.
    #[must_use]
    pub fn bytes_for_id(&self, id: u32) -> Option<&[u8]> {
        if (SURFACE_BYTE_BASE_ID..SURFACE_ENTRY_BASE_ID).contains(&id) {
            return None;
        }
        let index = id.checked_sub(SURFACE_ENTRY_BASE_ID)?;
        self.entries
            .get(usize::try_from(index).ok()?)
            .map(Vec::as_slice)
    }

    /// Encodes exact tokenizer surface pieces without boundary-control tokens.
    ///
    /// The exact boundary policy is encoded by the vocabulary asset. Legacy and
    /// morphology-BPE assets preserve morphology cuts; lexical-BPE keeps scanner
    /// boundaries while allowing one ordinary ASCII inter-word space to prefix
    /// the following word/number. Unmatched content always falls back to exact bytes.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid spans, ID overflow, or byte-accounting mismatch.
    pub fn encode_document(
        &self,
        document: &TokenizedDocument,
        newline: bool,
    ) -> Result<TrainingEncoding, TokenizerError> {
        self.encode_document_with_policy(document, newline, true)
    }

    /// Encodes learned pieces inside tokenizer units while ignoring morphology cuts.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid spans, ID overflow, or byte-accounting mismatch.
    pub fn encode_document_without_morphology(
        &self,
        document: &TokenizedDocument,
        newline: bool,
    ) -> Result<TrainingEncoding, TokenizerError> {
        self.encode_document_with_policy(document, newline, false)
    }

    fn encode_document_with_policy(
        &self,
        document: &TokenizedDocument,
        newline: bool,
        use_morphology: bool,
    ) -> Result<TrainingEncoding, TokenizerError> {
        document.validate()?;
        self.encode_units(document.raw(), document.units(), newline, use_morphology)
    }

    pub(crate) fn encode_units(
        &self,
        raw: &[u8],
        units: &[TokenizedUnit],
        newline: bool,
        use_morphology: bool,
    ) -> Result<TrainingEncoding, TokenizerError> {
        let mut output = TrainingEncoding {
            ids: Vec::with_capacity(raw.len().saturating_add(2)),
            lengths: Vec::with_capacity(raw.len().saturating_add(2)),
        };
        self.encode_units_into(
            raw,
            units,
            newline,
            use_morphology,
            &mut output.ids,
            &mut output.lengths,
        )?;
        Ok(output)
    }

    pub(crate) fn encode_units_into(
        &self,
        raw: &[u8],
        units: &[TokenizedUnit],
        newline: bool,
        use_morphology: bool,
        ids: &mut Vec<u16>,
        lengths: &mut Vec<u8>,
    ) -> Result<(), TokenizerError> {
        validate_units(raw, units)?;
        let length_start = lengths.len();
        push_parts(ids, lengths, SURFACE_BOS_ID, 0)?;
        let use_morphology = use_morphology
            && self.kind != SurfaceVocabularyKind::LexicalByteBpe;
        for segment in surface_bpe_segments_from_units(raw, units, use_morphology)? {
            let start = usize::try_from(segment.start)
                .map_err(|_| TokenizerError::LengthOverflow("surface segment start"))?;
            let end = usize::try_from(segment.end)
                .map_err(|_| TokenizerError::LengthOverflow("surface segment end"))?;
            self.encode_segment(raw, start, end, ids, lengths)?;
        }
        if newline {
            push_parts(ids, lengths, SURFACE_BYTE_BASE_ID + u32::from(b'\n'), 1)?;
        }
        push_parts(ids, lengths, SURFACE_EOS_ID, 0)?;
        let expected = raw.len().saturating_add(usize::from(newline));
        let actual = lengths[length_start..]
            .iter()
            .map(|value| usize::from(*value))
            .sum::<usize>();
        if actual != expected {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "surface vocabulary byte accounting differs from source",
            ));
        }
        Ok(())
    }

    pub(crate) fn begin_cached_surface_document(
        &self,
        ids: &mut Vec<u16>,
        lengths: &mut Vec<u8>,
    ) -> Result<usize, TokenizerError> {
        let length_start = lengths.len();
        push_parts(ids, lengths, SURFACE_BOS_ID, 0)?;
        Ok(length_start)
    }

    pub(crate) fn encode_flat_unit_direct(
        &self,
        raw: &[u8],
        unit: &FlatSurfaceUnit,
        maximum_chars: usize,
        ids: &mut Vec<u16>,
        lengths: &mut Vec<u8>,
    ) -> Result<(), TokenizerError> {
        self.encode_flat_range_into(
            raw,
            std::slice::from_ref(unit),
            &[],
            0..1,
            maximum_chars,
            false,
            ids,
            lengths,
        )
    }

    pub(crate) fn finish_cached_surface_document(
        &self,
        raw_len: usize,
        newline: bool,
        length_start: usize,
        ids: &mut Vec<u16>,
        lengths: &mut Vec<u8>,
    ) -> Result<(), TokenizerError> {
        if newline {
            push_parts(ids, lengths, SURFACE_BYTE_BASE_ID + u32::from(b'\n'), 1)?;
        }
        push_parts(ids, lengths, SURFACE_EOS_ID, 0)?;
        let expected = raw_len.saturating_add(usize::from(newline));
        let actual = lengths[length_start..]
            .iter()
            .map(|value| usize::from(*value))
            .sum::<usize>();
        if actual != expected {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "cached Turkish surface byte accounting differs from source",
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_flat_units_with_programs(
        &self,
        raw: &[u8],
        units: &[FlatSurfaceUnit],
        cuts: &[u64],
        programs: &[SurfaceProgramUse],
        maximum_chars: usize,
        newline: bool,
        use_morphology: bool,
        ids: &mut Vec<u16>,
        lengths: &mut Vec<u8>,
    ) -> Result<(), TokenizerError> {
        validate_flat_units(raw, units, cuts)?;
        let length_start = lengths.len();
        push_parts(ids, lengths, SURFACE_BOS_ID, 0)?;
        let mut cursor = 0_usize;
        for usage in programs {
            if usage.start_unit < cursor
                || usage.end_unit <= usage.start_unit
                || usage.end_unit > units.len()
                || usage.program.surface_ids.len() != usage.program.surface_lengths.len()
            {
                return Err(TokenizerError::InvalidTrainingEncoding(
                    "surface program unit range or output cardinality is invalid",
                ));
            }
            self.encode_flat_range_into(
                raw,
                units,
                cuts,
                cursor..usage.start_unit,
                maximum_chars,
                use_morphology,
                ids,
                lengths,
            )?;
            ids.extend_from_slice(&usage.program.surface_ids);
            lengths.extend_from_slice(&usage.program.surface_lengths);
            cursor = usage.end_unit;
        }
        self.encode_flat_range_into(
            raw,
            units,
            cuts,
            cursor..units.len(),
            maximum_chars,
            use_morphology,
            ids,
            lengths,
        )?;
        if newline {
            push_parts(ids, lengths, SURFACE_BYTE_BASE_ID + u32::from(b'\n'), 1)?;
        }
        push_parts(ids, lengths, SURFACE_EOS_ID, 0)?;
        let expected = raw.len().saturating_add(usize::from(newline));
        let actual = lengths[length_start..]
            .iter()
            .map(|value| usize::from(*value))
            .sum::<usize>();
        if actual != expected {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "surface program byte accounting differs from source",
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_flat_range_into(
        &self,
        raw: &[u8],
        units: &[FlatSurfaceUnit],
        cuts: &[u64],
        range: Range<usize>,
        maximum_chars: usize,
        use_morphology: bool,
        ids: &mut Vec<u16>,
        lengths: &mut Vec<u8>,
    ) -> Result<(), TokenizerError> {
        let selected = units
            .get(range)
            .ok_or(TokenizerError::InvalidTrainingEncoding(
                "flat surface encode range is invalid",
            ))?;
        let use_morphology = use_morphology
            && self.kind != SurfaceVocabularyKind::LexicalByteBpe;
        let mut unit_index = 0_usize;
        while unit_index < selected.len() {
            let unit = &selected[unit_index];
            if let Some(following) = selected.get(unit_index.saturating_add(1)) {
                if !flat_unit_requires_split(raw, following, maximum_chars)? {
                    let following_cuts = following.cuts(cuts)?;
                    if let Some(bridge) = prefix_bridge_range(
                        raw,
                        unit.span,
                        unit.kind,
                        following.span,
                        following.kind,
                        following_cuts,
                        use_morphology,
                    )? {
                        self.encode_segment(raw, bridge.start, bridge.end, ids, lengths)?;
                        if use_morphology {
                            let mut left = bridge.end;
                            for boundary in following_cuts
                                .iter()
                                .copied()
                                .chain(std::iter::once(following.span.end))
                            {
                                let right = usize::try_from(boundary).map_err(|_| {
                                    TokenizerError::LengthOverflow(
                                        "flat surface bridged segment end",
                                    )
                                })?;
                                if right <= left {
                                    continue;
                                }
                                self.encode_segment(raw, left, right, ids, lengths)?;
                                left = right;
                            }
                        }
                        unit_index = unit_index.saturating_add(2);
                        continue;
                    }
                }
            }

            if flat_unit_requires_split(raw, unit, maximum_chars)? {
                for span in crate::chunk_span(
                    raw,
                    unit.span,
                    maximum_chars,
                    unit.mode == crate::TokenMode::Opaque,
                )? {
                    let start = usize::try_from(span.start)
                        .map_err(|_| TokenizerError::LengthOverflow("flat surface chunk start"))?;
                    let end = usize::try_from(span.end)
                        .map_err(|_| TokenizerError::LengthOverflow("flat surface chunk end"))?;
                    self.encode_segment(raw, start, end, ids, lengths)?;
                }
                unit_index = unit_index.saturating_add(1);
                continue;
            }
            let start = usize::try_from(unit.span.start)
                .map_err(|_| TokenizerError::LengthOverflow("flat surface unit start"))?;
            if use_morphology {
                let mut left = start;
                for boundary in unit
                    .cuts(cuts)?
                    .iter()
                    .copied()
                    .chain(std::iter::once(unit.span.end))
                {
                    let right = usize::try_from(boundary)
                        .map_err(|_| TokenizerError::LengthOverflow("flat surface segment end"))?;
                    self.encode_segment(raw, left, right, ids, lengths)?;
                    left = right;
                }
            } else {
                let end = usize::try_from(unit.span.end)
                    .map_err(|_| TokenizerError::LengthOverflow("flat surface unit end"))?;
                self.encode_segment(raw, start, end, ids, lengths)?;
            }
            unit_index = unit_index.saturating_add(1);
        }
        Ok(())
    }

    /// Reconstructs exact bytes from surface-piece token IDs.
    ///
    /// Padding and document-control IDs contribute no bytes. Every other ID
    /// must be a learned piece or exact byte fallback.
    ///
    /// # Errors
    ///
    /// Returns an error for an ID outside this vocabulary.
    pub fn decode_ids(&self, ids: &[u16]) -> Result<Vec<u8>, TokenizerError> {
        let mut output = Vec::new();
        for raw_id in ids {
            let id = u32::from(*raw_id);
            match id {
                SURFACE_PAD_ID | SURFACE_BOS_ID | SURFACE_EOS_ID => {}
                SURFACE_BYTE_BASE_ID..SURFACE_ENTRY_BASE_ID => {
                    output.push(
                        u8::try_from(id - SURFACE_BYTE_BASE_ID).map_err(|_| {
                            TokenizerError::InvalidVocabulary("bad byte fallback ID")
                        })?,
                    );
                }
                _ => {
                    let index = id
                        .checked_sub(SURFACE_ENTRY_BASE_ID)
                        .and_then(|value| usize::try_from(value).ok())
                        .ok_or(TokenizerError::InvalidVocabulary("bad learned surface ID"))?;
                    output.extend_from_slice(self.entries.get(index).ok_or(
                        TokenizerError::InvalidVocabulary("unknown surface token ID"),
                    )?);
                }
            }
        }
        Ok(output)
    }

    /// Stable checksum-protected binary representation.
    ///
    /// # Errors
    ///
    /// Returns an error when entry lengths cannot fit the stable format.
    pub fn to_bytes(&self) -> Result<Vec<u8>, TokenizerError> {
        let mut payload = Vec::new();
        for entry in &self.entries {
            let length = u32::try_from(entry.len())
                .map_err(|_| TokenizerError::LengthOverflow("surface entry length"))?;
            payload.extend_from_slice(&length.to_le_bytes());
            payload.extend_from_slice(entry);
        }
        let mut output = Vec::with_capacity(44 + payload.len());
        output.extend_from_slice(match self.kind {
            SurfaceVocabularyKind::GreedyLongest => SURFACE_GREEDY_MAGIC,
            SurfaceVocabularyKind::ByteBpe => SURFACE_BPE_MAGIC,
            SurfaceVocabularyKind::LexicalByteBpe => SURFACE_LEXICAL_BPE_MAGIC,
        });
        output.extend_from_slice(
            &u32::try_from(self.entries.len())
                .map_err(|_| TokenizerError::LengthOverflow("surface entry count"))?
                .to_le_bytes(),
        );
        output.extend_from_slice(&Sha256::digest(&payload));
        output.extend_from_slice(&payload);
        Ok(output)
    }

    /// Loads the stable checksum-protected binary representation.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed identity, checksum, lengths, or entries.
    pub fn from_bytes(input: &[u8]) -> Result<Self, TokenizerError> {
        if input.len() < 44 {
            return Err(TokenizerError::InvalidVocabulary(
                "bad surface vocabulary header",
            ));
        }
        let kind = match input.get(..8) {
            Some(value) if value == SURFACE_GREEDY_MAGIC => SurfaceVocabularyKind::GreedyLongest,
            Some(value) if value == SURFACE_BPE_MAGIC => SurfaceVocabularyKind::ByteBpe,
            Some(value) if value == SURFACE_LEXICAL_BPE_MAGIC => {
                SurfaceVocabularyKind::LexicalByteBpe
            }
            _ => {
                return Err(TokenizerError::InvalidVocabulary(
                    "bad surface vocabulary header",
                ));
            }
        };
        let count = u32::from_le_bytes(
            input[8..12]
                .try_into()
                .map_err(|_| TokenizerError::InvalidVocabulary("truncated surface count"))?,
        ) as usize;
        let expected: [u8; 32] = input[12..44]
            .try_into()
            .map_err(|_| TokenizerError::InvalidVocabulary("truncated surface hash"))?;
        let payload = &input[44..];
        let actual: [u8; 32] = Sha256::digest(payload).into();
        if actual != expected {
            return Err(TokenizerError::InvalidVocabulary(
                "surface vocabulary checksum mismatch",
            ));
        }
        let mut position = 0_usize;
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            let length_end = position
                .checked_add(4)
                .ok_or(TokenizerError::LengthOverflow("surface payload position"))?;
            let length = u32::from_le_bytes(
                payload
                    .get(position..length_end)
                    .ok_or(TokenizerError::InvalidVocabulary(
                        "truncated surface entry length",
                    ))?
                    .try_into()
                    .map_err(|_| TokenizerError::InvalidVocabulary("bad surface entry length"))?,
            ) as usize;
            position = length_end;
            let end = position
                .checked_add(length)
                .ok_or(TokenizerError::LengthOverflow("surface payload entry"))?;
            entries.push(
                payload
                    .get(position..end)
                    .ok_or(TokenizerError::InvalidVocabulary("truncated surface entry"))?
                    .to_vec(),
            );
            position = end;
        }
        if position != payload.len() {
            return Err(TokenizerError::InvalidVocabulary(
                "trailing surface vocabulary bytes",
            ));
        }
        match kind {
            SurfaceVocabularyKind::GreedyLongest => Self::from_ranked(entries),
            SurfaceVocabularyKind::ByteBpe => Self::from_bpe_ranked(entries),
            SurfaceVocabularyKind::LexicalByteBpe => Self::from_lexical_bpe_ranked(entries),
        }
    }

    #[inline(always)]
    fn trie_child(&self, node: u32, byte: u8) -> Option<u32> {
        let node = self.trie_nodes.get(usize::try_from(node).ok()?)?;
        let start = usize::try_from(node.first_edge).ok()?;
        let end = start.checked_add(usize::from(node.edge_count))?;
        let edges = self.trie_edges.get(start..end)?;
        if edges.len() <= TRIE_LINEAR_EDGE_LIMIT {
            edges
                .iter()
                .copied()
                .find(|edge| trie_edge_byte(*edge) == byte)
                .map(trie_edge_next)
        } else {
            edges
                .binary_search_by_key(&byte, |edge| trie_edge_byte(*edge))
                .ok()
                .map(|index| trie_edge_next(edges[index]))
        }
    }

    fn encode_segment(
        &self,
        raw: &[u8],
        start: usize,
        end: usize,
        ids: &mut Vec<u16>,
        lengths: &mut Vec<u8>,
    ) -> Result<(), TokenizerError> {
        match self.kind {
            SurfaceVocabularyKind::GreedyLongest => {
                self.encode_segment_greedy(raw, start, end, ids, lengths)
            }
            SurfaceVocabularyKind::ByteBpe | SurfaceVocabularyKind::LexicalByteBpe => {
                self.encode_segment_bpe(raw, start, end, ids, lengths)
            }
        }
    }

    fn encode_segment_greedy(
        &self,
        raw: &[u8],
        start: usize,
        end: usize,
        ids: &mut Vec<u16>,
        lengths: &mut Vec<u8>,
    ) -> Result<(), TokenizerError> {
        if start > end || end > raw.len() {
            return Err(TokenizerError::UnitOutsideDocument);
        }
        let mut position = start;
        while position < end {
            let mut matched = None;
            let first = self.trie_root[usize::from(raw[position])];
            if first != NO_TRIE_NODE {
                let mut node = first;
                let mut cursor = position + 1;
                let terminal = self.trie_nodes[usize::try_from(node).unwrap_or(0)].terminal_id;
                if terminal != 0 {
                    matched = Some((u32::from(terminal), 1_usize));
                }
                while cursor < end {
                    let Some(next) = self.trie_child(node, raw[cursor]) else {
                        break;
                    };
                    node = next;
                    cursor += 1;
                    let terminal = self.trie_nodes[usize::try_from(node).unwrap_or(0)].terminal_id;
                    if terminal != 0 {
                        matched = Some((u32::from(terminal), cursor - position));
                    }
                }
            }
            if let Some((id, length)) = matched {
                push_parts(
                    ids,
                    lengths,
                    id,
                    u8::try_from(length)
                        .map_err(|_| TokenizerError::LengthOverflow("surface token length"))?,
                )?;
                position += length;
            } else {
                push_parts(
                    ids,
                    lengths,
                    SURFACE_BYTE_BASE_ID + u32::from(raw[position]),
                    1,
                )?;
                position += 1;
            }
        }
        Ok(())
    }

    fn encode_segment_bpe(
        &self,
        raw: &[u8],
        start: usize,
        end: usize,
        ids: &mut Vec<u16>,
        lengths: &mut Vec<u8>,
    ) -> Result<(), TokenizerError> {
        if start > end || end > raw.len() {
            return Err(TokenizerError::UnitOutsideDocument);
        }
        if start == end {
            return Ok(());
        }
        let piece = raw
            .get(start..end)
            .ok_or(TokenizerError::UnitOutsideDocument)?;
        if piece.len() == 1 {
            return push_parts(
                ids,
                lengths,
                SURFACE_BYTE_BASE_ID + u32::from(piece[0]),
                1,
            );
        }

        // Boundaries delimit the current byte tokens. Removing one boundary merges
        // the adjacent pair. The learned-entry ID order is the merge priority.
        let mut boundaries = (0..=piece.len()).collect::<Vec<_>>();
        loop {
            let mut best_rank = u32::MAX;
            let mut best_boundary = None;
            for index in 0..boundaries.len().saturating_sub(2) {
                let left = boundaries[index];
                let right = boundaries[index + 2];
                if let Some(id) = self.id_for_piece(&piece[left..right]) {
                    let rank = id.saturating_sub(SURFACE_ENTRY_BASE_ID);
                    if rank < best_rank {
                        best_rank = rank;
                        best_boundary = Some(index + 1);
                    }
                }
            }
            let Some(boundary) = best_boundary else {
                break;
            };
            boundaries.remove(boundary);
        }

        for window in boundaries.windows(2) {
            let left = window[0];
            let right = window[1];
            let token = &piece[left..right];
            if token.len() == 1 {
                push_parts(
                    ids,
                    lengths,
                    SURFACE_BYTE_BASE_ID + u32::from(token[0]),
                    1,
                )?;
            } else {
                let id = self.id_for_piece(token).ok_or(
                    TokenizerError::InvalidVocabulary(
                        "surface BPE merge produced a token missing from the vocabulary",
                    ),
                )?;
                push_parts(
                    ids,
                    lengths,
                    id,
                    u8::try_from(token.len()).map_err(|_| {
                        TokenizerError::LengthOverflow("surface BPE token length")
                    })?,
                )?;
            }
        }
        Ok(())
    }
}

fn validate_bpe_merge_order(entries: &[Vec<u8>]) -> Result<(), TokenizerError> {
    let mut available = HashSet::<Vec<u8>>::with_capacity(entries.len().saturating_mul(2));
    for (rank, entry) in entries.iter().enumerate() {
        if entry.len() < 2 {
            return Err(TokenizerError::InvalidVocabulary(
                "surface BPE learned entries must contain at least two bytes",
            ));
        }
        let mut valid = false;
        for split in 1..entry.len() {
            let left = &entry[..split];
            let right = &entry[split..];
            let left_available = left.len() == 1 || available.contains(left);
            let right_available = right.len() == 1 || available.contains(right);
            if left_available && right_available {
                valid = true;
                break;
            }
        }
        if !valid {
            let _ = rank;
            return Err(TokenizerError::InvalidVocabulary(
                "surface BPE entry cannot be formed from earlier merge ranks",
            ));
        }
        if !available.insert(entry.clone()) {
            return Err(TokenizerError::InvalidVocabulary(
                "surface BPE entries must be unique",
            ));
        }
    }
    Ok(())
}

fn validate_flat_units(
    raw: &[u8],
    units: &[FlatSurfaceUnit],
    cuts: &[u64],
) -> Result<(), TokenizerError> {
    let document_len = u64::try_from(raw.len())
        .map_err(|_| TokenizerError::LengthOverflow("flat surface document length"))?;
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
        for cut in unit.cuts(cuts)? {
            if *cut <= previous || *cut >= unit.span.end {
                return Err(TokenizerError::InvalidTrainingEncoding(
                    "flat surface cut is outside its unit or not increasing",
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

fn validate_units(raw: &[u8], units: &[TokenizedUnit]) -> Result<(), TokenizerError> {
    let document_len = u64::try_from(raw.len())
        .map_err(|_| TokenizerError::LengthOverflow("surface document length"))?;
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
                    "surface cut is outside its unit or not increasing",
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

fn push_parts(
    ids: &mut Vec<u16>,
    lengths: &mut Vec<u8>,
    id: u32,
    length: u8,
) -> Result<(), TokenizerError> {
    ids.push(u16::try_from(id).map_err(|_| TokenizerError::LengthOverflow("surface token ID"))?);
    lengths.push(length);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        SurfaceVocabulary, SurfaceVocabularyTrainer, SURFACE_BOS_ID, SURFACE_BYTE_BASE_ID,
        SURFACE_EOS_ID,
    };
    use crate::{Tokenizer, TokenizerConfig};

    fn id(value: u32) -> Result<u16, crate::TokenizerError> {
        u16::try_from(value)
            .map_err(|_| crate::TokenizerError::LengthOverflow("surface test token ID"))
    }

    #[test]
    fn learned_morphology_pieces_are_tokens_without_boundary_controls(
    ) -> Result<(), crate::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let documents = tokenizer.tokenize_batch(
            &[b"gidicem gidicem".to_vec(), b"geliyom gidicem".to_vec()],
            1,
        )?;
        let mut trainer = SurfaceVocabularyTrainer::new(4096)?;
        for document in &documents {
            trainer.observe(document)?;
        }
        let vocabulary = trainer.finish(512)?;
        for expected in [b"gid".as_slice(), b"ice".as_slice(), b"m".as_slice()] {
            assert!(vocabulary.id_for_piece(expected).is_some());
        }
        let encoded = vocabulary.encode_document(&documents[0], true)?;
        assert_eq!(encoded.ids.first().copied(), Some(id(SURFACE_BOS_ID)?));
        assert_eq!(encoded.ids.last().copied(), Some(id(SURFACE_EOS_ID)?));
        assert_eq!(
            encoded
                .lengths
                .iter()
                .map(|value| usize::from(*value))
                .sum::<usize>(),
            16
        );
        assert_eq!(
            encoded
                .lengths
                .iter()
                .fold(0_usize, |total, length| total + usize::from(*length == 0)),
            2
        );
        assert_eq!(vocabulary.decode_ids(&encoded.ids)?, b"gidicem gidicem\n");
        assert_eq!(
            SurfaceVocabulary::from_bytes(&vocabulary.to_bytes()?)?,
            vocabulary
        );
        Ok(())
    }

    #[test]
    fn no_morphology_ablation_can_cross_morphology_cuts() -> Result<(), crate::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let document = tokenizer.tokenize(b"gidicem".to_vec())?;
        assert!(!document.units()[0].cuts.is_empty());

        let mut trainer = SurfaceVocabularyTrainer::new(4096)?;
        trainer.observe_without_morphology(&document)?;
        let vocabulary = trainer.finish(512)?;
        assert!(vocabulary.id_for_piece(b"gidicem").is_some());

        let ablated = vocabulary.encode_document_without_morphology(&document, false)?;
        let morphology_aware = vocabulary.encode_document(&document, false)?;
        assert_eq!(ablated.lengths, vec![0, 7, 0]);
        assert!(morphology_aware.ids.len() > ablated.ids.len());
        assert_eq!(vocabulary.decode_ids(&ablated.ids)?, b"gidicem");
        assert_eq!(vocabulary.decode_ids(&morphology_aware.ids)?, b"gidicem");
        Ok(())
    }

    #[test]
    fn ordinary_space_can_prefix_the_following_morphology_segment(
    ) -> Result<(), crate::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let document = tokenizer.tokenize(b"gidicem gidicem".to_vec())?;
        assert!(!document.units()[0].cuts.is_empty());
        assert!(!document.units()[2].cuts.is_empty());

        let mut trainer = SurfaceVocabularyTrainer::new(4096)?;
        trainer.observe(&document)?;
        let vocabulary = trainer.finish(512)?;
        assert!(vocabulary.id_for_piece(b" gid").is_some());
        assert!(vocabulary.id_for_piece(b" gidice").is_none());

        let encoded = vocabulary.encode_document(&document, false)?;
        let bridge_id = id(vocabulary.id_for_piece(b" gid").expect("prefix piece"))?;
        assert!(encoded.ids.contains(&bridge_id));
        assert_eq!(vocabulary.decode_ids(&encoded.ids)?, b"gidicem gidicem");
        Ok(())
    }

    #[test]
    fn utf8_word_prefix_bridge_is_one_exact_learned_token_in_rich_and_flat_paths(
    ) -> Result<(), crate::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let raw = "x Россия".as_bytes().to_vec();
        let document = tokenizer.tokenize(raw.clone())?;
        let vocabulary = SurfaceVocabulary::from_ranked(vec![" Россия".as_bytes().to_vec()])?;
        let bridge_id = id(vocabulary.id_for_piece(" Россия".as_bytes()).expect("prefix piece"))?;

        let rich = vocabulary.encode_document(&document, false)?;
        assert_eq!(rich.lengths, vec![0, 1, 13, 0]);
        assert_eq!(rich.ids[2], bridge_id);
        assert_eq!(vocabulary.decode_ids(&rich.ids)?, raw);

        let flat = tokenizer.encode_surface_batch(&[raw.clone()], &[false], &vocabulary, 1, true)?;
        assert_eq!(flat.ids, rich.ids);
        assert_eq!(flat.lengths, rich.lengths);
        assert_eq!(flat.document_offsets, vec![0, 4]);
        Ok(())
    }

    #[test]
    fn multi_space_runs_stay_standalone_and_can_be_learned() -> Result<(), crate::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let document = tokenizer.tokenize(b"a        b".to_vec())?;
        let mut trainer = SurfaceVocabularyTrainer::new(4096)?;
        trainer.observe(&document)?;
        let vocabulary = trainer.finish(512)?;
        assert!(vocabulary.id_for_piece(b"        ").is_some());
        assert!(vocabulary.id_for_piece(b"        b").is_none());
        assert_eq!(vocabulary.decode_ids(&vocabulary.encode_document(&document, false)?.ids)?, b"a        b");
        Ok(())
    }

    #[test]
    fn trainer_learns_frequent_non_ascii_unicode_scalars() -> Result<(), crate::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let document = tokenizer.tokenize("Россия Россия 中文 中文".as_bytes().to_vec())?;
        let mut trainer = SurfaceVocabularyTrainer::new(4096)?;
        trainer.observe(&document)?;
        let vocabulary = trainer.finish(512)?;
        for scalar in ["Р", "о", "с", "и", "я", "中", "文"] {
            assert!(
                vocabulary.id_for_piece(scalar.as_bytes()).is_some(),
                "missing learned scalar {scalar}"
            );
        }
        assert_eq!(
            vocabulary.decode_ids(&vocabulary.encode_document(&document, false)?.ids)?,
            document.raw()
        );
        Ok(())
    }

    #[test]
    fn valid_foreign_utf8_has_no_unknown_token_and_byte_fallback_is_exact(
    ) -> Result<(), crate::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let raw = "Россия 中文".as_bytes().to_vec();
        let document = tokenizer.tokenize(raw.clone())?;
        let vocabulary = SurfaceVocabulary::from_ranked(Vec::new())?;
        let encoded = vocabulary.encode_document(&document, false)?;
        for token in &encoded.ids[1..encoded.ids.len().saturating_sub(1)] {
            let token = u32::from(*token);
            assert!((SURFACE_BYTE_BASE_ID..super::SURFACE_ENTRY_BASE_ID).contains(&token));
        }
        assert_eq!(vocabulary.decode_ids(&encoded.ids)?, raw);
        Ok(())
    }

    #[test]
    fn byte_bpe_uses_merge_rank_and_round_trips() -> Result<(), crate::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let document = tokenizer.tokenize(b"abcd".to_vec())?;
        let vocabulary = SurfaceVocabulary::from_bpe_ranked(vec![
            b"ab".to_vec(),
            b"cd".to_vec(),
            b"abcd".to_vec(),
        ])?;
        assert_eq!(vocabulary.kind(), super::SurfaceVocabularyKind::ByteBpe);
        let encoded = vocabulary.encode_document(&document, false)?;
        assert_eq!(encoded.lengths, vec![0, 4, 0]);
        assert_eq!(vocabulary.decode_ids(&encoded.ids)?, b"abcd");
        let reloaded = SurfaceVocabulary::from_bytes(&vocabulary.to_bytes()?)?;
        assert_eq!(reloaded, vocabulary);
        assert_eq!(reloaded.kind(), super::SurfaceVocabularyKind::ByteBpe);
        Ok(())
    }

    #[test]
    fn byte_bpe_rejects_out_of_order_merges() {
        let error = SurfaceVocabulary::from_bpe_ranked(vec![
            b"abcd".to_vec(),
            b"ab".to_vec(),
            b"cd".to_vec(),
        ])
        .expect_err("a merge must be constructible from earlier ranks");
        assert!(matches!(error, crate::TokenizerError::InvalidVocabulary(_)));
    }

    #[test]
    fn byte_bpe_can_merge_space_into_following_word() -> Result<(), crate::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let document = tokenizer.tokenize(b"a hava".to_vec())?;
        let vocabulary = SurfaceVocabulary::from_bpe_ranked(vec![
            b" h".to_vec(),
            b" ha".to_vec(),
            b" hav".to_vec(),
            b" hava".to_vec(),
        ])?;
        let encoded = vocabulary.encode_document(&document, false)?;
        assert_eq!(vocabulary.decode_ids(&encoded.ids)?, b"a hava");
        assert!(encoded.lengths.iter().any(|length| *length == 5));
        Ok(())
    }

    #[test]
    fn lexical_byte_bpe_ignores_morphology_cuts_but_round_trips() -> Result<(), crate::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let document = tokenizer.tokenize(b"gidicem".to_vec())?;
        assert!(!document.units()[0].cuts.is_empty());
        let vocabulary = SurfaceVocabulary::from_lexical_bpe_ranked(vec![
            b"gi".to_vec(),
            b"gid".to_vec(),
            b"gidi".to_vec(),
            b"gidic".to_vec(),
            b"gidice".to_vec(),
            b"gidicem".to_vec(),
        ])?;
        let encoded = vocabulary.encode_document(&document, false)?;
        assert_eq!(encoded.lengths, vec![0, 7, 0]);
        assert_eq!(vocabulary.decode_ids(&encoded.ids)?, b"gidicem");
        let reloaded = SurfaceVocabulary::from_bytes(&vocabulary.to_bytes()?)?;
        assert_eq!(reloaded.kind(), super::SurfaceVocabularyKind::LexicalByteBpe);
        Ok(())
    }

    #[test]
    fn arbitrary_bytes_fall_back_exactly() -> Result<(), crate::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let document = tokenizer.tokenize(vec![0x00, 0xff])?;
        let vocabulary = SurfaceVocabulary::from_ranked(Vec::new())?;
        let encoded = vocabulary.encode_document(&document, false)?;
        assert_eq!(
            encoded.ids,
            vec![
                id(SURFACE_BOS_ID)?,
                id(SURFACE_BYTE_BASE_ID)?,
                id(SURFACE_BYTE_BASE_ID + 0xff)?,
                id(SURFACE_EOS_ID)?,
            ]
        );
        assert_eq!(encoded.lengths, vec![0, 1, 1, 0]);
        Ok(())
    }
}
