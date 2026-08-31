//! Self-contained, byte-exact Turkish morphological tokenizer.

#![forbid(unsafe_code)]

use core::fmt;
use std::{
    collections::HashMap,
    ops::Range,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread,
};

use nedo_core::{scan, scan_compact, CompactScanResult, LexicalKind, ScanError};
use nedo_format::{ByteSpan, FormatError, LosslessDocument, SurfaceUnit};
use nedo_morph_bundle::{
    ambiguity_word_data, AmbiguityScoringCode, AmbiguityWordData, BinaryError, DisambiguationError,
    DisambiguationScoreCache, NativeAnalysis, NativeDisambiguator, NativeMorpheme,
    NativeMorphology, SharedPreparedWordCache,
};
use sha2::{Digest, Sha256};

mod alignment;
mod code;
mod codec;
#[cfg(feature = "compiled-surface-table")]
mod compiled_analysis_table;
mod flat_surface;
mod nedoformer;
mod nedoformer_contract;
mod nedoformer_input;
pub(crate) mod nedoformer_sidecar;
mod nedoformer_vocab;
mod quality;
mod sharded_surface;
mod surface_vocab;
mod training;
mod vocab;

pub use codec::{decode as decode_tokenized, encode as encode_tokenized};
#[cfg(feature = "compiled-surface-table")]
pub use compiled_analysis_table::{
    encode_compiled_surface_analysis_table, CompiledSurfaceAnalysisEntry,
    CompiledSurfaceAnalysisTable, CompiledSurfaceAnalysisTableError, CompiledSurfaceCandidateEntry,
};
pub use nedo_format::{decode_binary as decode_lossless, encode_binary as encode_lossless};
pub use nedoformer::{
    NedoFormerLatticeDocument, NedoFormerLatticeUnit, NedoFormerSamplingPolicy,
    NedoFormerSegmentationCandidate,
};
pub use nedoformer_contract::{
    NedoFormerContractFingerprint, NEDOFORMER_CODE_SEGMENTATION_VERSION,
    NEDOFORMER_INPUT_ENCODING_VERSION, NEDOFORMER_LATTICE_SCHEMA_VERSION,
    NEDOFORMER_NUMERIC_SEGMENTATION_VERSION, NEDOFORMER_SHADOW_NORMALIZATION_VERSION,
    NEDOFORMER_TOKENIZER_CONTRACT_VERSION, NEDOFORMER_WHITESPACE_SCHEMA_VERSION,
};
pub use nedoformer_input::NedoFormerInputEncoding;
pub use nedoformer_sidecar::{
    NedoFormerLatticeSidecar, NedoFormerSidecarCandidate, NedoFormerSidecarUnit,
    NEDOFORMER_SIDECAR_SCHEMA_VERSION,
};
pub use nedoformer_vocab::{
    NedoFormerGenerationEncoding, NedoFormerVocabEntry, NedoFormerVocabKind, NedoFormerVocabulary,
};
pub use sharded_surface::{
    merge_surface_shards, ShardedSurfaceEncoder, ShardedSurfaceRuntimeCache, SurfaceShardBatch,
};
pub use surface_vocab::{
    surface_bpe_segments, SurfaceVocabulary, SurfaceVocabularyKind, SurfaceVocabularyTrainer,
    SURFACE_BOS_ID, SURFACE_BYTE_BASE_ID, SURFACE_ENTRY_BASE_ID, SURFACE_EOS_ID, SURFACE_PAD_ID,
};
pub use training::{
    encode_byte_training_document, TrainingBatch, TrainingEncoding, TrainingEncodingOptions,
};
pub use vocab::{
    CharacterVocabulary, GenerationVocabulary, BOS_ID, BYTE_BASE_ID, CHAR_BASE_ID, CODE_END_ID,
    CODE_START_ID, EOS_ID, MORPHEME_BOUNDARY_ID, PAD_ID, UNIT_BOUNDARY_ID,
};

use alignment::align_analysis;
use code::{auto_code_spans, explicit_code_spans, identifier_cuts};
use quality::{analyze_token_with_nedoformer_shadow, analyze_token_with_quality_fallback};

/// Stable tokenized-document schema version.
pub const TOKENIZER_SCHEMA_VERSION: u32 = 1;
/// Embedded native morphology SHA-256.
pub const MORPHOLOGY_SHA256: &str =
    "8b78ba8c6352e2abba8b7148da143896257606e916693f640649b2b57a0a23d6";
/// Embedded native disambiguation model SHA-256.
pub const MODEL_SHA256: &str = "1aaaaf5343fda967d5864dcae4b44540c25ba859c385c5f103aaabc988056ee0";

const EMBEDDED_MORPHOLOGY: &[u8] = include_bytes!("../../../assets/native/nedo-morph-v1.bin");
const EMBEDDED_MODEL: &[u8] = include_bytes!("../../../assets/native/model-compressed");
const MAX_SENTENCE_TOKENS: usize = 512;
const BATCH_ANALYSIS_CACHE_ENTRIES: usize = 65_536;
const LOW_PARALLEL_ANALYSIS_CACHE_BUDGET: usize = 262_144;

fn analysis_cache_entries_for_parallelism(parallelism: usize) -> usize {
    LOW_PARALLEL_ANALYSIS_CACHE_BUDGET
        .checked_div(parallelism.max(1))
        .unwrap_or(BATCH_ANALYSIS_CACHE_ENTRIES)
        .max(BATCH_ANALYSIS_CACHE_ENTRIES)
}
const MIN_BATCH_CHUNK_BYTES: usize = 1 << 20;
const BATCH_CHUNKS_PER_WORKER: usize = 16;
const SEGMENT_PROGRAM_CACHE_ENTRIES: usize = 32_768;
const SURFACE_PROGRAM_TABLE_SLOTS: usize = SEGMENT_PROGRAM_CACHE_ENTRIES * 2;
const DOCUMENT_PROGRAM_CACHE_ENTRIES: usize = 8_192;
const MAX_FALLBACK_CHARS_CONFIG: usize = 4_096;

/// Requested tokenizer mode.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum TokenizerMode {
    /// Detect explicit and high-confidence unmarked code spans.
    #[default]
    Auto = 0,
    /// Treat the complete input as Turkish/mixed natural language.
    Turkish = 1,
    /// Treat the complete input as code and bypass morphology.
    Code = 2,
}

/// Effective mode for one exact surface unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TokenMode {
    /// Native Turkish morphology path.
    Turkish = 1,
    /// Exact code/raw character path.
    Code = 2,
    /// Invalid UTF-8 or control payload path.
    Opaque = 3,
}

impl TryFrom<u8> for TokenMode {
    type Error = TokenizerError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Turkish),
            2 => Ok(Self::Code),
            3 => Ok(Self::Opaque),
            _ => Err(TokenizerError::InvalidCodecEnum("token mode", value)),
        }
    }
}

/// Analysis state for one unit; no fallback is disguised as morphology.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TokenStatus {
    /// Whitespace, line break, punctuation, symbol, or other structural content.
    Structural = 1,
    /// Native morphology and contextual selection succeeded.
    Morphological = 2,
    /// Explicit unknown analysis generated by the contextual model.
    Unknown = 3,
    /// Code-mode content bypassed morphology by configuration/detection.
    Code = 4,
    /// Invalid UTF-8/control content preserved exactly.
    Opaque = 5,
}

impl TryFrom<u8> for TokenStatus {
    type Error = TokenizerError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Structural),
            2 => Ok(Self::Morphological),
            3 => Ok(Self::Unknown),
            4 => Ok(Self::Code),
            5 => Ok(Self::Opaque),
            _ => Err(TokenizerError::InvalidCodecEnum("token status", value)),
        }
    }
}

/// Immutable tokenizer behavior configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenizerConfig {
    /// Input/code mode policy.
    pub mode: TokenizerMode,
    /// Maximum contextual sentence token count before a deterministic chunk boundary.
    pub max_sentence_tokens: usize,
    /// Maximum Unicode-scalar count for unknown/code fallback pieces.
    pub max_fallback_chars: usize,
    /// Whether sentence context selects among morphological candidates.
    pub contextual_disambiguation: bool,
    /// Whether `Auto` mode detects high-confidence unmarked code lines/blocks.
    pub detect_unmarked_code: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            mode: TokenizerMode::Auto,
            max_sentence_tokens: MAX_SENTENCE_TOKENS,
            max_fallback_chars: 48,
            contextual_disambiguation: true,
            detect_unmarked_code: true,
        }
    }
}

/// Exact contextual-score cache policy for the final surface encoder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContextCachePolicy {
    /// Largest cache: best retention, highest worker-local memory use.
    Full,
    /// Intermediate cache sized for high-shard exact preprocessing.
    Balanced,
    /// Smallest cache: minimum memory, more score recomputation after saturation.
    Compact,
}

/// Runtime policy for the final surface-piece encoder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SurfaceEncoderOptions {
    /// Whether Turkish morphology contributes internal surface boundaries.
    pub use_morphology: bool,
    /// Maximum exact whole-document programs retained per worker or shard.
    ///
    /// Set this to zero for a one-pass corpus build. Word, morphology, segment,
    /// and surface-program caches remain enabled; only exact document replay is
    /// disabled, avoiding a persistent copy of every new document and output.
    pub document_cache_entries: usize,
    /// Whether contextual trigram scores use the smaller exact shard-local table.
    ///
    /// When full, new scores are computed without retention; output is unchanged.
    pub context_cache_policy: ContextCachePolicy,
}

impl SurfaceEncoderOptions {
    /// Standard reusable-service policy with exact document replay enabled.
    #[must_use]
    pub const fn cached(use_morphology: bool) -> Self {
        Self {
            use_morphology,
            document_cache_entries: DOCUMENT_PROGRAM_CACHE_ENTRIES,
            context_cache_policy: ContextCachePolicy::Full,
        }
    }

    /// High-throughput one-pass corpus policy that does not retain complete documents.
    #[must_use]
    pub const fn one_pass(use_morphology: bool) -> Self {
        Self {
            use_morphology,
            document_cache_entries: 0,
            context_cache_policy: ContextCachePolicy::Full,
        }
    }

    /// Balanced one-pass policy for high-shard preprocessing.
    #[must_use]
    pub const fn one_pass_balanced(use_morphology: bool) -> Self {
        Self {
            use_morphology,
            document_cache_entries: 0,
            context_cache_policy: ContextCachePolicy::Balanced,
        }
    }

    /// Lower-memory one-pass policy with a compact exact contextual-score cache.
    ///
    /// This can improve eight-shard throughput and memory use, but may reduce
    /// throughput at higher shard counts. Capacity exhaustion never changes output.
    #[must_use]
    pub const fn one_pass_compact(use_morphology: bool) -> Self {
        Self {
            use_morphology,
            document_cache_entries: 0,
            context_cache_policy: ContextCachePolicy::Compact,
        }
    }
}

impl Default for SurfaceEncoderOptions {
    fn default() -> Self {
        Self::cached(true)
    }
}

/// One aligned morpheme over original bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AlignedMorpheme {
    /// Stable morpheme ID.
    pub id: String,
    /// Normalized realized surface.
    pub surface: String,
    /// Original-byte interval consumed by the surface; epsilon spans are empty.
    pub span: ByteSpan,
    /// Whether this morpheme starts a derivational group.
    pub derivational: bool,
}

/// Selected analysis metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnalysisMetadata {
    /// Canonical reference key.
    pub canonical: String,
    /// Stable dictionary/root ID.
    pub dictionary_id: String,
    /// Perceptron lemma.
    pub lemma: String,
    /// Primary POS short name.
    pub primary_pos: String,
    /// Secondary POS short name.
    pub secondary_pos: String,
    /// Ordered aligned morphemes.
    pub morphemes: Vec<AlignedMorpheme>,
}

/// One exact surface unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizedUnit {
    /// Full original-byte span.
    pub span: ByteSpan,
    /// Scanner class.
    pub kind: LexicalKind,
    /// Effective processing mode.
    pub mode: TokenMode,
    /// Explicit analysis state.
    pub status: TokenStatus,
    /// Inner-model group ID; clitic groups may span whitespace units.
    pub group_id: Option<u32>,
    /// Strictly increasing byte cuts inside `span`.
    pub cuts: Vec<u64>,
    /// Selected analysis when applicable.
    pub analysis: Option<AnalysisMetadata>,
}

/// Complete byte-exact tokenized document.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizedDocument {
    raw: Vec<u8>,
    units: Vec<TokenizedUnit>,
}

impl TokenizedDocument {
    /// Creates and validates a tokenized document.
    ///
    /// # Errors
    ///
    /// Returns an error if coverage, cuts, status, or analysis spans are invalid.
    pub fn new(raw: Vec<u8>, units: Vec<TokenizedUnit>) -> Result<Self, TokenizerError> {
        let document = Self { raw, units };
        document.validate()?;
        Ok(document)
    }

    /// Exact original bytes.
    #[must_use]
    pub fn raw(&self) -> &[u8] {
        &self.raw
    }

    /// Exact metadata units.
    #[must_use]
    pub fn units(&self) -> &[TokenizedUnit] {
        &self.units
    }

    /// Byte-exact decode.
    #[must_use]
    pub fn decode(&self) -> &[u8] {
        &self.raw
    }

    /// Converts to the lower-level lossless span/cut document.
    ///
    /// # Errors
    ///
    /// Returns an error if stored unit spans or cuts violate the lower-level format.
    pub fn lossless_document(&self) -> Result<LosslessDocument, TokenizerError> {
        let units = self
            .units
            .iter()
            .map(|unit| SurfaceUnit::new(unit.span, unit.cuts.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(LosslessDocument::new(self.raw.clone(), units)?)
    }

    /// Validates exact coverage, cuts, analysis spans, and status invariants.
    ///
    /// # Errors
    ///
    /// Returns an error for any coverage, metadata, status, or span inconsistency.
    pub fn validate(&self) -> Result<(), TokenizerError> {
        let mut expected = 0_u64;
        let document_len = u64::try_from(self.raw.len())
            .map_err(|_| TokenizerError::LengthOverflow("document length"))?;
        for (index, unit) in self.units.iter().enumerate() {
            if unit.span.start != expected || unit.span.end <= unit.span.start {
                return Err(TokenizerError::InvalidUnitCoverage {
                    index,
                    expected,
                    start: unit.span.start,
                    end: unit.span.end,
                });
            }
            if unit.span.end > document_len {
                return Err(TokenizerError::InvalidUnitCoverage {
                    index,
                    expected,
                    start: unit.span.start,
                    end: unit.span.end,
                });
            }
            let surface = SurfaceUnit::new(unit.span, unit.cuts.clone())?;
            surface.validate()?;
            validate_unit_metadata(index, unit)?;
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
}

fn validate_unit_metadata(index: usize, unit: &TokenizedUnit) -> Result<(), TokenizerError> {
    let mode_status_valid = matches!(
        (unit.mode, unit.status),
        (
            TokenMode::Turkish,
            TokenStatus::Structural | TokenStatus::Morphological | TokenStatus::Unknown
        ) | (TokenMode::Code, TokenStatus::Code)
            | (TokenMode::Opaque, TokenStatus::Opaque)
    );
    if !mode_status_valid {
        return Err(TokenizerError::InvalidUnitMetadata {
            index,
            reason: "mode and status are inconsistent",
        });
    }
    if matches!(unit.kind, LexicalKind::Opaque | LexicalKind::Control)
        != (unit.mode == TokenMode::Opaque)
    {
        return Err(TokenizerError::InvalidUnitMetadata {
            index,
            reason: "opaque/control kind and mode are inconsistent",
        });
    }
    match unit.status {
        TokenStatus::Morphological | TokenStatus::Unknown => {
            if unit.analysis.is_none() {
                return Err(TokenizerError::MissingAnalysis { index });
            }
        }
        TokenStatus::Structural | TokenStatus::Code | TokenStatus::Opaque => {
            if unit.analysis.is_some() {
                return Err(TokenizerError::UnexpectedAnalysis { index });
            }
        }
    }
    if let Some(analysis) = &unit.analysis {
        validate_analysis_metadata(index, unit, analysis)?;
    }
    Ok(())
}

fn validate_analysis_metadata(
    index: usize,
    unit: &TokenizedUnit,
    analysis: &AnalysisMetadata,
) -> Result<(), TokenizerError> {
    if analysis.canonical.is_empty()
        || analysis.dictionary_id.is_empty()
        || analysis.primary_pos.is_empty()
        || analysis.morphemes.is_empty()
    {
        return Err(TokenizerError::InvalidUnitMetadata {
            index,
            reason: "analysis identity is incomplete",
        });
    }
    let is_unknown = analysis.dictionary_id == "UNK_Unk_Unk";
    if (unit.status == TokenStatus::Unknown) != is_unknown {
        return Err(TokenizerError::InvalidUnitMetadata {
            index,
            reason: "unknown status and dictionary identity differ",
        });
    }
    let mut previous_end = unit.span.start;
    for morpheme in &analysis.morphemes {
        if morpheme.span.start < unit.span.start
            || morpheme.span.end > unit.span.end
            || morpheme.span.start > morpheme.span.end
        {
            return Err(TokenizerError::MorphemeOutsideUnit { index });
        }
        if morpheme.span.start < previous_end {
            return Err(TokenizerError::InvalidUnitMetadata {
                index,
                reason: "morpheme spans overlap or are out of order",
            });
        }
        if morpheme.surface.is_empty() != morpheme.span.is_empty() {
            return Err(TokenizerError::InvalidUnitMetadata {
                index,
                reason: "morpheme surface and span emptiness differ",
            });
        }
        previous_end = previous_end.max(morpheme.span.end);
    }
    Ok(())
}

/// Self-contained tokenizer using validated native assets.
pub struct Tokenizer<'a> {
    morphology: NativeMorphology<'a>,
    disambiguator: NativeDisambiguator,
    config: TokenizerConfig,
    #[cfg(feature = "compiled-surface-table")]
    compiled_surface_analysis_table: Option<CompiledSurfaceAnalysisTable>,
    #[cfg(feature = "compiled-surface-table")]
    nedoformer_compiled_surface_analysis_table: Option<CompiledSurfaceAnalysisTable>,
}

#[derive(Debug)]
struct RichAnalysisSet {
    analyses: Box<[NativeAnalysis]>,
    ambiguity: Box<[AmbiguityWordData]>,
}

struct AnalysisCache {
    entries: HashMap<String, Arc<RichAnalysisSet>>,
    capacity: usize,
}

impl AnalysisCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity.min(4_096)),
            capacity,
        }
    }

    fn analyze(
        &mut self,
        morphology: &NativeMorphology<'_>,
        token: &str,
    ) -> Result<Arc<RichAnalysisSet>, TokenizerError> {
        if let Some(analyses) = self.entries.get(token) {
            return Ok(Arc::clone(analyses));
        }
        let analyses = surface_valid_candidates(
            token,
            0,
            analyze_token_with_quality_fallback(morphology, token)?,
        );
        let ambiguity = analyses.iter().map(ambiguity_word_data).collect::<Vec<_>>();
        let set = Arc::new(RichAnalysisSet {
            analyses: analyses.into_boxed_slice(),
            ambiguity: ambiguity.into_boxed_slice(),
        });
        if self.entries.len() < self.capacity {
            self.entries.insert(token.to_owned(), Arc::clone(&set));
        }
        Ok(set)
    }

    fn analyze_nedoformer(
        &mut self,
        morphology: &NativeMorphology<'_>,
        token: &str,
    ) -> Result<Arc<RichAnalysisSet>, TokenizerError> {
        if let Some(cached) = self.entries.get(token) {
            return Ok(Arc::clone(cached));
        }
        let analyses = surface_valid_candidates(
            token,
            0,
            analyze_token_with_nedoformer_shadow(morphology, token)?,
        );
        let ambiguity = analyses.iter().map(ambiguity_word_data).collect::<Vec<_>>();
        let set = Arc::new(RichAnalysisSet {
            analyses: analyses.into_boxed_slice(),
            ambiguity: ambiguity.into_boxed_slice(),
        });
        if self.entries.len() < self.capacity {
            self.entries.insert(token.to_owned(), Arc::clone(&set));
        }
        Ok(set)
    }
}

#[derive(Debug)]
struct FlatAnalysisSet {
    relative_cuts: Box<[Vec<u32>]>,
    ambiguity: Box<[AmbiguityWordData]>,
    scoring_codes: Box<[AmbiguityScoringCode]>,
    unknown: Box<[bool]>,
    output_invariant: bool,
}

#[cfg(feature = "compiled-surface-table")]
enum FlatAnalysisSource<'a> {
    Compiled(&'a FlatAnalysisSet),
    Live(Arc<FlatAnalysisSet>),
}

#[cfg(feature = "compiled-surface-table")]
impl FlatAnalysisSource<'_> {
    fn set(&self) -> &FlatAnalysisSet {
        match self {
            Self::Compiled(set) => set,
            Self::Live(set) => set.as_ref(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct FlatSegmentTokenProgram {
    cut_start: u32,
    cut_len: u16,
    unknown: bool,
}

#[derive(Debug)]
pub(crate) struct FlatSegmentProgram {
    tokens: Box<[FlatSegmentTokenProgram]>,
    relative_cuts: Box<[u32]>,
    pub(crate) surface_ids: Box<[u16]>,
    pub(crate) surface_lengths: Box<[u8]>,
}

#[derive(Debug)]
struct FlatSurfaceProgramEntry {
    fingerprint: u64,
    kind: u8,
    exact: Box<[u8]>,
    program: Arc<FlatSegmentProgram>,
}

struct FlatSurfaceProgramTable {
    slots: Box<[Option<FlatSurfaceProgramEntry>]>,
}

impl FlatSurfaceProgramTable {
    fn new() -> Self {
        Self {
            slots: std::iter::repeat_with(|| None)
                .take(SURFACE_PROGRAM_TABLE_SLOTS)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    fn clear(&mut self) {
        self.slots.fill_with(|| None);
    }
}

struct FlatDocumentProgramEntry {
    fingerprint: u64,
    newline: bool,
    exact: Box<[u8]>,
    ids: Box<[u16]>,
    lengths: Box<[u8]>,
}

struct FlatDocumentProgramTable {
    slots: Box<[Option<FlatDocumentProgramEntry>]>,
}

impl FlatDocumentProgramTable {
    fn new(capacity: usize) -> Self {
        let slots = if capacity == 0 {
            0
        } else {
            capacity
                .saturating_mul(2)
                .max(8)
                .checked_next_power_of_two()
                .unwrap_or(usize::MAX / 2 + 1)
        };
        Self {
            slots: std::iter::repeat_with(|| None)
                .take(slots)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    fn clear(&mut self) {
        self.slots.fill_with(|| None);
    }
}

/// Aggregate telemetry for worker-local flat morphology caches.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TrainingCacheStats {
    /// Successful surface lookups.
    pub hits: u64,
    /// Morphology calls caused by absent surfaces.
    pub misses: u64,
    /// Misses that could not be retained because a worker cache was full.
    pub saturated_misses: u64,
    /// Currently retained unique surfaces across workers.
    pub entries: u64,
    /// Approximate owned heap bytes for keys and compact candidate templates.
    pub approximate_bytes: u64,
}

impl TrainingCacheStats {
    const fn merge(&mut self, other: Self) {
        self.hits = self.hits.saturating_add(other.hits);
        self.misses = self.misses.saturating_add(other.misses);
        self.saturated_misses = self.saturated_misses.saturating_add(other.saturated_misses);
        self.entries = self.entries.saturating_add(other.entries);
        self.approximate_bytes = self
            .approximate_bytes
            .saturating_add(other.approximate_bytes);
    }
}

#[derive(Clone, Debug, Default)]
struct AmbiguityScoringInterner {
    canonicals: HashMap<String, u32>,
    lemmas: HashMap<String, u32>,
    igs: HashMap<String, u32>,
    ig_sequences: HashMap<Vec<u32>, u32>,
    signatures: HashMap<(u32, u32), u32>,
}

impl AmbiguityScoringInterner {
    fn code(&mut self, word: &AmbiguityWordData) -> Result<AmbiguityScoringCode, TokenizerError> {
        let canonical = intern_scoring_string(
            &mut self.canonicals,
            &word.canonical,
            "canonical-analysis identity",
        )?;
        let lemma = intern_scoring_string(&mut self.lemmas, &word.lemma, "lemma identity")?;
        let mut ig_ids = Vec::with_capacity(word.igs.len());
        for ig in &word.igs {
            ig_ids.push(intern_scoring_string(
                &mut self.igs,
                ig,
                "inflectional-group identity",
            )?);
        }
        let next_sequence = u32::try_from(self.ig_sequences.len())
            .map_err(|_| TokenizerError::LengthOverflow("IG-sequence identity"))?;
        let ig_sequence = *self.ig_sequences.entry(ig_ids).or_insert(next_sequence);
        let next_signature = u32::try_from(self.signatures.len())
            .map_err(|_| TokenizerError::LengthOverflow("ambiguity scoring signature"))?;
        let signature = *self
            .signatures
            .entry((lemma, ig_sequence))
            .or_insert(next_signature);
        Ok(AmbiguityScoringCode::new(
            signature,
            canonical,
            word.java_hash,
        ))
    }

    fn signature_count(&self) -> usize {
        self.signatures.len()
    }

    fn code_with_base(
        &mut self,
        base: &Self,
        word: &AmbiguityWordData,
    ) -> Result<AmbiguityScoringCode, TokenizerError> {
        let canonical = intern_scoring_string_layered(
            &base.canonicals,
            &mut self.canonicals,
            &word.canonical,
            "canonical-analysis identity",
        )?;
        let lemma = intern_scoring_string_layered(
            &base.lemmas,
            &mut self.lemmas,
            &word.lemma,
            "lemma identity",
        )?;
        let mut ig_ids = Vec::with_capacity(word.igs.len());
        for ig in &word.igs {
            ig_ids.push(intern_scoring_string_layered(
                &base.igs,
                &mut self.igs,
                ig,
                "inflectional-group identity",
            )?);
        }
        let ig_sequence = if let Some(&id) = base.ig_sequences.get(&ig_ids) {
            id
        } else if let Some(&id) = self.ig_sequences.get(&ig_ids) {
            id
        } else {
            let next = base
                .ig_sequences
                .len()
                .checked_add(self.ig_sequences.len())
                .ok_or(TokenizerError::LengthOverflow("IG-sequence identity"))?;
            let id = u32::try_from(next)
                .map_err(|_| TokenizerError::LengthOverflow("IG-sequence identity"))?;
            self.ig_sequences.insert(ig_ids, id);
            id
        };
        let signature_key = (lemma, ig_sequence);
        let signature = if let Some(&id) = base.signatures.get(&signature_key) {
            id
        } else if let Some(&id) = self.signatures.get(&signature_key) {
            id
        } else {
            let next = base
                .signatures
                .len()
                .checked_add(self.signatures.len())
                .ok_or(TokenizerError::LengthOverflow(
                    "ambiguity scoring signature",
                ))?;
            let id = u32::try_from(next)
                .map_err(|_| TokenizerError::LengthOverflow("ambiguity scoring signature"))?;
            self.signatures.insert(signature_key, id);
            id
        };
        Ok(AmbiguityScoringCode::new(
            signature,
            canonical,
            word.java_hash,
        ))
    }

    fn approximate_bytes(&self) -> u64 {
        let mut bytes = self
            .canonicals
            .capacity()
            .saturating_mul(std::mem::size_of::<(String, u32)>())
            .saturating_add(
                self.lemmas
                    .capacity()
                    .saturating_mul(std::mem::size_of::<(String, u32)>()),
            )
            .saturating_add(
                self.igs
                    .capacity()
                    .saturating_mul(std::mem::size_of::<(String, u32)>()),
            )
            .saturating_add(
                self.ig_sequences
                    .capacity()
                    .saturating_mul(std::mem::size_of::<(Vec<u32>, u32)>()),
            )
            .saturating_add(
                self.signatures
                    .capacity()
                    .saturating_mul(std::mem::size_of::<((u32, u32), u32)>()),
            );
        for value in self
            .canonicals
            .keys()
            .chain(self.lemmas.keys())
            .chain(self.igs.keys())
        {
            bytes = bytes.saturating_add(value.capacity());
        }
        for value in self.ig_sequences.keys() {
            bytes =
                bytes.saturating_add(value.capacity().saturating_mul(std::mem::size_of::<u32>()));
        }
        u64::try_from(bytes).unwrap_or(u64::MAX)
    }
}

fn intern_scoring_string(
    values: &mut HashMap<String, u32>,
    value: &str,
    field: &'static str,
) -> Result<u32, TokenizerError> {
    if let Some(&id) = values.get(value) {
        return Ok(id);
    }
    let id = u32::try_from(values.len()).map_err(|_| TokenizerError::LengthOverflow(field))?;
    values.insert(value.to_owned(), id);
    Ok(id)
}

fn intern_scoring_string_layered(
    base: &HashMap<String, u32>,
    delta: &mut HashMap<String, u32>,
    value: &str,
    field: &'static str,
) -> Result<u32, TokenizerError> {
    if let Some(&id) = base.get(value) {
        return Ok(id);
    }
    if let Some(&id) = delta.get(value) {
        return Ok(id);
    }
    let next = base
        .len()
        .checked_add(delta.len())
        .ok_or(TokenizerError::LengthOverflow(field))?;
    let id = u32::try_from(next).map_err(|_| TokenizerError::LengthOverflow(field))?;
    delta.insert(value.to_owned(), id);
    Ok(id)
}

const SHORT_ANALYSIS_KEY_BYTES: usize = 15;

struct ShortAnalysisCache {
    keys: Vec<u128>,
    values: Vec<Option<Arc<FlatAnalysisSet>>>,
    slots: usize,
}

impl ShortAnalysisCache {
    fn new(capacity: usize) -> Self {
        let slots = capacity
            .saturating_mul(2)
            .max(8)
            .checked_next_power_of_two()
            .unwrap_or(usize::MAX / 2 + 1);
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            slots,
        }
    }

    #[inline(always)]
    fn get(&self, key: u128) -> Option<&Arc<FlatAnalysisSet>> {
        if self.keys.is_empty() {
            return None;
        }
        let modulo = self.keys.len() - 1;
        let mut slot = short_analysis_hash(key) & modulo;
        loop {
            let stored = self.keys[slot];
            if stored == key {
                return self.values[slot].as_ref();
            }
            if stored == 0 {
                return None;
            }
            slot = (slot + 1) & modulo;
        }
    }

    fn insert(&mut self, key: u128, value: Arc<FlatAnalysisSet>) {
        if self.keys.is_empty() {
            self.keys = vec![0; self.slots];
            self.values = vec![None; self.slots];
        }
        let modulo = self.keys.len() - 1;
        let mut slot = short_analysis_hash(key) & modulo;
        loop {
            if self.keys[slot] == 0 || self.keys[slot] == key {
                self.keys[slot] = key;
                self.values[slot] = Some(value);
                return;
            }
            slot = (slot + 1) & modulo;
        }
    }

    fn clear(&mut self) {
        self.keys = Vec::new();
        self.values = Vec::new();
    }

    fn approximate_bytes(&self) -> u64 {
        let bytes = self
            .keys
            .capacity()
            .saturating_mul(std::mem::size_of::<u128>())
            .saturating_add(
                self.values
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Option<Arc<FlatAnalysisSet>>>()),
            );
        u64::try_from(bytes).unwrap_or(u64::MAX)
    }
}

#[inline(always)]
fn pack_short_analysis_key(token: &str) -> Option<u128> {
    let bytes = token.as_bytes();
    if bytes.is_empty() || bytes.len() > SHORT_ANALYSIS_KEY_BYTES {
        return None;
    }
    let mut key = (bytes.len() as u128) << 120;
    for (index, byte) in bytes.iter().enumerate() {
        key |= u128::from(*byte) << (index * 8);
    }
    Some(key)
}

#[inline(always)]
fn short_analysis_hash(key: u128) -> usize {
    let mut value = (key as u64) ^ ((key >> 64) as u64).rotate_left(23);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    usize::try_from(value).unwrap_or(0)
}

struct FlatAnalysisCache {
    entries: HashMap<String, Arc<FlatAnalysisSet>>,
    short_entries: ShortAnalysisCache,
    capacity: usize,
    entry_count: usize,
    hits: u64,
    misses: u64,
    saturated_misses: u64,
    approximate_bytes: u64,
    scoring_interner: AmbiguityScoringInterner,
    #[cfg(feature = "compiled-surface-table")]
    scoring_interner_seed: Option<Arc<AmbiguityScoringInterner>>,
    disambiguation_scores: DisambiguationScoreCache,
    segment_programs: HashMap<Vec<u8>, Arc<FlatSegmentProgram>>,
    segment_key: Vec<u8>,
    surface_programs: FlatSurfaceProgramTable,
    surface_program_entries: usize,
    surface_program_bytes: u64,
    document_programs: FlatDocumentProgramTable,
    document_program_capacity: usize,
    document_program_entries: usize,
    document_program_hits: u64,
    document_program_misses: u64,
    document_program_bytes: u64,
    surface_program_context: Option<(usize, bool)>,
    segment_program_hits: u64,
    segment_program_misses: u64,
    segment_program_bytes: u64,
    phase_scan_ns: u128,
    phase_code_ns: u128,
    phase_split_ns: u128,
    phase_analysis_ns: u128,
    phase_long_ns: u128,
    phase_vocab_ns: u128,
    phase_documents: u64,
}

fn estimate_flat_entry_bytes(token: &str, set: &FlatAnalysisSet) -> u64 {
    let mut bytes = token.len();
    for cuts in &set.relative_cuts {
        bytes = bytes.saturating_add(cuts.len().saturating_mul(std::mem::size_of::<u32>()));
    }
    for ambiguity in &set.ambiguity {
        bytes = bytes
            .saturating_add(ambiguity.canonical.len())
            .saturating_add(ambiguity.lemma.len());
        for group in &ambiguity.igs {
            bytes = bytes.saturating_add(group.len());
        }
    }
    bytes = bytes
        .saturating_add(
            set.scoring_codes
                .len()
                .saturating_mul(std::mem::size_of::<AmbiguityScoringCode>()),
        )
        .saturating_add(set.unknown.len());
    u64::try_from(bytes).map_or(u64::MAX, |value| value)
}

impl FlatAnalysisCache {
    fn new(
        capacity: usize,
        document_program_capacity: usize,
        context_cache_policy: ContextCachePolicy,
    ) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity.min(4_096)),
            short_entries: ShortAnalysisCache::new(capacity),
            capacity,
            entry_count: 0,
            hits: 0,
            misses: 0,
            saturated_misses: 0,
            approximate_bytes: 0,
            scoring_interner: AmbiguityScoringInterner::default(),
            #[cfg(feature = "compiled-surface-table")]
            scoring_interner_seed: None,
            disambiguation_scores: match context_cache_policy {
                ContextCachePolicy::Full => DisambiguationScoreCache::default(),
                ContextCachePolicy::Balanced => DisambiguationScoreCache::balanced(),
                ContextCachePolicy::Compact => DisambiguationScoreCache::compact(),
            },
            segment_programs: HashMap::with_capacity(4_096),
            segment_key: Vec::with_capacity(512),
            surface_programs: FlatSurfaceProgramTable::new(),
            surface_program_entries: 0,
            surface_program_bytes: 0,
            document_programs: FlatDocumentProgramTable::new(document_program_capacity),
            document_program_capacity,
            document_program_entries: 0,
            document_program_hits: 0,
            document_program_misses: 0,
            document_program_bytes: 0,
            surface_program_context: None,
            segment_program_hits: 0,
            segment_program_misses: 0,
            segment_program_bytes: 0,
            phase_scan_ns: 0,
            phase_code_ns: 0,
            phase_split_ns: 0,
            phase_analysis_ns: 0,
            phase_long_ns: 0,
            phase_vocab_ns: 0,
            phase_documents: 0,
        }
    }

    #[cfg(feature = "compiled-surface-table")]
    fn new_seeded(
        capacity: usize,
        document_program_capacity: usize,
        context_cache_policy: ContextCachePolicy,
        scoring_interner_seed: Arc<AmbiguityScoringInterner>,
        shared_prepared_words: Option<Arc<SharedPreparedWordCache>>,
    ) -> Self {
        let mut disambiguation_scores = match context_cache_policy {
            ContextCachePolicy::Full => DisambiguationScoreCache::default(),
            ContextCachePolicy::Balanced => DisambiguationScoreCache::balanced(),
            ContextCachePolicy::Compact => DisambiguationScoreCache::compact(),
        };
        if let Some(shared_prepared_words) = shared_prepared_words {
            disambiguation_scores.set_shared_prepared_words(shared_prepared_words);
        }
        Self {
            entries: HashMap::with_capacity(capacity.min(4_096)),
            short_entries: ShortAnalysisCache::new(capacity),
            capacity,
            entry_count: 0,
            hits: 0,
            misses: 0,
            saturated_misses: 0,
            approximate_bytes: 0,
            scoring_interner: AmbiguityScoringInterner::default(),
            scoring_interner_seed: Some(scoring_interner_seed),
            disambiguation_scores,
            segment_programs: HashMap::with_capacity(4_096),
            segment_key: Vec::with_capacity(512),
            surface_programs: FlatSurfaceProgramTable::new(),
            surface_program_entries: 0,
            surface_program_bytes: 0,
            document_programs: FlatDocumentProgramTable::new(document_program_capacity),
            document_program_capacity,
            document_program_entries: 0,
            document_program_hits: 0,
            document_program_misses: 0,
            document_program_bytes: 0,
            surface_program_context: None,
            segment_program_hits: 0,
            segment_program_misses: 0,
            segment_program_bytes: 0,
            phase_scan_ns: 0,
            phase_code_ns: 0,
            phase_split_ns: 0,
            phase_analysis_ns: 0,
            phase_long_ns: 0,
            phase_vocab_ns: 0,
            phase_documents: 0,
        }
    }

    fn scoring_code(
        &mut self,
        word: &AmbiguityWordData,
    ) -> Result<AmbiguityScoringCode, TokenizerError> {
        #[cfg(feature = "compiled-surface-table")]
        if let Some(base) = self.scoring_interner_seed.as_deref() {
            return self.scoring_interner.code_with_base(base, word);
        }
        self.scoring_interner.code(word)
    }

    fn scoring_interner_counts(&self) -> (usize, usize, usize, usize) {
        #[cfg(feature = "compiled-surface-table")]
        let base = self.scoring_interner_seed.as_deref();
        #[cfg(not(feature = "compiled-surface-table"))]
        let base: Option<&AmbiguityScoringInterner> = None;
        let base_signatures = base.map_or(0, |value| value.signatures.len());
        let base_lemmas = base.map_or(0, |value| value.lemmas.len());
        let base_igs = base.map_or(0, |value| value.igs.len());
        let base_sequences = base.map_or(0, |value| value.ig_sequences.len());
        (
            base_signatures.saturating_add(self.scoring_interner.signatures.len()),
            base_lemmas.saturating_add(self.scoring_interner.lemmas.len()),
            base_igs.saturating_add(self.scoring_interner.igs.len()),
            base_sequences.saturating_add(self.scoring_interner.ig_sequences.len()),
        )
    }

    fn analyze(
        &mut self,
        morphology: &NativeMorphology<'_>,
        token: &str,
    ) -> Result<Arc<FlatAnalysisSet>, TokenizerError> {
        let short_key = pack_short_analysis_key(token);
        if let Some(key) = short_key {
            if let Some(analyses) = self.short_entries.get(key) {
                self.hits = self.hits.saturating_add(1);
                return Ok(Arc::clone(analyses));
            }
        } else if let Some(analyses) = self.entries.get(token) {
            self.hits = self.hits.saturating_add(1);
            return Ok(Arc::clone(analyses));
        }
        self.misses = self.misses.saturating_add(1);
        let analyses = analyze_token_with_quality_fallback(morphology, token)?;
        let mut compact = build_flat_analysis_set(token, analyses)?;
        compact.scoring_codes = compact
            .ambiguity
            .iter()
            .map(|word| self.scoring_code(word))
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice();
        let compact = Arc::new(compact);
        if self.entry_count < self.capacity {
            self.approximate_bytes = self
                .approximate_bytes
                .saturating_add(estimate_flat_entry_bytes(token, &compact));
            if let Some(key) = short_key {
                self.short_entries.insert(key, Arc::clone(&compact));
            } else {
                self.entries.insert(token.to_owned(), Arc::clone(&compact));
            }
            self.entry_count += 1;
        } else {
            self.saturated_misses = self.saturated_misses.saturating_add(1);
        }
        Ok(compact)
    }

    fn analyze_nedoformer(
        &mut self,
        morphology: &NativeMorphology<'_>,
        token: &str,
    ) -> Result<Arc<FlatAnalysisSet>, TokenizerError> {
        let short_key = pack_short_analysis_key(token);
        if let Some(key) = short_key {
            if let Some(analyses) = self.short_entries.get(key) {
                self.hits = self.hits.saturating_add(1);
                return Ok(Arc::clone(analyses));
            }
        } else if let Some(analyses) = self.entries.get(token) {
            self.hits = self.hits.saturating_add(1);
            return Ok(Arc::clone(analyses));
        }
        self.misses = self.misses.saturating_add(1);
        let analyses = analyze_token_with_nedoformer_shadow(morphology, token)?;
        let mut compact = build_flat_analysis_set(token, analyses)?;
        compact.scoring_codes = compact
            .ambiguity
            .iter()
            .map(|word| self.scoring_code(word))
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice();
        let compact = Arc::new(compact);
        if self.entry_count < self.capacity {
            self.approximate_bytes = self
                .approximate_bytes
                .saturating_add(estimate_flat_entry_bytes(token, &compact));
            if let Some(key) = short_key {
                self.short_entries.insert(key, Arc::clone(&compact));
            } else {
                self.entries.insert(token.to_owned(), Arc::clone(&compact));
            }
            self.entry_count += 1;
        } else {
            self.saturated_misses = self.saturated_misses.saturating_add(1);
        }
        Ok(compact)
    }

    fn segment_program(
        &mut self,
        raw: &[u8],
        units: &[TokenizedUnit],
        indices: &[usize],
    ) -> Result<Option<Arc<FlatSegmentProgram>>, TokenizerError> {
        self.segment_key.clear();
        self.segment_key.extend_from_slice(
            &u32::try_from(indices.len())
                .map_err(|_| TokenizerError::LengthOverflow("segment program token count"))?
                .to_le_bytes(),
        );
        for index in indices {
            let unit = units
                .get(*index)
                .ok_or(TokenizerError::InvalidTrainingEncoding(
                    "segment program unit index is out of range",
                ))?;
            self.segment_key.push(unit.kind as u8);
            let bytes = unit_bytes(raw, unit.span)?;
            self.segment_key.extend_from_slice(
                &u32::try_from(bytes.len())
                    .map_err(|_| TokenizerError::LengthOverflow("segment program surface length"))?
                    .to_le_bytes(),
            );
            self.segment_key.extend_from_slice(bytes);
        }
        if let Some(program) = self.segment_programs.get(self.segment_key.as_slice()) {
            self.segment_program_hits = self.segment_program_hits.saturating_add(1);
            self.hits = self
                .hits
                .saturating_add(u64::try_from(indices.len()).unwrap_or(u64::MAX));
            return Ok(Some(Arc::clone(program)));
        }
        self.segment_program_misses = self.segment_program_misses.saturating_add(1);
        Ok(None)
    }

    fn insert_segment_program(&mut self, program: FlatSegmentProgram) {
        if self.segment_programs.len() >= SEGMENT_PROGRAM_CACHE_ENTRIES {
            return;
        }
        let bytes = self
            .segment_key
            .len()
            .saturating_add(
                program
                    .tokens
                    .len()
                    .saturating_mul(std::mem::size_of::<FlatSegmentTokenProgram>()),
            )
            .saturating_add(
                program
                    .relative_cuts
                    .len()
                    .saturating_mul(std::mem::size_of::<u32>()),
            )
            .saturating_add(
                program
                    .surface_ids
                    .len()
                    .saturating_mul(std::mem::size_of::<u16>()),
            )
            .saturating_add(program.surface_lengths.len());
        self.segment_program_bytes = self
            .segment_program_bytes
            .saturating_add(u64::try_from(bytes).unwrap_or(u64::MAX));
        self.segment_programs
            .insert(self.segment_key.clone(), Arc::new(program));
    }

    fn prepare_surface_program_context(
        &mut self,
        vocabulary: &SurfaceVocabulary,
        use_morphology: bool,
    ) {
        let context = (
            vocabulary as *const SurfaceVocabulary as usize,
            use_morphology,
        );
        if self.surface_program_context == Some(context) {
            return;
        }
        self.surface_programs.clear();
        self.surface_program_entries = 0;
        self.surface_program_bytes = 0;
        self.document_programs.clear();
        self.document_program_entries = 0;
        self.document_program_hits = 0;
        self.document_program_misses = 0;
        self.document_program_bytes = 0;
        self.surface_program_context = Some(context);
    }

    fn stats(&self) -> TrainingCacheStats {
        if std::env::var_os("NEDO_SEGMENT_PROGRAM_TELEMETRY").is_some() {
            eprintln!(
                "NEDO_SEGMENT_PROGRAM hits={} misses={} entries={} bytes={}",
                self.segment_program_hits,
                self.segment_program_misses,
                self.segment_programs
                    .len()
                    .saturating_add(self.surface_program_entries),
                self.segment_program_bytes
                    .saturating_add(self.surface_program_bytes)
                    .saturating_add(self.document_program_bytes),
            );
        }
        if std::env::var_os("NEDO_SURFACE_PHASE_TELEMETRY").is_some() {
            let total = self
                .phase_scan_ns
                .saturating_add(self.phase_code_ns)
                .saturating_add(self.phase_split_ns)
                .saturating_add(self.phase_analysis_ns)
                .saturating_add(self.phase_long_ns)
                .saturating_add(self.phase_vocab_ns);
            eprintln!(
                "NEDO_SURFACE_PHASE documents={} total_ns={} scan_ns={} code_ns={} split_ns={} analysis_ns={} long_ns={} vocab_ns={}",
                self.phase_documents,
                total,
                self.phase_scan_ns,
                self.phase_code_ns,
                self.phase_split_ns,
                self.phase_analysis_ns,
                self.phase_long_ns,
                self.phase_vocab_ns,
            );
        }
        if std::env::var_os("NEDO_TRIGRAM_SCORE_CACHE_TELEMETRY").is_some() {
            let (hits, misses, entries, slots) = self.disambiguation_scores.stats();
            let (rows, row_capacity, pair_slots, overflow_entries, overflow_slots) =
                self.disambiguation_scores.layout_stats();
            let (
                dense_attempts,
                dense_successes,
                dense_duplicate_fallbacks,
                dense_state_fallbacks,
                dense_tie_fallbacks,
            ) = self.disambiguation_scores.dense_stats();
            eprintln!(
                "NEDO_TRIGRAM_SCORE_CACHE hits={hits} misses={misses} entries={entries} slots={slots} rows={rows} row_capacity={row_capacity} pair_slots={pair_slots} overflow_entries={overflow_entries} overflow_slots={overflow_slots} dense_attempts={dense_attempts} dense_successes={dense_successes} dense_duplicate_fallbacks={dense_duplicate_fallbacks} dense_state_fallbacks={dense_state_fallbacks} dense_tie_fallbacks={dense_tie_fallbacks} signatures={} lemmas={} igs={} ig_sequences={}",
                self.scoring_interner_counts().0,
                self.scoring_interner_counts().1,
                self.scoring_interner_counts().2,
                self.scoring_interner_counts().3,
            );
        }
        if std::env::var_os("NEDO_DOCUMENT_PROGRAM_TELEMETRY").is_some() {
            eprintln!(
                "NEDO_DOCUMENT_PROGRAM hits={} misses={} entries={} bytes={}",
                self.document_program_hits,
                self.document_program_misses,
                self.document_program_entries,
                self.document_program_bytes,
            );
        }
        TrainingCacheStats {
            hits: self.hits,
            misses: self.misses,
            saturated_misses: self.saturated_misses,
            entries: u64::try_from(self.entry_count).map_or(u64::MAX, |value| value),
            approximate_bytes: self
                .approximate_bytes
                .saturating_add(self.scoring_interner.approximate_bytes())
                .saturating_add(self.disambiguation_scores.approximate_bytes())
                .saturating_add(self.short_entries.approximate_bytes())
                .saturating_add(self.segment_program_bytes)
                .saturating_add(self.surface_program_bytes)
                .saturating_add(self.document_program_bytes),
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.short_entries.clear();
        self.entry_count = 0;
        self.hits = 0;
        self.misses = 0;
        self.saturated_misses = 0;
        self.approximate_bytes = 0;
        #[cfg(feature = "compiled-surface-table")]
        {
            self.scoring_interner = AmbiguityScoringInterner::default();
        }
        #[cfg(not(feature = "compiled-surface-table"))]
        {
            self.scoring_interner = AmbiguityScoringInterner::default();
        }
        self.disambiguation_scores.clear();
        self.segment_programs.clear();
        self.segment_key.clear();
        self.surface_programs.clear();
        self.surface_program_entries = 0;
        self.surface_program_bytes = 0;
        self.document_programs.clear();
        self.document_program_entries = 0;
        self.document_program_hits = 0;
        self.document_program_misses = 0;
        self.document_program_bytes = 0;
        self.surface_program_context = None;
        self.segment_program_hits = 0;
        self.segment_program_misses = 0;
        self.segment_program_bytes = 0;
        self.phase_scan_ns = 0;
        self.phase_code_ns = 0;
        self.phase_split_ns = 0;
        self.phase_analysis_ns = 0;
        self.phase_long_ns = 0;
        self.phase_vocab_ns = 0;
        self.phase_documents = 0;
    }
}

/// Stateful metadata-free training encoder with persistent worker-local caches.
pub struct FlatTrainingEncoder<'tokenizer, 'assets> {
    tokenizer: &'tokenizer Tokenizer<'assets>,
    vocabulary: &'tokenizer CharacterVocabulary,
    threads: usize,
    options: TrainingEncodingOptions,
    caches: Vec<FlatAnalysisCache>,
}

impl FlatTrainingEncoder<'_, '_> {
    /// Encodes one batch while preserving worker-local caches across calls.
    ///
    /// # Errors
    ///
    /// Returns an error for input/newline cardinality mismatch, worker failure,
    /// or any tokenizer/training-stream invariant violation.
    pub fn encode_batch(
        &mut self,
        inputs: &[Vec<u8>],
        newline_flags: &[bool],
    ) -> Result<TrainingBatch, TokenizerError> {
        if inputs.len() != newline_flags.len() {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "flat input and newline counts differ",
            ));
        }
        if inputs.is_empty() {
            return Ok(TrainingBatch {
                ids: Vec::new(),
                lengths: Vec::new(),
                document_offsets: vec![0],
            });
        }
        let ranges = build_batch_ranges(inputs, self.threads)?;
        let worker_count = self.threads.min(ranges.len());
        let chunks = if worker_count == 1 {
            let cache = self
                .caches
                .first_mut()
                .ok_or(TokenizerError::InvalidConfiguration(
                    "flat encoder has no worker cache",
                ))?;
            let mut encoded = Vec::with_capacity(inputs.len());
            for (raw, newline) in inputs.iter().zip(newline_flags) {
                encoded.push(self.tokenizer.encode_flat_document(
                    raw.clone(),
                    *newline,
                    self.vocabulary,
                    self.options,
                    cache,
                )?);
            }
            vec![(0, encoded)]
        } else {
            let next = AtomicUsize::new(0);
            let tokenizer = self.tokenizer;
            let vocabulary = self.vocabulary;
            let options = self.options;
            let worker_chunks = thread::scope(|scope| {
                let handles = self.caches[..worker_count]
                    .iter_mut()
                    .map(|cache| {
                        let ranges = &ranges;
                        let next = &next;
                        scope.spawn(move || {
                            let mut completed = Vec::new();
                            loop {
                                let chunk_index = next.fetch_add(1, Ordering::Relaxed);
                                let Some(range) = ranges.get(chunk_index) else {
                                    break;
                                };
                                let mut encoded = Vec::with_capacity(range.len());
                                for index in range.clone() {
                                    encoded.push(tokenizer.encode_flat_document(
                                        inputs[index].clone(),
                                        newline_flags[index],
                                        vocabulary,
                                        options,
                                        cache,
                                    )?);
                                }
                                completed.push((chunk_index, encoded));
                            }
                            Ok::<_, TokenizerError>(completed)
                        })
                    })
                    .collect::<Vec<_>>();
                handles
                    .into_iter()
                    .map(|handle| handle.join().map_err(|_| TokenizerError::WorkerPanicked)?)
                    .collect::<Result<Vec<_>, _>>()
            })?;
            worker_chunks.into_iter().flatten().collect::<Vec<_>>()
        };
        concatenate_training_chunks(chunks)
    }

    /// Returns aggregate cache telemetry across workers.
    #[must_use]
    pub fn cache_stats(&self) -> TrainingCacheStats {
        let mut stats = TrainingCacheStats::default();
        for cache in &self.caches {
            stats.merge(cache.stats());
        }
        stats
    }

    /// Clears all worker-local cache entries and counters.
    pub fn clear_caches(&mut self) {
        for cache in &mut self.caches {
            cache.clear();
        }
    }
}

fn encode_flat_surface_batch_with_caches(
    tokenizer: &Tokenizer<'_>,
    inputs: &[Vec<u8>],
    newline_flags: &[bool],
    vocabulary: &SurfaceVocabulary,
    threads: usize,
    use_morphology: bool,
    caches: &mut [FlatAnalysisCache],
) -> Result<TrainingBatch, TokenizerError> {
    if inputs.len() != newline_flags.len() {
        return Err(TokenizerError::InvalidTrainingEncoding(
            "flat surface input and newline counts differ",
        ));
    }
    if inputs.is_empty() {
        return Ok(TrainingBatch {
            ids: Vec::new(),
            lengths: Vec::new(),
            document_offsets: vec![0],
        });
    }
    if caches.len() < threads {
        return Err(TokenizerError::InvalidConfiguration(
            "surface runtime has fewer worker caches than threads",
        ));
    }
    let ranges = build_batch_ranges(inputs, threads)?;
    let worker_count = threads.min(ranges.len());
    if worker_count == 1 {
        let cache = caches
            .first_mut()
            .ok_or(TokenizerError::InvalidConfiguration(
                "surface runtime has no worker cache",
            ))?;
        return encode_flat_surface_range(
            tokenizer,
            inputs,
            newline_flags,
            0..inputs.len(),
            vocabulary,
            use_morphology,
            cache,
        );
    }
    let next = AtomicUsize::new(0);
    let worker_chunks = thread::scope(|scope| {
        let handles = caches[..worker_count]
            .iter_mut()
            .map(|cache| {
                let ranges = &ranges;
                let next = &next;
                scope.spawn(move || {
                    let mut completed = Vec::new();
                    loop {
                        let chunk_index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(range) = ranges.get(chunk_index) else {
                            break;
                        };
                        completed.push((
                            chunk_index,
                            encode_flat_surface_range(
                                tokenizer,
                                inputs,
                                newline_flags,
                                range.clone(),
                                vocabulary,
                                use_morphology,
                                cache,
                            )?,
                        ));
                    }
                    Ok::<_, TokenizerError>(completed)
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().map_err(|_| TokenizerError::WorkerPanicked)?)
            .collect::<Result<Vec<_>, _>>()
    })?;
    concatenate_surface_chunks(worker_chunks.into_iter().flatten().collect())
}

/// Opaque persistent worker-local runtime for repeated surface batches.
///
/// The runtime is bound to the exact tokenizer configuration and compiled-table
/// payload identity that created it. Passing it to a different tokenizer is
/// rejected before any document is encoded.
pub struct SurfaceRuntimeCache {
    config: TokenizerConfig,
    compiled_table_digest: Option<[u8; 32]>,
    threads: usize,
    options: SurfaceEncoderOptions,
    caches: Vec<FlatAnalysisCache>,
}

impl SurfaceRuntimeCache {
    /// Number of worker caches retained by this runtime.
    #[must_use]
    pub const fn threads(&self) -> usize {
        self.threads
    }

    /// Returns aggregate persistent-cache telemetry.
    #[must_use]
    pub fn cache_stats(&self) -> TrainingCacheStats {
        let mut stats = TrainingCacheStats::default();
        for cache in &self.caches {
            stats.merge(cache.stats());
        }
        stats
    }

    /// Clears every worker-local cache and counter without changing runtime policy.
    pub fn clear(&mut self) {
        for cache in &mut self.caches {
            cache.clear();
        }
    }
}

/// Stateful metadata-free surface-piece encoder with persistent worker-local caches.
pub struct FlatSurfaceEncoder<'tokenizer, 'assets> {
    tokenizer: &'tokenizer Tokenizer<'assets>,
    vocabulary: &'tokenizer SurfaceVocabulary,
    runtime: SurfaceRuntimeCache,
}

impl FlatSurfaceEncoder<'_, '_> {
    /// Encodes one batch directly to final surface-piece IDs.
    ///
    /// # Errors
    ///
    /// Returns an error for input/newline cardinality mismatch, worker failure,
    /// morphology/disambiguation failure, or surface encoding failure.
    pub fn encode_batch(
        &mut self,
        inputs: &[Vec<u8>],
        newline_flags: &[bool],
    ) -> Result<TrainingBatch, TokenizerError> {
        self.tokenizer.encode_surface_batch_with_runtime(
            inputs,
            newline_flags,
            self.vocabulary,
            &mut self.runtime,
        )
    }

    /// Returns aggregate cache telemetry across workers.
    #[must_use]
    pub fn cache_stats(&self) -> TrainingCacheStats {
        self.runtime.cache_stats()
    }

    /// Clears all worker-local cache entries and counters.
    pub fn clear_caches(&mut self) {
        self.runtime.clear();
    }
}

impl Tokenizer<'static> {
    /// Creates the production tokenizer from embedded, checksum-pinned assets.
    ///
    /// # Errors
    ///
    /// Returns an error if embedded checksums or native asset validation fails.
    pub fn embedded(config: TokenizerConfig) -> Result<Self, TokenizerError> {
        verify_sha(EMBEDDED_MORPHOLOGY, MORPHOLOGY_SHA256, "morphology")?;
        verify_sha(EMBEDDED_MODEL, MODEL_SHA256, "disambiguation model")?;
        Self::from_bytes(EMBEDDED_MORPHOLOGY, EMBEDDED_MODEL, config)
    }
}

impl<'a> Tokenizer<'a> {
    /// Creates a tokenizer only when supplied assets exactly match schema-v1 identities.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid configuration, checksums, morphology, or model bytes.
    pub fn from_bytes(
        morphology: &'a [u8],
        model: &[u8],
        config: TokenizerConfig,
    ) -> Result<Self, TokenizerError> {
        if config.max_sentence_tokens == 0 {
            return Err(TokenizerError::InvalidConfiguration(
                "max_sentence_tokens must be positive",
            ));
        }
        if config.max_fallback_chars == 0 || config.max_fallback_chars > MAX_FALLBACK_CHARS_CONFIG {
            return Err(TokenizerError::InvalidConfiguration(
                "max_fallback_chars must be in 1..=4096",
            ));
        }
        verify_sha(morphology, MORPHOLOGY_SHA256, "morphology")?;
        verify_sha(model, MODEL_SHA256, "disambiguation model")?;
        Ok(Self {
            morphology: NativeMorphology::parse(morphology)?,
            disambiguator: NativeDisambiguator::from_bytes(model)?,
            config,
            #[cfg(feature = "compiled-surface-table")]
            compiled_surface_analysis_table: None,
            #[cfg(feature = "compiled-surface-table")]
            nedoformer_compiled_surface_analysis_table: None,
        })
    }

    /// Attaches a full compiled analysis table after exact semantic revalidation.
    ///
    /// Every surface and candidate is regenerated by the active native morphology.
    /// Candidate order, cuts, unknown status, canonical identity, lemma, IG sequence,
    /// and Java hash must all match before the table can enter the hot path.
    ///
    /// # Errors
    ///
    /// Returns an error on invalid UTF-8, morphology/alignment failure, or any
    /// candidate metadata mismatch.
    #[cfg(feature = "compiled-surface-table")]
    pub fn with_verified_compiled_surface_analysis_table(
        mut self,
        table: CompiledSurfaceAnalysisTable,
    ) -> Result<Self, TokenizerError> {
        for (surface_bytes, compiled) in table.entries() {
            let surface = std::str::from_utf8(surface_bytes).map_err(|_| {
                TokenizerError::InvalidCompiledSurfaceTable {
                    surface: hex_bytes(surface_bytes),
                    reason: "surface key is not valid UTF-8",
                }
            })?;
            let exact = build_flat_analysis_set(
                surface,
                analyze_token_with_quality_fallback(&self.morphology, surface)?,
            )?;
            if exact.relative_cuts != compiled.relative_cuts {
                return Err(TokenizerError::InvalidCompiledSurfaceTable {
                    surface: surface.to_owned(),
                    reason: "candidate cuts or candidate order differ",
                });
            }
            if exact.unknown != compiled.unknown {
                return Err(TokenizerError::InvalidCompiledSurfaceTable {
                    surface: surface.to_owned(),
                    reason: "candidate unknown status or order differs",
                });
            }
            if exact.ambiguity != compiled.ambiguity {
                return Err(TokenizerError::InvalidCompiledSurfaceTable {
                    surface: surface.to_owned(),
                    reason: "candidate scoring metadata or order differs",
                });
            }
            if compiled.scoring_codes.len() != compiled.ambiguity.len() {
                return Err(TokenizerError::InvalidCompiledSurfaceTable {
                    surface: surface.to_owned(),
                    reason: "compiled scoring-code cardinality differs",
                });
            }
        }
        self.compiled_surface_analysis_table = Some(table);
        Ok(self)
    }

    /// Attaches an exact NedoFormer shadow-analysis lookup table.
    ///
    /// Unlike the standalone surface table, every entry is regenerated through
    /// the NedoFormer raw/lowercase/deasciify shadow analyzer before acceptance.
    /// This prevents a normal-surface table from silently changing lattice cuts.
    ///
    /// # Errors
    ///
    /// Returns an error on invalid UTF-8, shadow-analysis/alignment failure, or
    /// any candidate metadata mismatch.
    #[cfg(feature = "compiled-surface-table")]
    pub fn with_verified_nedoformer_compiled_surface_analysis_table(
        mut self,
        table: CompiledSurfaceAnalysisTable,
    ) -> Result<Self, TokenizerError> {
        for (surface_bytes, compiled) in table.entries() {
            let surface = std::str::from_utf8(surface_bytes).map_err(|_| {
                TokenizerError::InvalidCompiledSurfaceTable {
                    surface: hex_bytes(surface_bytes),
                    reason: "NedoFormer surface key is not valid UTF-8",
                }
            })?;
            let exact = build_flat_analysis_set(
                surface,
                analyze_token_with_nedoformer_shadow(&self.morphology, surface)?,
            )?;
            if exact.relative_cuts != compiled.relative_cuts {
                return Err(TokenizerError::InvalidCompiledSurfaceTable {
                    surface: surface.to_owned(),
                    reason: "NedoFormer candidate cuts or order differ",
                });
            }
            if exact.unknown != compiled.unknown {
                return Err(TokenizerError::InvalidCompiledSurfaceTable {
                    surface: surface.to_owned(),
                    reason: "NedoFormer candidate unknown status or order differs",
                });
            }
            if exact.ambiguity != compiled.ambiguity {
                return Err(TokenizerError::InvalidCompiledSurfaceTable {
                    surface: surface.to_owned(),
                    reason: "NedoFormer candidate scoring metadata or order differs",
                });
            }
            if compiled.scoring_codes.len() != compiled.ambiguity.len() {
                return Err(TokenizerError::InvalidCompiledSurfaceTable {
                    surface: surface.to_owned(),
                    reason: "NedoFormer compiled scoring-code cardinality differs",
                });
            }
        }
        self.nedoformer_compiled_surface_analysis_table = Some(table);
        Ok(self)
    }

    /// Attaches a NedoFormer compiled table whose payload digest was pinned by a trusted manifest.
    ///
    /// [`CompiledSurfaceAnalysisTable::from_bytes`] has already validated the table schema,
    /// embedded morphology/model identities, payload checksum, UTF-8 surfaces, cuts, and
    /// candidate structure. This fast path additionally requires the payload digest to match
    /// an independently trusted value, allowing production startup to skip regenerating every
    /// table entry through morphology. The expected digest must come from a trusted release
    /// manifest or other out-of-band seal; deriving it from the same untrusted table defeats
    /// the purpose of this check.
    ///
    /// Use [`Self::with_verified_nedoformer_compiled_surface_analysis_table`] when establishing
    /// trust in a newly built table or when no trusted digest is available.
    ///
    /// # Errors
    ///
    /// Returns an error when the parsed table payload digest differs from `expected_digest`.
    #[cfg(feature = "compiled-surface-table")]
    pub fn with_pinned_nedoformer_compiled_surface_analysis_table(
        mut self,
        table: CompiledSurfaceAnalysisTable,
        expected_digest: [u8; 32],
    ) -> Result<Self, TokenizerError> {
        let actual = table.digest();
        if actual != expected_digest {
            return Err(TokenizerError::CompiledSurfaceTableSealMismatch {
                expected: hex_bytes(&expected_digest),
                actual: hex_bytes(&actual),
            });
        }
        self.nedoformer_compiled_surface_analysis_table = Some(table);
        Ok(self)
    }

    /// Tokenizes one arbitrary byte document without normalization or replacement.
    ///
    /// # Errors
    ///
    /// Returns an error for scanning, morphology, context decoding, alignment, or invariant failure.
    pub fn tokenize(&self, raw: Vec<u8>) -> Result<TokenizedDocument, TokenizerError> {
        let mut cache = AnalysisCache::new(0);
        self.tokenize_with_cache(raw, &mut cache)
    }

    fn tokenize_with_cache(
        &self,
        raw: Vec<u8>,
        cache: &mut AnalysisCache,
    ) -> Result<TokenizedDocument, TokenizerError> {
        let scan = scan(raw)?;
        let raw = scan.document().decode();
        let code_spans = match self.config.mode {
            TokenizerMode::Auto if self.config.detect_unmarked_code => {
                auto_code_spans(raw, scan.code_hints())
            }
            TokenizerMode::Auto => explicit_code_spans(raw),
            TokenizerMode::Turkish | TokenizerMode::Code => Vec::new(),
        };
        let mut units = split_units(&scan, &code_spans, self.config.mode)?;
        self.apply_contextual_analysis(raw, &mut units, cache)?;
        let mut units = split_long_fallback_units(raw, units, self.config.max_fallback_chars)?;
        assign_inner_groups(raw, &mut units)?;
        let (raw, _) = scan.into_document().into_parts();
        TokenizedDocument::new(raw, units)
    }

    /// Deterministic native parallel batch tokenization.
    ///
    /// Work is split into contiguous chunks and results are reassembled in input
    /// order. No per-token language-boundary callback is used.
    ///
    /// # Errors
    ///
    /// Returns an error for zero threads, worker failure, or any document tokenization failure.
    pub fn tokenize_batch(
        &self,
        inputs: &[Vec<u8>],
        threads: usize,
    ) -> Result<Vec<TokenizedDocument>, TokenizerError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        if threads == 0 {
            return Err(TokenizerError::InvalidConfiguration(
                "batch thread count must be positive",
            ));
        }
        let ranges = build_batch_ranges(inputs, threads)?;
        let worker_count = threads.min(ranges.len());
        if worker_count == 1 {
            let mut cache = AnalysisCache::new(BATCH_ANALYSIS_CACHE_ENTRIES);
            return inputs
                .iter()
                .cloned()
                .map(|raw| self.tokenize_with_cache(raw, &mut cache))
                .collect();
        }
        let next = AtomicUsize::new(0);
        let worker_chunks = thread::scope(|scope| {
            let handles = (0..worker_count)
                .map(|_| {
                    let ranges = &ranges;
                    let next = &next;
                    scope.spawn(move || {
                        let mut cache = AnalysisCache::new(BATCH_ANALYSIS_CACHE_ENTRIES);
                        let mut completed = Vec::new();
                        loop {
                            let chunk_index = next.fetch_add(1, Ordering::Relaxed);
                            let Some(range) = ranges.get(chunk_index) else {
                                break;
                            };
                            let documents = inputs[range.clone()]
                                .iter()
                                .cloned()
                                .map(|raw| self.tokenize_with_cache(raw, &mut cache))
                                .collect::<Result<Vec<_>, _>>()?;
                            completed.push((chunk_index, documents));
                        }
                        Ok::<_, TokenizerError>(completed)
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().map_err(|_| TokenizerError::WorkerPanicked)?)
                .collect::<Result<Vec<_>, _>>()
        })?;
        let mut chunks = worker_chunks.into_iter().flatten().collect::<Vec<_>>();
        chunks.sort_unstable_by_key(|entry| entry.0);
        Ok(chunks
            .into_iter()
            .flat_map(|(_, documents)| documents)
            .collect())
    }

    /// Encodes a batch directly into flat training streams without building
    /// rich [`TokenizedDocument`] or [`AnalysisMetadata`] object graphs.
    ///
    /// Input order, structural controls, selected morphology, byte lengths, and
    /// document token offsets are exact with the rich reference encoder.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid thread counts, cardinality mismatch,
    /// morphology/disambiguation failures, or any flat-stream invariant failure.
    pub fn encode_training_batch(
        &self,
        inputs: &[Vec<u8>],
        newline_flags: &[bool],
        vocabulary: &CharacterVocabulary,
        threads: usize,
        options: TrainingEncodingOptions,
    ) -> Result<TrainingBatch, TokenizerError> {
        let mut encoder = self.training_encoder(vocabulary, threads, options)?;
        encoder.encode_batch(inputs, newline_flags)
    }

    #[cfg_attr(
        not(feature = "compiled-surface-table"),
        allow(clippy::missing_const_for_fn, clippy::unused_self)
    )]
    fn compiled_surface_table_digest(&self) -> Option<[u8; 32]> {
        #[cfg(feature = "compiled-surface-table")]
        {
            self.compiled_surface_analysis_table
                .as_ref()
                .map(CompiledSurfaceAnalysisTable::digest)
        }
        #[cfg(not(feature = "compiled-surface-table"))]
        {
            None
        }
    }

    fn new_flat_analysis_cache(
        &self,
        capacity: usize,
        document_program_capacity: usize,
        context_cache_policy: ContextCachePolicy,
    ) -> FlatAnalysisCache {
        #[cfg(feature = "compiled-surface-table")]
        if let Some(table) = self.compiled_surface_analysis_table.as_ref() {
            return FlatAnalysisCache::new_seeded(
                capacity,
                document_program_capacity,
                context_cache_policy,
                table.scoring_interner().clone(),
                None,
            );
        }
        FlatAnalysisCache::new(capacity, document_program_capacity, context_cache_policy)
    }

    #[cfg_attr(
        not(feature = "compiled-surface-table"),
        allow(clippy::missing_const_for_fn, clippy::unused_self)
    )]
    fn nedoformer_compiled_surface_table_digest(&self) -> Option<[u8; 32]> {
        #[cfg(feature = "compiled-surface-table")]
        {
            self.nedoformer_compiled_surface_analysis_table
                .as_ref()
                .map(CompiledSurfaceAnalysisTable::digest)
        }
        #[cfg(not(feature = "compiled-surface-table"))]
        {
            None
        }
    }

    fn new_nedoformer_flat_analysis_cache(
        &self,
        capacity: usize,
        context_cache_policy: ContextCachePolicy,
        shared_prepared_words: Option<Arc<SharedPreparedWordCache>>,
    ) -> FlatAnalysisCache {
        #[cfg(feature = "compiled-surface-table")]
        if let Some(table) = self.nedoformer_compiled_surface_analysis_table.as_ref() {
            return FlatAnalysisCache::new_seeded(
                capacity,
                0,
                context_cache_policy,
                table.scoring_interner().clone(),
                shared_prepared_words,
            );
        }
        FlatAnalysisCache::new(capacity, 0, context_cache_policy)
    }

    #[cfg_attr(
        not(feature = "compiled-surface-table"),
        allow(clippy::missing_const_for_fn, clippy::unused_self)
    )]
    fn new_nedoformer_shared_prepared_word_cache(&self) -> Option<Arc<SharedPreparedWordCache>> {
        #[cfg(feature = "compiled-surface-table")]
        {
            self.nedoformer_compiled_surface_analysis_table
                .as_ref()
                .map(|table| {
                    Arc::new(SharedPreparedWordCache::with_slots(
                        table.scoring_interner().signature_count(),
                    ))
                })
        }
        #[cfg(not(feature = "compiled-surface-table"))]
        {
            None
        }
    }

    /// Creates a stateful flat encoder whose worker-local caches survive calls.
    ///
    /// # Errors
    ///
    /// Returns an error when `threads` is zero.
    pub fn training_encoder<'tokenizer>(
        &'tokenizer self,
        vocabulary: &'tokenizer CharacterVocabulary,
        threads: usize,
        options: TrainingEncodingOptions,
    ) -> Result<FlatTrainingEncoder<'tokenizer, 'a>, TokenizerError> {
        if threads == 0 {
            return Err(TokenizerError::InvalidConfiguration(
                "flat batch thread count must be positive",
            ));
        }
        let analysis_cache_entries = analysis_cache_entries_for_parallelism(threads);
        Ok(FlatTrainingEncoder {
            tokenizer: self,
            vocabulary,
            threads,
            options,
            caches: (0..threads)
                .map(|_| {
                    self.new_flat_analysis_cache(
                        analysis_cache_entries,
                        0,
                        ContextCachePolicy::Full,
                    )
                })
                .collect(),
        })
    }

    /// Creates an opaque persistent runtime for repeated surface batches.
    ///
    /// # Errors
    ///
    /// Returns an error when `threads` is zero.
    pub fn surface_runtime_cache(
        &self,
        threads: usize,
        options: SurfaceEncoderOptions,
    ) -> Result<SurfaceRuntimeCache, TokenizerError> {
        if threads == 0 {
            return Err(TokenizerError::InvalidConfiguration(
                "surface runtime thread count must be positive",
            ));
        }
        self.surface_runtime_cache_with_analysis_entries(
            threads,
            options,
            analysis_cache_entries_for_parallelism(threads),
        )
    }

    pub(crate) fn surface_runtime_cache_with_analysis_entries(
        &self,
        threads: usize,
        options: SurfaceEncoderOptions,
        analysis_cache_entries: usize,
    ) -> Result<SurfaceRuntimeCache, TokenizerError> {
        if threads == 0 {
            return Err(TokenizerError::InvalidConfiguration(
                "surface runtime thread count must be positive",
            ));
        }
        Ok(SurfaceRuntimeCache {
            config: self.config,
            compiled_table_digest: self.compiled_surface_table_digest(),
            threads,
            options,
            caches: (0..threads)
                .map(|_| {
                    self.new_flat_analysis_cache(
                        analysis_cache_entries,
                        options.document_cache_entries,
                        options.context_cache_policy,
                    )
                })
                .collect(),
        })
    }

    /// Encodes with a persistent runtime created by [`Self::surface_runtime_cache`].
    ///
    /// # Errors
    ///
    /// Rejects cardinality mismatch, a runtime from a different tokenizer/table,
    /// worker failures, and all normal encoding failures.
    pub fn encode_surface_batch_with_runtime(
        &self,
        inputs: &[Vec<u8>],
        newline_flags: &[bool],
        vocabulary: &SurfaceVocabulary,
        runtime: &mut SurfaceRuntimeCache,
    ) -> Result<TrainingBatch, TokenizerError> {
        if runtime.config != self.config
            || runtime.compiled_table_digest != self.compiled_surface_table_digest()
        {
            return Err(TokenizerError::InvalidConfiguration(
                "surface runtime belongs to a different tokenizer configuration or compiled table",
            ));
        }
        encode_flat_surface_batch_with_caches(
            self,
            inputs,
            newline_flags,
            vocabulary,
            runtime.threads,
            runtime.options.use_morphology,
            &mut runtime.caches,
        )
    }

    /// Encodes one batch directly to final surface-piece IDs.
    ///
    /// # Errors
    ///
    /// Returns an error when the thread count is zero or encoding fails.
    pub fn encode_surface_batch(
        &self,
        inputs: &[Vec<u8>],
        newline_flags: &[bool],
        vocabulary: &SurfaceVocabulary,
        threads: usize,
        use_morphology: bool,
    ) -> Result<TrainingBatch, TokenizerError> {
        let mut encoder = self.surface_encoder(vocabulary, threads, use_morphology)?;
        encoder.encode_batch(inputs, newline_flags)
    }

    /// Creates a stateful final surface-piece encoder with persistent caches.
    ///
    /// # Errors
    ///
    /// Returns an error when `threads` is zero.
    pub fn surface_encoder<'tokenizer>(
        &'tokenizer self,
        vocabulary: &'tokenizer SurfaceVocabulary,
        threads: usize,
        use_morphology: bool,
    ) -> Result<FlatSurfaceEncoder<'tokenizer, 'a>, TokenizerError> {
        self.surface_encoder_with_options(
            vocabulary,
            threads,
            SurfaceEncoderOptions::cached(use_morphology),
        )
    }

    /// Creates a stateful final surface-piece encoder with an explicit cache policy.
    ///
    /// Use [`SurfaceEncoderOptions::one_pass`] for new-corpus preprocessing so
    /// complete source documents and their output vectors are not retained.
    ///
    /// # Errors
    ///
    /// Returns an error when `threads` is zero.
    pub fn surface_encoder_with_options<'tokenizer>(
        &'tokenizer self,
        vocabulary: &'tokenizer SurfaceVocabulary,
        threads: usize,
        options: SurfaceEncoderOptions,
    ) -> Result<FlatSurfaceEncoder<'tokenizer, 'a>, TokenizerError> {
        if threads == 0 {
            return Err(TokenizerError::InvalidConfiguration(
                "flat surface batch thread count must be positive",
            ));
        }
        self.surface_encoder_with_analysis_cache_entries(
            vocabulary,
            threads,
            options,
            analysis_cache_entries_for_parallelism(threads),
        )
    }

    pub(crate) fn surface_encoder_with_analysis_cache_entries<'tokenizer>(
        &'tokenizer self,
        vocabulary: &'tokenizer SurfaceVocabulary,
        threads: usize,
        options: SurfaceEncoderOptions,
        analysis_cache_entries: usize,
    ) -> Result<FlatSurfaceEncoder<'tokenizer, 'a>, TokenizerError> {
        Ok(FlatSurfaceEncoder {
            tokenizer: self,
            vocabulary,
            runtime: self.surface_runtime_cache_with_analysis_entries(
                threads,
                options,
                analysis_cache_entries,
            )?,
        })
    }

    fn encode_flat_document(
        &self,
        raw: Vec<u8>,
        newline: bool,
        vocabulary: &CharacterVocabulary,
        options: TrainingEncodingOptions,
        cache: &mut FlatAnalysisCache,
    ) -> Result<TrainingEncoding, TokenizerError> {
        let scan = scan_compact(raw)?;
        let raw = scan.raw();
        let code_spans = match self.config.mode {
            TokenizerMode::Auto if self.config.detect_unmarked_code => {
                auto_code_spans(raw, scan.code_hints())
            }
            TokenizerMode::Auto => explicit_code_spans(raw),
            TokenizerMode::Turkish | TokenizerMode::Code => Vec::new(),
        };
        let mut units = split_compact_units(&scan, &code_spans, self.config.mode)?;
        self.apply_flat_contextual_analysis(raw, &mut units, cache)?;
        let units = split_long_flat_units(raw, units, self.config.max_fallback_chars)?;
        vocabulary.encode_training_units(raw, &units, newline, options)
    }

    fn append_flat_surface_document(
        &self,
        raw: &[u8],
        newline: bool,
        vocabulary: &SurfaceVocabulary,
        use_morphology: bool,
        cache: &mut FlatAnalysisCache,
        output: &mut TrainingBatch,
    ) -> Result<(), TokenizerError> {
        cache.prepare_surface_program_context(vocabulary, use_morphology);
        if flat_surface::try_append_cached_document_program(
            &raw,
            newline,
            cache,
            &mut output.ids,
            &mut output.lengths,
        )? {
            output.document_offsets.push(
                u64::try_from(output.ids.len())
                    .map_err(|_| TokenizerError::LengthOverflow("surface token offset"))?,
            );
            return Ok(());
        }

        let id_start = output.ids.len();
        let length_start = output.lengths.len();
        let exact = (cache.document_program_entries < cache.document_program_capacity)
            .then(|| raw.to_vec().into_boxed_slice());
        self.append_flat_surface_document_uncached(
            raw,
            newline,
            vocabulary,
            use_morphology,
            cache,
            output,
        )?;
        if let Some(exact) = exact {
            flat_surface::insert_document_program(
                cache,
                exact,
                newline,
                &output.ids[id_start..],
                &output.lengths[length_start..],
            )?;
        }
        Ok(())
    }

    fn append_flat_surface_document_uncached(
        &self,
        raw: &[u8],
        newline: bool,
        vocabulary: &SurfaceVocabulary,
        use_morphology: bool,
        cache: &mut FlatAnalysisCache,
        output: &mut TrainingBatch,
    ) -> Result<(), TokenizerError> {
        let telemetry = std::env::var_os("NEDO_SURFACE_PHASE_TELEMETRY").is_some();
        match self.config.mode {
            TokenizerMode::Auto => {
                let started = telemetry.then(std::time::Instant::now);
                if flat_surface::try_append_cached_auto_raw_document(
                    raw,
                    newline,
                    vocabulary,
                    self.config.max_sentence_tokens,
                    self.config.max_fallback_chars,
                    self.config.detect_unmarked_code,
                    cache,
                    &mut output.ids,
                    &mut output.lengths,
                )? {
                    if let Some(started) = started {
                        cache.phase_scan_ns = cache
                            .phase_scan_ns
                            .saturating_add(started.elapsed().as_nanos());
                        cache.phase_documents = cache.phase_documents.saturating_add(1);
                    }
                    output.document_offsets.push(
                        u64::try_from(output.ids.len())
                            .map_err(|_| TokenizerError::LengthOverflow("surface token offset"))?,
                    );
                    return Ok(());
                }
                let scan = scan_compact(raw.to_vec())?;
                if let Some(started) = started {
                    cache.phase_scan_ns = cache
                        .phase_scan_ns
                        .saturating_add(started.elapsed().as_nanos());
                }
                let raw = scan.raw();
                let started = telemetry.then(std::time::Instant::now);
                let code_spans = if self.config.detect_unmarked_code {
                    auto_code_spans(raw, scan.code_hints())
                } else {
                    explicit_code_spans(raw)
                };
                if let Some(started) = started {
                    cache.phase_code_ns = cache
                        .phase_code_ns
                        .saturating_add(started.elapsed().as_nanos());
                }
                let started = telemetry.then(std::time::Instant::now);
                if code_spans.is_empty()
                    && flat_surface::try_append_cached_auto_document(
                        &scan,
                        newline,
                        vocabulary,
                        self.config.max_sentence_tokens,
                        self.config.max_fallback_chars,
                        cache,
                        &mut output.ids,
                        &mut output.lengths,
                    )?
                {
                    if let Some(started) = started {
                        cache.phase_split_ns = cache
                            .phase_split_ns
                            .saturating_add(started.elapsed().as_nanos());
                        cache.phase_documents = cache.phase_documents.saturating_add(1);
                    }
                    output.document_offsets.push(
                        u64::try_from(output.ids.len())
                            .map_err(|_| TokenizerError::LengthOverflow("surface token offset"))?,
                    );
                    return Ok(());
                }
                let units = flat_surface::split_units(&scan, &code_spans, self.config.mode)?;
                if let Some(started) = started {
                    cache.phase_split_ns = cache
                        .phase_split_ns
                        .saturating_add(started.elapsed().as_nanos());
                }
                self.append_flat_surface_units(
                    raw,
                    units,
                    newline,
                    vocabulary,
                    use_morphology,
                    telemetry,
                    cache,
                    output,
                )
            }
            TokenizerMode::Turkish => {
                let started = telemetry.then(std::time::Instant::now);
                if flat_surface::try_append_cached_turkish_document(
                    raw,
                    newline,
                    vocabulary,
                    self.config.max_sentence_tokens,
                    self.config.max_fallback_chars,
                    cache,
                    &mut output.ids,
                    &mut output.lengths,
                )? {
                    if let Some(started) = started {
                        cache.phase_scan_ns = cache
                            .phase_scan_ns
                            .saturating_add(started.elapsed().as_nanos());
                        cache.phase_documents = cache.phase_documents.saturating_add(1);
                    }
                    output.document_offsets.push(
                        u64::try_from(output.ids.len())
                            .map_err(|_| TokenizerError::LengthOverflow("surface token offset"))?,
                    );
                    return Ok(());
                }
                let units = flat_surface::scan_fixed_units(raw, self.config.mode)?;
                if let Some(started) = started {
                    cache.phase_scan_ns = cache
                        .phase_scan_ns
                        .saturating_add(started.elapsed().as_nanos());
                }
                self.append_flat_surface_units(
                    raw,
                    units,
                    newline,
                    vocabulary,
                    use_morphology,
                    telemetry,
                    cache,
                    output,
                )
            }
            TokenizerMode::Code => {
                let started = telemetry.then(std::time::Instant::now);
                flat_surface::append_cached_code_document(
                    raw,
                    newline,
                    vocabulary,
                    self.config.max_fallback_chars,
                    cache,
                    &mut output.ids,
                    &mut output.lengths,
                )?;
                if let Some(started) = started {
                    cache.phase_scan_ns = cache
                        .phase_scan_ns
                        .saturating_add(started.elapsed().as_nanos());
                    cache.phase_documents = cache.phase_documents.saturating_add(1);
                }
                output.document_offsets.push(
                    u64::try_from(output.ids.len())
                        .map_err(|_| TokenizerError::LengthOverflow("surface token offset"))?,
                );
                Ok(())
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn append_flat_surface_units(
        &self,
        raw: &[u8],
        mut units: Vec<flat_surface::FlatSurfaceUnit>,
        newline: bool,
        vocabulary: &SurfaceVocabulary,
        use_morphology: bool,
        telemetry: bool,
        cache: &mut FlatAnalysisCache,
        output: &mut TrainingBatch,
    ) -> Result<(), TokenizerError> {
        let mut cuts = Vec::new();
        let started = telemetry.then(std::time::Instant::now);
        let surface_programs = flat_surface::apply_contextual_analysis(
            self,
            raw,
            &mut units,
            &mut cuts,
            vocabulary,
            use_morphology,
            self.config.max_fallback_chars,
            cache,
        )?;
        if let Some(started) = started {
            cache.phase_analysis_ns = cache
                .phase_analysis_ns
                .saturating_add(started.elapsed().as_nanos());
        }
        let started = telemetry.then(std::time::Instant::now);
        vocabulary.encode_flat_units_with_programs(
            raw,
            &units,
            &cuts,
            &surface_programs,
            self.config.max_fallback_chars,
            newline,
            use_morphology,
            &mut output.ids,
            &mut output.lengths,
        )?;
        if let Some(started) = started {
            cache.phase_vocab_ns = cache
                .phase_vocab_ns
                .saturating_add(started.elapsed().as_nanos());
            cache.phase_documents = cache.phase_documents.saturating_add(1);
        }
        output.document_offsets.push(
            u64::try_from(output.ids.len())
                .map_err(|_| TokenizerError::LengthOverflow("surface token offset"))?,
        );
        Ok(())
    }

    fn apply_contextual_analysis(
        &self,
        raw: &[u8],
        units: &mut [TokenizedUnit],
        cache: &mut AnalysisCache,
    ) -> Result<(), TokenizerError> {
        let mut segment = Vec::new();
        for index in 0..units.len() {
            let unit = &units[index];
            if unit.mode != TokenMode::Turkish
                || matches!(
                    unit.kind,
                    LexicalKind::LineBreak | LexicalKind::Control | LexicalKind::Opaque
                )
            {
                self.flush_segment(raw, units, &mut segment, cache)?;
                continue;
            }
            if matches!(unit.kind, LexicalKind::Whitespace) {
                continue;
            }
            segment.push(index);
            let boundary = is_sentence_boundary(unit.kind, raw, unit.span)?;
            if boundary || segment.len() >= self.config.max_sentence_tokens {
                self.flush_segment(raw, units, &mut segment, cache)?;
            }
        }
        self.flush_segment(raw, units, &mut segment, cache)
    }

    fn flush_segment(
        &self,
        raw: &[u8],
        units: &mut [TokenizedUnit],
        indices: &mut Vec<usize>,
        cache: &mut AnalysisCache,
    ) -> Result<(), TokenizerError> {
        if indices.is_empty() {
            return Ok(());
        }
        let tokens = indices
            .iter()
            .map(|index| unit_str(raw, units[*index].span))
            .collect::<Result<Vec<_>, _>>()?;
        let mut candidate_sets = Vec::with_capacity(tokens.len());
        for token in &tokens {
            candidate_sets.push(cache.analyze(&self.morphology, token)?);
        }
        if self.config.contextual_disambiguation {
            let ambiguity = candidate_sets
                .iter()
                .map(|set| set.ambiguity.as_ref())
                .collect::<Vec<_>>();
            let selected = self.disambiguator.disambiguate_indices_causal(&ambiguity)?;
            if selected.len() != indices.len() {
                return Err(TokenizerError::ContextLengthMismatch);
            }
            for ((&index, set), candidate_index) in
                indices.iter().zip(&candidate_sets).zip(selected)
            {
                let analysis = set.analyses.get(candidate_index).ok_or(
                    TokenizerError::InvalidTrainingEncoding(
                        "rich disambiguator selected an out-of-range candidate",
                    ),
                )?;
                apply_selected_analysis(raw, &mut units[index], analysis.clone())?;
            }
        } else {
            for (&index, set) in indices.iter().zip(candidate_sets) {
                let mut applied = false;
                for analysis in &set.analyses {
                    match apply_selected_analysis(raw, &mut units[index], analysis.clone()) {
                        Ok(()) => {
                            applied = true;
                            break;
                        }
                        Err(TokenizerError::AlignmentMismatch { .. }) => {}
                        Err(error) => return Err(error),
                    }
                }
                if !applied {
                    units[index].status = TokenStatus::Unknown;
                    units[index].cuts.clear();
                    units[index].analysis = Some(unknown_chunk_analysis(raw, units[index].span)?);
                }
            }
        }
        indices.clear();
        Ok(())
    }

    fn apply_flat_contextual_analysis(
        &self,
        raw: &[u8],
        units: &mut [TokenizedUnit],
        cache: &mut FlatAnalysisCache,
    ) -> Result<(), TokenizerError> {
        let mut segment = Vec::new();
        for index in 0..units.len() {
            let unit = &units[index];
            if unit.mode != TokenMode::Turkish
                || matches!(
                    unit.kind,
                    LexicalKind::LineBreak | LexicalKind::Control | LexicalKind::Opaque
                )
            {
                self.flush_flat_segment(raw, units, &mut segment, cache)?;
                continue;
            }
            if matches!(unit.kind, LexicalKind::Whitespace) {
                continue;
            }
            segment.push(index);
            let boundary = is_sentence_boundary(unit.kind, raw, unit.span)?;
            if boundary || segment.len() >= self.config.max_sentence_tokens {
                self.flush_flat_segment(raw, units, &mut segment, cache)?;
            }
        }
        self.flush_flat_segment(raw, units, &mut segment, cache)
    }

    fn flush_flat_segment(
        &self,
        raw: &[u8],
        units: &mut [TokenizedUnit],
        indices: &mut Vec<usize>,
        cache: &mut FlatAnalysisCache,
    ) -> Result<(), TokenizerError> {
        if indices.is_empty() {
            return Ok(());
        }
        if let Some(program) = cache.segment_program(raw, units, indices)? {
            apply_flat_segment_program(units, indices, &program)?;
            indices.clear();
            return Ok(());
        }

        #[cfg(feature = "compiled-surface-table")]
        if let Some(table) = self.compiled_surface_analysis_table.as_ref() {
            let mut sources = Vec::with_capacity(indices.len());
            for index in indices.iter().copied() {
                let token = unit_str(raw, units[index].span)?;
                if let Some(set) = table.get(token) {
                    sources.push(FlatAnalysisSource::Compiled(set));
                } else {
                    sources.push(FlatAnalysisSource::Live(
                        cache.analyze(&self.morphology, token)?,
                    ));
                }
            }
            let sets = sources
                .iter()
                .map(FlatAnalysisSource::set)
                .collect::<Vec<_>>();
            self.apply_flat_sets(raw, units, indices, &sets, cache)?;
        } else {
            let mut owned_sets = Vec::with_capacity(indices.len());
            for index in indices.iter().copied() {
                let token = unit_str(raw, units[index].span)?;
                owned_sets.push(cache.analyze(&self.morphology, token)?);
            }
            let sets = owned_sets.iter().map(AsRef::as_ref).collect::<Vec<_>>();
            self.apply_flat_sets(raw, units, indices, &sets, cache)?;
        }
        #[cfg(not(feature = "compiled-surface-table"))]
        {
            let mut owned_sets = Vec::with_capacity(indices.len());
            for index in indices.iter().copied() {
                let token = unit_str(raw, units[index].span)?;
                owned_sets.push(cache.analyze(&self.morphology, token)?);
            }
            let sets = owned_sets.iter().map(AsRef::as_ref).collect::<Vec<_>>();
            self.apply_flat_sets(raw, units, indices, &sets, cache)?;
        }

        cache.insert_segment_program(capture_flat_segment_program(units, indices)?);
        indices.clear();
        Ok(())
    }

    fn apply_flat_sets(
        &self,
        raw: &[u8],
        units: &mut [TokenizedUnit],
        indices: &[usize],
        sets: &[&FlatAnalysisSet],
        cache: &mut FlatAnalysisCache,
    ) -> Result<(), TokenizerError> {
        let output_invariant = sets.iter().all(|set| set.output_invariant);

        if self.config.contextual_disambiguation && !output_invariant {
            let ambiguity = sets
                .iter()
                .map(|set| set.ambiguity.as_ref())
                .collect::<Vec<_>>();
            let scoring_codes = sets
                .iter()
                .map(|set| set.scoring_codes.as_ref())
                .collect::<Vec<_>>();
            let selected = self.disambiguator.disambiguate_indices_scored_causal(
                &ambiguity,
                &scoring_codes,
                &mut cache.disambiguation_scores,
            )?;
            if selected.len() != indices.len() {
                return Err(TokenizerError::ContextLengthMismatch);
            }
            for ((&index, set), candidate_index) in indices.iter().zip(sets).zip(selected) {
                apply_flat_candidate(raw, &mut units[index], set, candidate_index)?;
            }
        } else {
            for (&index, set) in indices.iter().zip(sets) {
                apply_flat_candidate(raw, &mut units[index], set, 0)?;
            }
        }
        Ok(())
    }
}

fn capture_flat_segment_program(
    units: &[TokenizedUnit],
    indices: &[usize],
) -> Result<FlatSegmentProgram, TokenizerError> {
    let mut tokens = Vec::with_capacity(indices.len());
    let mut relative_cuts = Vec::new();
    for index in indices {
        let unit = units
            .get(*index)
            .ok_or(TokenizerError::InvalidTrainingEncoding(
                "segment program capture index is out of range",
            ))?;
        let cut_start = u32::try_from(relative_cuts.len())
            .map_err(|_| TokenizerError::LengthOverflow("segment program cut offset"))?;
        for cut in &unit.cuts {
            let relative =
                cut.checked_sub(unit.span.start)
                    .ok_or(TokenizerError::InvalidTrainingEncoding(
                        "segment program cut precedes its unit",
                    ))?;
            relative_cuts.push(
                u32::try_from(relative)
                    .map_err(|_| TokenizerError::LengthOverflow("segment program relative cut"))?,
            );
        }
        tokens.push(FlatSegmentTokenProgram {
            cut_start,
            cut_len: u16::try_from(unit.cuts.len())
                .map_err(|_| TokenizerError::LengthOverflow("segment program cut count"))?,
            unknown: unit.status == TokenStatus::Unknown,
        });
    }
    Ok(FlatSegmentProgram {
        tokens: tokens.into_boxed_slice(),
        relative_cuts: relative_cuts.into_boxed_slice(),
        surface_ids: Box::new([]),
        surface_lengths: Box::new([]),
    })
}

fn apply_flat_segment_program(
    units: &mut [TokenizedUnit],
    indices: &[usize],
    program: &FlatSegmentProgram,
) -> Result<(), TokenizerError> {
    if indices.len() != program.tokens.len() {
        return Err(TokenizerError::ContextLengthMismatch);
    }
    for (index, token) in indices.iter().copied().zip(program.tokens.iter()) {
        let unit = units
            .get_mut(index)
            .ok_or(TokenizerError::InvalidTrainingEncoding(
                "segment program apply index is out of range",
            ))?;
        let start = usize::try_from(token.cut_start)
            .map_err(|_| TokenizerError::LengthOverflow("segment program cut start"))?;
        let end = start
            .checked_add(usize::from(token.cut_len))
            .ok_or(TokenizerError::LengthOverflow("segment program cut end"))?;
        let relative = program.relative_cuts.get(start..end).ok_or(
            TokenizerError::InvalidTrainingEncoding("segment program cut range is invalid"),
        )?;
        unit.status = if token.unknown {
            TokenStatus::Unknown
        } else {
            TokenStatus::Morphological
        };
        unit.cuts.clear();
        unit.cuts.reserve(relative.len());
        for cut in relative {
            unit.cuts
                .push(unit.span.start.checked_add(u64::from(*cut)).ok_or(
                    TokenizerError::LengthOverflow("segment program absolute cut"),
                )?);
        }
    }
    Ok(())
}

fn apply_flat_candidate(
    raw: &[u8],
    unit: &mut TokenizedUnit,
    set: &FlatAnalysisSet,
    candidate_index: usize,
) -> Result<(), TokenizerError> {
    let relative =
        set.relative_cuts
            .get(candidate_index)
            .ok_or(TokenizerError::InvalidTrainingEncoding(
                "flat disambiguator selected an out-of-range candidate",
            ))?;
    let unknown =
        *set.unknown
            .get(candidate_index)
            .ok_or(TokenizerError::InvalidTrainingEncoding(
                "flat candidate status is out of range",
            ))?;
    apply_flat_output(raw, unit, relative, unknown)
}

fn apply_flat_output(
    raw: &[u8],
    unit: &mut TokenizedUnit,
    relative: &[u32],
    unknown: bool,
) -> Result<(), TokenizerError> {
    if matches!(unit.kind, LexicalKind::Punctuation | LexicalKind::Symbol) {
        unit.status = TokenStatus::Structural;
        unit.cuts.clear();
        return Ok(());
    }
    unit.status = if unknown {
        TokenStatus::Unknown
    } else {
        TokenStatus::Morphological
    };
    unit.cuts.clear();
    unit.cuts.reserve(relative.len());
    for cut in relative {
        unit.cuts.push(
            unit.span
                .start
                .checked_add(u64::from(*cut))
                .ok_or(TokenizerError::LengthOverflow("flat absolute cut"))?,
        );
    }
    if unit.kind == LexicalKind::Number {
        unit.cuts.extend(numeric_micro_cuts(raw, unit.span)?);
        unit.cuts.sort_unstable();
        unit.cuts.dedup();
    }
    Ok(())
}

fn encode_flat_surface_range(
    tokenizer: &Tokenizer<'_>,
    inputs: &[Vec<u8>],
    newline_flags: &[bool],
    range: Range<usize>,
    vocabulary: &SurfaceVocabulary,
    use_morphology: bool,
    cache: &mut FlatAnalysisCache,
) -> Result<TrainingBatch, TokenizerError> {
    let document_count = range.len();
    let token_capacity = range.clone().try_fold(0_usize, |total, index| {
        total
            .checked_add(inputs[index].len().saturating_add(2))
            .ok_or(TokenizerError::LengthOverflow("surface batch capacity"))
    })?;
    let mut batch = TrainingBatch {
        ids: Vec::with_capacity(token_capacity),
        lengths: Vec::with_capacity(token_capacity),
        document_offsets: Vec::with_capacity(document_count.saturating_add(1)),
    };
    batch.document_offsets.push(0);
    for index in range {
        tokenizer.append_flat_surface_document(
            &inputs[index],
            newline_flags[index],
            vocabulary,
            use_morphology,
            cache,
            &mut batch,
        )?;
    }
    Ok(batch)
}

fn concatenate_surface_chunks(
    mut chunks: Vec<(usize, TrainingBatch)>,
) -> Result<TrainingBatch, TokenizerError> {
    chunks.sort_unstable_by_key(|entry| entry.0);
    let document_count = chunks.iter().try_fold(0_usize, |total, (_, batch)| {
        total
            .checked_add(batch.document_offsets.len().saturating_sub(1))
            .ok_or(TokenizerError::LengthOverflow("surface document count"))
    })?;
    let token_count = chunks.iter().try_fold(0_usize, |total, (_, batch)| {
        total
            .checked_add(batch.ids.len())
            .ok_or(TokenizerError::LengthOverflow("surface token count"))
    })?;
    let mut output = TrainingBatch {
        ids: Vec::with_capacity(token_count),
        lengths: Vec::with_capacity(token_count),
        document_offsets: Vec::with_capacity(document_count.saturating_add(1)),
    };
    output.document_offsets.push(0);
    for (_, mut batch) in chunks {
        if batch.ids.len() != batch.lengths.len() || batch.document_offsets.first() != Some(&0) {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "surface chunk metadata is inconsistent",
            ));
        }
        let base = u64::try_from(output.ids.len())
            .map_err(|_| TokenizerError::LengthOverflow("surface chunk base"))?;
        for offset in batch.document_offsets.into_iter().skip(1) {
            output.document_offsets.push(
                base.checked_add(offset)
                    .ok_or(TokenizerError::LengthOverflow("surface chunk offset"))?,
            );
        }
        output.ids.append(&mut batch.ids);
        output.lengths.append(&mut batch.lengths);
    }
    Ok(output)
}

fn concatenate_training_chunks(
    mut chunks: Vec<(usize, Vec<TrainingEncoding>)>,
) -> Result<TrainingBatch, TokenizerError> {
    chunks.sort_unstable_by_key(|entry| entry.0);
    let document_count = chunks.iter().try_fold(0_usize, |total, (_, documents)| {
        total
            .checked_add(documents.len())
            .ok_or(TokenizerError::LengthOverflow("flat document count"))
    })?;
    let token_count = chunks.iter().try_fold(0_usize, |total, (_, documents)| {
        documents.iter().try_fold(total, |subtotal, document| {
            subtotal
                .checked_add(document.ids.len())
                .ok_or(TokenizerError::LengthOverflow("flat token count"))
        })
    })?;
    let mut batch = TrainingBatch {
        ids: Vec::with_capacity(token_count),
        lengths: Vec::with_capacity(token_count),
        document_offsets: Vec::with_capacity(
            document_count
                .checked_add(1)
                .ok_or(TokenizerError::LengthOverflow("flat offset count"))?,
        ),
    };
    batch.document_offsets.push(0);
    for (_, documents) in chunks {
        for mut document in documents {
            if document.ids.len() != document.lengths.len() {
                return Err(TokenizerError::InvalidTrainingEncoding(
                    "flat document ID and length counts differ",
                ));
            }
            batch.ids.append(&mut document.ids);
            batch.lengths.append(&mut document.lengths);
            batch.document_offsets.push(
                u64::try_from(batch.ids.len())
                    .map_err(|_| TokenizerError::LengthOverflow("flat token offset"))?,
            );
        }
    }
    Ok(batch)
}

fn build_batch_ranges(
    inputs: &[Vec<u8>],
    threads: usize,
) -> Result<Vec<Range<usize>>, TokenizerError> {
    let total_bytes = inputs.iter().try_fold(0_usize, |total, input| {
        total
            .checked_add(input.len())
            .ok_or(TokenizerError::LengthOverflow("batch bytes"))
    })?;
    let desired_chunks = threads.saturating_mul(BATCH_CHUNKS_PER_WORKER).max(1);
    let target_bytes = total_bytes
        .div_ceil(desired_chunks)
        .max(MIN_BATCH_CHUNK_BYTES);
    let mut ranges = Vec::new();
    let mut start = 0_usize;
    let mut bytes = 0_usize;
    for (index, input) in inputs.iter().enumerate() {
        bytes = bytes
            .checked_add(input.len())
            .ok_or(TokenizerError::LengthOverflow("batch chunk bytes"))?;
        let end = index + 1;
        if bytes >= target_bytes && end < inputs.len() {
            ranges.push(start..end);
            start = end;
            bytes = 0;
        }
    }
    ranges.push(start..inputs.len());
    Ok(ranges)
}

fn flat_candidates_output_invariant(relative_cuts: &[Vec<u32>], unknown: &[bool]) -> bool {
    let Some(first_cuts) = relative_cuts.first() else {
        return false;
    };
    let Some(&first_unknown) = unknown.first() else {
        return false;
    };
    relative_cuts
        .iter()
        .zip(unknown)
        .all(|(cuts, candidate_unknown)| cuts == first_cuts && *candidate_unknown == first_unknown)
}

fn build_flat_analysis_set(
    token: &str,
    analyses: Vec<NativeAnalysis>,
) -> Result<FlatAnalysisSet, TokenizerError> {
    let numeric = token.chars().any(char::is_numeric);
    let mut relative_cuts = Vec::new();
    let mut ambiguity = Vec::new();
    let mut unknown = Vec::new();
    for analysis in analyses {
        let alignment = if numeric {
            let mut aligned_analysis = analysis.clone();
            complete_numeric_derivation(token, &mut aligned_analysis);
            align_analysis(token, 0, &aligned_analysis)
        } else {
            align_analysis(token, 0, &analysis)
        };
        let aligned = match alignment {
            Ok(value) => value,
            Err(TokenizerError::AlignmentMismatch { .. }) => continue,
            Err(error) => return Err(error),
        };
        let cuts = aligned
            .cuts
            .into_iter()
            .map(|cut| {
                u32::try_from(cut).map_err(|_| TokenizerError::LengthOverflow("flat relative cut"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        relative_cuts.push(cuts);
        ambiguity.push(ambiguity_word_data(&analysis));
        unknown.push(analysis.dictionary_id == "UNK_Unk_Unk");
    }
    if relative_cuts.is_empty() {
        let unknown_candidate = unknown_native_analysis(token);
        relative_cuts.push(Vec::new());
        ambiguity.push(ambiguity_word_data(&unknown_candidate));
        unknown.push(true);
    }
    if relative_cuts.len() != ambiguity.len() || ambiguity.len() != unknown.len() {
        return Err(TokenizerError::InvalidTrainingEncoding(
            "flat candidate metadata cardinalities differ",
        ));
    }
    let output_invariant = flat_candidates_output_invariant(&relative_cuts, &unknown);
    Ok(FlatAnalysisSet {
        relative_cuts: relative_cuts.into_boxed_slice(),
        ambiguity: ambiguity.into_boxed_slice(),
        scoring_codes: Box::new([]),
        unknown: unknown.into_boxed_slice(),
        output_invariant,
    })
}

fn surface_valid_candidates(
    token: &str,
    absolute_start: u64,
    analyses: Vec<NativeAnalysis>,
) -> Vec<NativeAnalysis> {
    let mut valid = analyses
        .into_iter()
        .filter(|analysis| {
            if analysis.primary_pos == "Num" {
                let mut candidate = analysis.clone();
                complete_numeric_derivation(token, &mut candidate);
                align_analysis(token, absolute_start, &candidate).is_ok()
            } else {
                align_analysis(token, absolute_start, analysis).is_ok()
            }
        })
        .collect::<Vec<_>>();
    if valid.is_empty() {
        valid.push(unknown_native_analysis(token));
    }
    valid
}

fn unknown_native_analysis(token: &str) -> NativeAnalysis {
    NativeAnalysis {
        canonical: format!("UNK_Unk_Unk\u{1}Unknown={token}\u{2}"),
        dictionary_id: "UNK_Unk_Unk".to_owned(),
        lemma: token.to_owned(),
        primary_pos: "Unk".to_owned(),
        secondary_pos: "Unk".to_owned(),
        surface_form: token.to_owned(),
        stem: token.to_owned(),
        ending: String::new(),
        morphemes: vec![NativeMorpheme {
            id: "Unknown".to_owned(),
            name: "Unknown".to_owned(),
            surface: token.to_owned(),
            derivational: false,
            informal: false,
            pos: None,
            mapped_id: None,
        }],
    }
}
fn apply_selected_analysis(
    raw: &[u8],
    unit: &mut TokenizedUnit,
    analysis: NativeAnalysis,
) -> Result<(), TokenizerError> {
    apply_analysis(raw, unit, analysis)?;
    if matches!(unit.kind, LexicalKind::Punctuation | LexicalKind::Symbol) {
        unit.status = TokenStatus::Structural;
        unit.cuts.clear();
        unit.analysis = None;
        return Ok(());
    }
    if unit.kind == LexicalKind::Number {
        let mut cuts = numeric_micro_cuts(raw, unit.span)?;
        cuts.extend_from_slice(&unit.cuts);
        cuts.sort_unstable();
        cuts.dedup();
        unit.cuts = cuts;
    }
    Ok(())
}
fn split_compact_units(
    lexical_scan: &CompactScanResult,
    code_spans: &[ByteSpan],
    requested: TokenizerMode,
) -> Result<Vec<TokenizedUnit>, TokenizerError> {
    let spans = lexical_scan.spans();
    let kinds = lexical_scan.kinds();
    if spans.len() != kinds.len() {
        return Err(TokenizerError::Scan(ScanError::MetadataLengthMismatch {
            unit_count: spans.len(),
            kind_count: kinds.len(),
        }));
    }
    if code_spans.is_empty() {
        return split_compact_units_without_code_spans(lexical_scan, requested);
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
                    "compact split source unit is absent",
                ))?;
        if span.start < source.start || span.end > source.end {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "compact split span crosses a lexical unit",
            ));
        }
        let kind = kinds[source_index];
        let mode = effective_mode(span, kind, code_spans, requested);
        let status = match mode {
            TokenMode::Code => TokenStatus::Code,
            TokenMode::Opaque => TokenStatus::Opaque,
            TokenMode::Turkish => TokenStatus::Structural,
        };
        units.push(TokenizedUnit {
            span,
            kind,
            mode,
            status,
            group_id: None,
            cuts: Vec::new(),
            analysis: None,
        });
    }
    Ok(units)
}

fn split_compact_units_without_code_spans(
    lexical_scan: &CompactScanResult,
    requested: TokenizerMode,
) -> Result<Vec<TokenizedUnit>, TokenizerError> {
    let spans = lexical_scan.spans();
    let kinds = lexical_scan.kinds();
    let mut units = Vec::with_capacity(spans.len());
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
        let status = match mode {
            TokenMode::Code => TokenStatus::Code,
            TokenMode::Opaque => TokenStatus::Opaque,
            TokenMode::Turkish => TokenStatus::Structural,
        };
        for window in boundaries.windows(2) {
            units.push(TokenizedUnit {
                span: ByteSpan {
                    start: window[0],
                    end: window[1],
                },
                kind,
                mode,
                status,
                group_id: None,
                cuts: Vec::new(),
                analysis: None,
            });
        }
    }
    Ok(units)
}

fn split_units(
    lexical_scan: &nedo_core::ScanResult,
    code_spans: &[ByteSpan],
    requested: TokenizerMode,
) -> Result<Vec<TokenizedUnit>, TokenizerError> {
    if code_spans.is_empty() {
        return split_units_without_code_spans(lexical_scan, requested);
    }
    let mut boundaries = Vec::new();
    for (index, unit) in lexical_scan.document().units().iter().enumerate() {
        boundaries.push(unit.span.start);
        boundaries.push(unit.span.end);
        let kind = lexical_scan.kind(index)?;
        let mode = effective_mode(unit.span, kind, code_spans, requested);
        if kind == LexicalKind::Punctuation {
            add_scalar_boundaries(
                lexical_scan.unit_bytes(index)?,
                unit.span.start,
                &mut boundaries,
            )?;
        } else if mode == TokenMode::Turkish && kind == LexicalKind::Symbol {
            let bytes = lexical_scan.unit_bytes(index)?;
            if bytes.iter().all(u8::is_ascii_punctuation) {
                add_scalar_boundaries(bytes, unit.span.start, &mut boundaries)?;
            }
        }
        if mode == TokenMode::Code && matches!(kind, LexicalKind::Word | LexicalKind::Number) {
            let text = std::str::from_utf8(lexical_scan.unit_bytes(index)?)
                .map_err(|_| TokenizerError::InvalidUtf8Unit)?;
            for relative in identifier_cuts(text) {
                boundaries.push(
                    unit.span.start
                        + u64::try_from(relative)
                            .map_err(|_| TokenizerError::LengthOverflow("identifier boundary"))?,
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
    let mut units = Vec::new();
    let mut source_index = 0_usize;
    for window in boundaries.windows(2) {
        let unit_span = ByteSpan {
            start: window[0],
            end: window[1],
        };
        while lexical_scan.document().units()[source_index].span.end <= unit_span.start {
            source_index += 1;
        }
        let kind = lexical_scan.kind(source_index)?;
        let mode = effective_mode(unit_span, kind, code_spans, requested);
        let status = match mode {
            TokenMode::Code => TokenStatus::Code,
            TokenMode::Opaque => TokenStatus::Opaque,
            TokenMode::Turkish => TokenStatus::Structural,
        };
        units.push(TokenizedUnit {
            span: unit_span,
            kind,
            mode,
            status,
            group_id: None,
            cuts: Vec::new(),
            analysis: None,
        });
    }
    Ok(units)
}

fn split_units_without_code_spans(
    lexical_scan: &nedo_core::ScanResult,
    requested: TokenizerMode,
) -> Result<Vec<TokenizedUnit>, TokenizerError> {
    let source_units = lexical_scan.document().units();
    let kinds = lexical_scan.kinds();
    if source_units.len() != kinds.len() {
        return Err(TokenizerError::Scan(ScanError::MetadataLengthMismatch {
            unit_count: source_units.len(),
            kind_count: kinds.len(),
        }));
    }
    let mut units = Vec::with_capacity(source_units.len());
    let mut boundaries = Vec::with_capacity(8);
    for (index, (source, kind)) in source_units.iter().zip(kinds.iter().copied()).enumerate() {
        let mode = effective_mode(source.span, kind, &[], requested);
        boundaries.clear();
        boundaries.push(source.span.start);
        boundaries.push(source.span.end);
        let bytes = lexical_scan.unit_bytes(index)?;
        if kind == LexicalKind::Punctuation {
            add_scalar_boundaries(bytes, source.span.start, &mut boundaries)?;
        } else if mode == TokenMode::Turkish
            && kind == LexicalKind::Symbol
            && bytes.iter().all(u8::is_ascii_punctuation)
        {
            add_scalar_boundaries(bytes, source.span.start, &mut boundaries)?;
        }
        if mode == TokenMode::Code && matches!(kind, LexicalKind::Word | LexicalKind::Number) {
            let text = std::str::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8Unit)?;
            for relative in identifier_cuts(text) {
                boundaries.push(
                    source
                        .span
                        .start
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
        let status = match mode {
            TokenMode::Code => TokenStatus::Code,
            TokenMode::Opaque => TokenStatus::Opaque,
            TokenMode::Turkish => TokenStatus::Structural,
        };
        for window in boundaries.windows(2) {
            units.push(TokenizedUnit {
                span: ByteSpan {
                    start: window[0],
                    end: window[1],
                },
                kind,
                mode,
                status,
                group_id: None,
                cuts: Vec::new(),
                analysis: None,
            });
        }
    }
    Ok(units)
}

fn add_scalar_boundaries(
    bytes: &[u8],
    absolute_start: u64,
    boundaries: &mut Vec<u64>,
) -> Result<(), TokenizerError> {
    let text = std::str::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8Unit)?;
    for (relative, _) in text.char_indices().skip(1) {
        boundaries.push(
            absolute_start
                + u64::try_from(relative)
                    .map_err(|_| TokenizerError::LengthOverflow("structural boundary"))?,
        );
    }
    Ok(())
}

fn effective_mode(
    span: ByteSpan,
    kind: LexicalKind,
    code_spans: &[ByteSpan],
    requested: TokenizerMode,
) -> TokenMode {
    if matches!(kind, LexicalKind::Opaque | LexicalKind::Control) {
        return TokenMode::Opaque;
    }
    match requested {
        TokenizerMode::Code => TokenMode::Code,
        TokenizerMode::Turkish => TokenMode::Turkish,
        TokenizerMode::Auto => {
            if code_spans
                .iter()
                .any(|code| span.start >= code.start && span.end <= code.end)
            {
                TokenMode::Code
            } else {
                TokenMode::Turkish
            }
        }
    }
}

fn apply_analysis(
    raw: &[u8],
    unit: &mut TokenizedUnit,
    mut analysis: NativeAnalysis,
) -> Result<(), TokenizerError> {
    let token = unit_str(raw, unit.span)?;
    if analysis.dictionary_id == "UNK_Unk_Unk" {
        unit.status = TokenStatus::Unknown;
        unit.analysis = Some(AnalysisMetadata {
            canonical: analysis.canonical,
            dictionary_id: analysis.dictionary_id,
            lemma: analysis.lemma,
            primary_pos: analysis.primary_pos,
            secondary_pos: analysis.secondary_pos,
            morphemes: vec![AlignedMorpheme {
                id: "Unknown".to_owned(),
                surface: token.to_owned(),
                span: unit.span,
                derivational: false,
            }],
        });
        return Ok(());
    }
    complete_numeric_derivation(token, &mut analysis);
    let aligned = align_analysis(token, unit.span.start, &analysis)?;
    unit.cuts = aligned.cuts;
    unit.status = TokenStatus::Morphological;
    unit.analysis = Some(AnalysisMetadata {
        canonical: analysis.canonical,
        dictionary_id: analysis.dictionary_id,
        lemma: analysis.lemma,
        primary_pos: analysis.primary_pos,
        secondary_pos: analysis.secondary_pos,
        morphemes: aligned.morphemes,
    });
    Ok(())
}

fn numeric_micro_cuts(raw: &[u8], span: ByteSpan) -> Result<Vec<u64>, TokenizerError> {
    let text = unit_str(raw, span)?;
    let chars = text.char_indices().collect::<Vec<_>>();
    let mut cuts = Vec::new();
    for (position, &(index, value)) in chars.iter().enumerate() {
        if !matches!(value, '.' | ',' | ':' | '/' | '-') {
            continue;
        }
        let previous_is_digit = position > 0 && chars[position - 1].1.is_numeric();
        let next_is_digit = chars
            .get(position + 1)
            .is_some_and(|entry| entry.1.is_numeric());
        if previous_is_digit && next_is_digit {
            let start = span.start
                + u64::try_from(index)
                    .map_err(|_| TokenizerError::LengthOverflow("numeric separator start"))?;
            let end = start
                + u64::try_from(value.len_utf8())
                    .map_err(|_| TokenizerError::LengthOverflow("numeric separator end"))?;
            cuts.push(start);
            cuts.push(end);
        }
    }
    cuts.sort_unstable();
    cuts.dedup();
    Ok(cuts)
}

fn split_long_flat_units(
    raw: &[u8],
    units: Vec<TokenizedUnit>,
    maximum_chars: usize,
) -> Result<Vec<TokenizedUnit>, TokenizerError> {
    let mut requires_split = false;
    for unit in &units {
        let should_split = matches!(unit.status, TokenStatus::Unknown | TokenStatus::Code)
            || unit.mode == TokenMode::Opaque;
        if !should_split {
            continue;
        }
        let bytes = unit_bytes(raw, unit.span)?;
        let limit = maximum_chars;
        if unit.mode == TokenMode::Opaque {
            requires_split = bytes.len() > limit;
        } else if bytes.len() > limit {
            let text = std::str::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8Unit)?;
            requires_split = text.chars().nth(limit).is_some();
        }
        if requires_split {
            break;
        }
    }
    if !requires_split {
        return Ok(units);
    }
    let mut output = Vec::with_capacity(units.len());
    for unit in units {
        let should_split = matches!(unit.status, TokenStatus::Unknown | TokenStatus::Code)
            || unit.mode == TokenMode::Opaque;
        if !should_split {
            output.push(unit);
            continue;
        }
        let limit = maximum_chars;
        let spans = chunk_span(raw, unit.span, limit, unit.mode == TokenMode::Opaque)?;
        if spans.len() == 1 {
            output.push(unit);
            continue;
        }
        for span in spans {
            output.push(TokenizedUnit {
                span,
                kind: unit.kind,
                mode: unit.mode,
                status: unit.status,
                group_id: None,
                cuts: Vec::new(),
                analysis: None,
            });
        }
    }
    Ok(output)
}

fn split_long_fallback_units(
    raw: &[u8],
    units: Vec<TokenizedUnit>,
    maximum_chars: usize,
) -> Result<Vec<TokenizedUnit>, TokenizerError> {
    let mut output = Vec::with_capacity(units.len());
    for unit in units {
        let should_split = matches!(unit.status, TokenStatus::Unknown | TokenStatus::Code)
            || unit.mode == TokenMode::Opaque;
        if !should_split {
            output.push(unit);
            continue;
        }
        let limit = maximum_chars;
        let spans = chunk_span(raw, unit.span, limit, unit.mode == TokenMode::Opaque)?;
        if spans.len() == 1 {
            output.push(unit);
            continue;
        }
        for span in spans {
            let analysis = if unit.status == TokenStatus::Unknown {
                Some(unknown_chunk_analysis(raw, span)?)
            } else {
                None
            };
            output.push(TokenizedUnit {
                span,
                kind: unit.kind,
                mode: unit.mode,
                status: unit.status,
                group_id: None,
                cuts: Vec::new(),
                analysis,
            });
        }
    }
    Ok(output)
}

fn chunk_span(
    raw: &[u8],
    span: ByteSpan,
    maximum: usize,
    byte_general: bool,
) -> Result<Vec<ByteSpan>, TokenizerError> {
    let bytes = unit_bytes(raw, span)?;
    if byte_general {
        if bytes.len() <= maximum {
            return Ok(vec![span]);
        }
        let mut result = Vec::new();
        let mut start = span.start;
        for chunk in bytes.chunks(maximum) {
            let end = start
                .checked_add(
                    u64::try_from(chunk.len())
                        .map_err(|_| TokenizerError::LengthOverflow("opaque fallback chunk"))?,
                )
                .ok_or(TokenizerError::LengthOverflow("opaque fallback end"))?;
            result.push(ByteSpan { start, end });
            start = end;
        }
        return Ok(result);
    }
    let text = std::str::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8Unit)?;
    if text.chars().count() <= maximum {
        return Ok(vec![span]);
    }
    let mut result = Vec::new();
    let mut chunk_start = 0_usize;
    let mut count = 0_usize;
    for (index, _) in text.char_indices() {
        if count == maximum {
            result.push(ByteSpan {
                start: span.start
                    + u64::try_from(chunk_start)
                        .map_err(|_| TokenizerError::LengthOverflow("fallback chunk start"))?,
                end: span.start
                    + u64::try_from(index)
                        .map_err(|_| TokenizerError::LengthOverflow("fallback chunk end"))?,
            });
            chunk_start = index;
            count = 0;
        }
        count += 1;
    }
    if chunk_start < text.len() {
        result.push(ByteSpan {
            start: span.start
                + u64::try_from(chunk_start)
                    .map_err(|_| TokenizerError::LengthOverflow("fallback final start"))?,
            end: span.end,
        });
    }
    Ok(result)
}

fn unknown_chunk_analysis(raw: &[u8], span: ByteSpan) -> Result<AnalysisMetadata, TokenizerError> {
    let surface = unit_str(raw, span)?.to_owned();
    Ok(AnalysisMetadata {
        canonical: format!("UNK_Unk_Unk\u{1}Unknown={surface}\u{2}"),
        dictionary_id: "UNK_Unk_Unk".to_owned(),
        lemma: surface.clone(),
        primary_pos: "Unk".to_owned(),
        secondary_pos: "Unk".to_owned(),
        morphemes: vec![AlignedMorpheme {
            id: "Unknown".to_owned(),
            surface,
            span,
            derivational: false,
        }],
    })
}

fn complete_numeric_derivation(token: &str, analysis: &mut NativeAnalysis) {
    if analysis.primary_pos != "Num" {
        return;
    }
    let Some((stem, suffix)) = split_apostrophe_suffix(token) else {
        return;
    };
    if analysis.stem != stem {
        return;
    }
    let Some(normalized_suffix) = turkish_lowercase_single_scalar(suffix) else {
        return;
    };
    let Some(derivation) = numeric_derivation_prefix(&normalized_suffix) else {
        return;
    };
    if analysis
        .morphemes
        .iter()
        .any(|morpheme| morpheme.id == derivation.id)
    {
        return;
    }
    let derivation_char_count = derivation.surface.chars().count();
    let original_prefix_end = suffix
        .char_indices()
        .nth(derivation_char_count)
        .map_or(suffix.len(), |(index, _)| index);
    let original_remaining = &suffix[original_prefix_end..];
    let Some(normalized_remaining) = turkish_lowercase_single_scalar(original_remaining) else {
        return;
    };
    let Some(normalized_analysis_ending) = turkish_lowercase_single_scalar(&analysis.ending) else {
        return;
    };
    if normalized_analysis_ending != normalized_remaining {
        return;
    }
    let old_ending = analysis.ending.clone();
    analysis.ending.clear();
    analysis.ending.push_str(derivation.surface);
    analysis.ending.push_str(&old_ending);
    analysis.surface_form.clear();
    analysis.surface_form.push_str(stem);
    analysis.surface_form.push_str(&analysis.ending);
    analysis.morphemes.insert(
        1,
        NativeMorpheme {
            id: derivation.id.to_owned(),
            name: derivation.name.to_owned(),
            surface: derivation.surface.to_owned(),
            derivational: true,
            informal: false,
            pos: Some(derivation.pos.to_owned()),
            mapped_id: None,
        },
    );
    analysis.canonical = tokenizer_canonical_key(&analysis.dictionary_id, &analysis.morphemes);
}

#[derive(Clone, Copy)]
struct NumericDerivation<'a> {
    id: &'static str,
    name: &'static str,
    pos: &'static str,
    surface: &'a str,
}

fn numeric_derivation_prefix(suffix: &str) -> Option<NumericDerivation<'_>> {
    const ORDINALS: [&str; 8] = ["ıncı", "inci", "uncu", "üncü", "ncı", "nci", "ncu", "ncü"];
    const DISTRIBUTIVES: [&str; 4] = ["şar", "şer", "ar", "er"];
    if let Some(surface) = ORDINALS
        .into_iter()
        .find(|candidate| suffix.starts_with(candidate))
    {
        return Some(NumericDerivation {
            id: "Ord",
            name: "Ordinal",
            pos: "Adj",
            surface,
        });
    }
    DISTRIBUTIVES
        .into_iter()
        .find(|candidate| suffix.starts_with(candidate))
        .map(|surface| NumericDerivation {
            id: "Dist",
            name: "Distribution",
            pos: "Num",
            surface,
        })
}

fn turkish_lowercase_single_scalar(value: &str) -> Option<String> {
    let mut output = String::with_capacity(value.len());
    for character in value.chars() {
        let lowered = match character {
            'I' => 'ı',
            'İ' => 'i',
            _ => {
                let mut lowercase = character.to_lowercase();
                let first = lowercase.next()?;
                if lowercase.next().is_some() {
                    return None;
                }
                first
            }
        };
        output.push(lowered);
    }
    Some(output)
}

fn tokenizer_canonical_key(dictionary_id: &str, morphemes: &[NativeMorpheme]) -> String {
    let mut canonical = String::with_capacity(dictionary_id.len() + morphemes.len() * 8);
    canonical.push_str(dictionary_id);
    canonical.push('\u{1}');
    for morpheme in morphemes {
        canonical.push_str(&morpheme.id);
        canonical.push('=');
        canonical.push_str(&morpheme.surface);
        canonical.push('\u{2}');
    }
    canonical
}

fn split_apostrophe_suffix(token: &str) -> Option<(&str, &str)> {
    for (index, value) in token.char_indices() {
        if matches!(
            value,
            '\'' | '\u{2018}' | '\u{2019}' | '\u{02bc}' | '\u{ff07}'
        ) {
            let end = index + value.len_utf8();
            if index > 0 && end < token.len() {
                return Some((&token[..index], &token[end..]));
            }
            return None;
        }
    }
    None
}

fn assign_inner_groups(raw: &[u8], units: &mut [TokenizedUnit]) -> Result<(), TokenizerError> {
    let mut next_group = 0_u32;
    let mut previous_content: Option<usize> = None;
    for index in 0..units.len() {
        if units[index].mode != TokenMode::Turkish {
            previous_content = None;
            continue;
        }
        if matches!(units[index].kind, LexicalKind::Whitespace) {
            continue;
        }
        if matches!(units[index].kind, LexicalKind::LineBreak) {
            previous_content = None;
            continue;
        }
        let token = unit_str(raw, units[index].span)?;
        if is_phonological_clitic(token) {
            if let Some(previous) = previous_content {
                let group = units[previous].group_id.unwrap_or(next_group);
                if units[previous].group_id.is_none() {
                    next_group = next_group
                        .checked_add(1)
                        .ok_or(TokenizerError::LengthOverflow("group id"))?;
                    units[previous].group_id = Some(group);
                }
                for unit in &mut units[previous + 1..=index] {
                    unit.group_id = Some(group);
                }
                previous_content = Some(index);
                continue;
            }
        }
        units[index].group_id = Some(next_group);
        next_group = next_group
            .checked_add(1)
            .ok_or(TokenizerError::LengthOverflow("group id"))?;
        previous_content = Some(index);
    }
    Ok(())
}

fn is_phonological_clitic(token: &str) -> bool {
    matches!(
        token.to_lowercase().as_str(),
        "mi" | "mı" | "mu" | "mü" | "de" | "da"
    )
}

fn is_sentence_boundary(
    kind: LexicalKind,
    raw: &[u8],
    span: ByteSpan,
) -> Result<bool, TokenizerError> {
    if kind != LexicalKind::Punctuation {
        return Ok(false);
    }
    let bytes = unit_bytes(raw, span)?;
    Ok(matches!(bytes, b"." | b"!" | b"?"))
}

fn unit_str(raw: &[u8], span: ByteSpan) -> Result<&str, TokenizerError> {
    std::str::from_utf8(unit_bytes(raw, span)?).map_err(|_| TokenizerError::InvalidUtf8Unit)
}

fn unit_bytes(raw: &[u8], span: ByteSpan) -> Result<&[u8], TokenizerError> {
    let start =
        usize::try_from(span.start).map_err(|_| TokenizerError::LengthOverflow("unit start"))?;
    let end = usize::try_from(span.end).map_err(|_| TokenizerError::LengthOverflow("unit end"))?;
    raw.get(start..end)
        .ok_or(TokenizerError::UnitOutsideDocument)
}

fn verify_sha(
    bytes: &[u8],
    expected: &'static str,
    asset: &'static str,
) -> Result<(), TokenizerError> {
    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual != expected {
        return Err(TokenizerError::AssetChecksumMismatch {
            asset,
            expected,
            actual,
        });
    }
    Ok(())
}

#[cfg(feature = "compiled-surface-table")]
fn hex_bytes(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

/// Tokenizer failure; every failure is explicit and inspectable.
#[derive(Debug)]
pub enum TokenizerError {
    /// Low-level scanner failure.
    Scan(ScanError),
    /// Lossless format failure.
    Format(FormatError),
    /// Native morphology binary/runtime failure.
    Morphology(BinaryError),
    /// Native contextual model failure.
    Disambiguation(DisambiguationError),
    /// Embedded or supplied asset checksum mismatch.
    AssetChecksumMismatch {
        asset: &'static str,
        expected: &'static str,
        actual: String,
    },
    /// Tokenizer codec asset identity differs.
    AssetIdentityMismatch,
    /// Morphological surface could not be mapped to original bytes exactly.
    AlignmentMismatch {
        token: String,
        expected: String,
        actual: String,
        canonical: String,
    },
    /// A lowercase operation expanded into multiple scalars and is unsupported in schema v1.
    UnsupportedNormalization { value: char },
    /// Length arithmetic or representation overflow.
    LengthOverflow(&'static str),
    /// Invalid configuration.
    InvalidConfiguration(&'static str),
    /// Worker thread panicked.
    WorkerPanicked,
    /// Unit did not cover the required interval.
    InvalidUnitCoverage {
        index: usize,
        expected: u64,
        start: u64,
        end: u64,
    },
    /// Units did not cover the full document.
    IncompleteDocument { covered: u64, document_len: u64 },
    /// Analysis-required status lacks metadata.
    MissingAnalysis { index: usize },
    /// Structural/code/opaque status unexpectedly carries morphology.
    UnexpectedAnalysis { index: usize },
    /// Morpheme span escapes its unit.
    MorphemeOutsideUnit { index: usize },
    /// Unit mode/status/analysis metadata are semantically inconsistent.
    InvalidUnitMetadata { index: usize, reason: &'static str },
    /// Unit span cannot be sliced.
    UnitOutsideDocument,
    /// Turkish-mode unit was not valid UTF-8.
    InvalidUtf8Unit,
    /// Decoder output length differed from contextual input.
    ContextLengthMismatch,
    /// Tokenized codec magic is invalid.
    BadCodecMagic,
    /// Tokenized codec schema is unsupported.
    UnsupportedCodecVersion(u32),
    /// Tokenized codec was truncated.
    TruncatedCodec,
    /// Tokenized codec payload checksum failed.
    CodecChecksumMismatch,
    /// Tokenized codec count cannot fit in remaining bytes.
    ImpossibleCodecCount(&'static str),
    /// Tokenized codec has trailing bytes.
    TrailingCodecBytes(usize),
    /// Tokenized codec contains an invalid enum value.
    InvalidCodecEnum(&'static str, u8),
    /// Tokenized codec contains a non-canonical boolean byte.
    InvalidCodecBoolean { field: &'static str, value: u8 },
    /// Tokenized codec contains invalid UTF-8 metadata.
    InvalidCodecUtf8,
    /// Vocabulary identity/content failure.
    InvalidVocabulary(&'static str),
    /// Rich-to-training conversion violated a flat-stream invariant.
    InvalidTrainingEncoding(&'static str),
    /// A compiled table did not match an independently trusted payload-digest seal.
    CompiledSurfaceTableSealMismatch { expected: String, actual: String },
    /// A compiled invariant surface entry failed exact startup verification.
    InvalidCompiledSurfaceTable {
        surface: String,
        reason: &'static str,
    },
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for TokenizerError {}

impl From<ScanError> for TokenizerError {
    fn from(value: ScanError) -> Self {
        Self::Scan(value)
    }
}

impl From<FormatError> for TokenizerError {
    fn from(value: FormatError) -> Self {
        Self::Format(value)
    }
}

impl From<BinaryError> for TokenizerError {
    fn from(value: BinaryError) -> Self {
        Self::Morphology(value)
    }
}

impl From<DisambiguationError> for TokenizerError {
    fn from(value: DisambiguationError) -> Self {
        Self::Disambiguation(value)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        decode_tokenized, encode_tokenized, CharacterVocabulary, SurfaceEncoderOptions,
        SurfaceVocabulary, SurfaceVocabularyKind, TokenMode, TokenStatus, Tokenizer, TokenizerConfig,
        TokenizerMode, TrainingEncodingOptions,
    };

    #[test]
    fn embedded_assets_tokenize_and_round_trip() -> Result<(), super::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let raw = "Ankara'da evlerimizden mi?\r\n```rust\nfn main() {}\n``` 😀"
            .as_bytes()
            .to_vec();
        let document = tokenizer.tokenize(raw.clone())?;
        assert_eq!(document.decode(), raw);
        assert!(document
            .units()
            .iter()
            .any(|unit| unit.status == TokenStatus::Morphological));
        assert!(document
            .units()
            .iter()
            .any(|unit| unit.mode == TokenMode::Code));
        let encoded = encode_tokenized(&document)?;
        assert_eq!(decode_tokenized(&encoded)?, document);
        Ok(())
    }

    #[test]
    fn production_surface_vocab_preserves_turkish_morphology_boundaries(
    ) -> Result<(), super::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let vocabulary = SurfaceVocabulary::from_bytes(include_bytes!(
            "../../../assets/surface-vocab.bin"
        ))?;
        assert_eq!(vocabulary.kind(), SurfaceVocabularyKind::ByteBpe);

        let raw = "Ananızı öpeyim".as_bytes().to_vec();
        let document = tokenizer.tokenize(raw.clone())?;
        assert_eq!(document.units()[0].cuts, vec![3, 7]);
        assert_eq!(document.units()[2].cuts, vec![13, 14]);

        let encoded = vocabulary.encode_document(&document, false)?;
        let mut cursor = 0_usize;
        let mut pieces = Vec::new();
        for &length in &encoded.lengths {
            if length == 0 {
                continue;
            }
            let end = cursor + usize::from(length);
            pieces.push(std::str::from_utf8(&raw[cursor..end]).expect("valid UTF-8 piece"));
            cursor = end;
        }
        assert_eq!(pieces, ["Ana", "nız", "ı", " öp", "e", "yim"]);
        assert_eq!(vocabulary.decode_ids(&encoded.ids)?, raw);
        Ok(())
    }

    #[test]
    fn clock_suffix_is_morphological_in_full_pipeline() -> Result<(), super::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let token = "14:30:05'te";
        let analyses = super::analyze_token_with_quality_fallback(&tokenizer.morphology, token)?;
        assert_eq!(analyses.len(), 1);
        assert_ne!(analyses[0].dictionary_id, "UNK_Unk_Unk");
        let set = super::build_flat_analysis_set(token, analyses)?;
        assert_eq!(set.unknown.as_ref(), &[false]);
        let document = tokenizer.tokenize(token.as_bytes().to_vec())?;
        assert_eq!(document.units().len(), 1);
        assert_eq!(document.units()[0].status, TokenStatus::Morphological);
        assert_eq!(document.decode(), token.as_bytes());
        Ok(())
    }

    #[test]
    fn embedded_informal_morphology_produces_surface_aligned_cuts(
    ) -> Result<(), super::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        for (surface, expected) in [
            ("gidicem", ["gid", "ice", "m"]),
            ("geliyom", ["gel", "iyo", "m"]),
            ("yapıyon", ["yap", "ıyo", "n"]),
            ("yapıyosun", ["yap", "ıyo", "sun"]),
            ("yazacam", ["yaz", "aca", "m"]),
            ("yazıcam", ["yaz", "ıca", "m"]),
        ] {
            let document = tokenizer.tokenize(surface.as_bytes().to_vec())?;
            assert_eq!(document.decode(), surface.as_bytes());
            assert_eq!(document.units().len(), 1);
            let unit = &document.units()[0];
            assert_eq!(unit.status, TokenStatus::Morphological);

            let mut boundaries = Vec::with_capacity(unit.cuts.len() + 2);
            boundaries.push(unit.span.start);
            boundaries.extend(unit.cuts.iter().copied());
            boundaries.push(unit.span.end);
            let pieces = boundaries
                .windows(2)
                .map(|pair| {
                    let start = usize::try_from(pair[0]).map_err(|_| {
                        super::TokenizerError::LengthOverflow("informal test start")
                    })?;
                    let end = usize::try_from(pair[1])
                        .map_err(|_| super::TokenizerError::LengthOverflow("informal test end"))?;
                    std::str::from_utf8(&document.raw()[start..end])
                        .map_err(|_| super::TokenizerError::InvalidUtf8Unit)
                })
                .collect::<Result<Vec<_>, _>>()?;
            assert_eq!(pieces.as_slice(), expected.as_slice());

            let analysis = unit
                .analysis
                .as_ref()
                .ok_or(super::TokenizerError::MissingAnalysis { index: 0 })?;
            assert!(analysis.canonical.contains("_Informal"));
        }
        Ok(())
    }

    #[test]
    fn technical_borrowings_remain_byte_exact_without_lexical_exceptions() -> Result<(), super::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let raw = "kanka deploy ettim ama loglar acayip".as_bytes().to_vec();
        let document = tokenizer.tokenize(raw.clone())?;
        assert_eq!(document.decode(), raw);
        Ok(())
    }

    #[test]
    fn auto_mode_detects_unmarked_code_without_reclassifying_turkish(
    ) -> Result<(), super::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let raw = "geliyom gidiyom.\nfoo_bar += 42;\nsonra dönerim."
            .as_bytes()
            .to_vec();
        let document = tokenizer.tokenize(raw.clone())?;
        assert_eq!(document.decode(), raw);
        let mut saw_code = false;
        let mut saw_informal_turkish = false;
        for unit in document.units() {
            let start = usize::try_from(unit.span.start)
                .map_err(|_| super::TokenizerError::LengthOverflow("auto-code test start"))?;
            let end = usize::try_from(unit.span.end)
                .map_err(|_| super::TokenizerError::LengthOverflow("auto-code test end"))?;
            let surface = std::str::from_utf8(&document.raw()[start..end])
                .map_err(|_| super::TokenizerError::InvalidUtf8Unit)?;
            if surface == "foo"
                || surface == "_"
                || surface == "bar"
                || surface == "+="
                || surface == "42"
            {
                assert_eq!(unit.mode, TokenMode::Code);
                assert_eq!(unit.status, TokenStatus::Code);
                saw_code = true;
            }
            if surface == "geliyom" || surface == "gidiyom" {
                assert_eq!(unit.mode, TokenMode::Turkish);
                assert_eq!(unit.status, TokenStatus::Morphological);
                saw_informal_turkish = true;
            }
        }
        assert!(saw_code);
        assert!(saw_informal_turkish);
        Ok(())
    }

    #[test]
    fn unmarked_code_detection_can_be_disabled_exactly() -> Result<(), super::TokenizerError> {
        let raw = b"foo_bar += 42".to_vec();
        let detected = Tokenizer::embedded(TokenizerConfig::default())?.tokenize(raw.clone())?;
        let legacy = Tokenizer::embedded(TokenizerConfig {
            detect_unmarked_code: false,
            ..TokenizerConfig::default()
        })?
        .tokenize(raw.clone())?;
        assert_eq!(detected.decode(), raw);
        assert_eq!(legacy.decode(), raw);
        assert!(detected
            .units()
            .iter()
            .all(|unit| unit.mode == TokenMode::Code));
        assert!(legacy
            .units()
            .iter()
            .any(|unit| unit.mode == TokenMode::Turkish));
        Ok(())
    }

    #[test]
    fn invalid_utf8_and_all_bytes_round_trip() -> Result<(), super::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig {
            mode: TokenizerMode::Turkish,
            ..TokenizerConfig::default()
        })?;
        let raw: Vec<u8> = (0_u8..=u8::MAX).collect();
        let document = tokenizer.tokenize(raw.clone())?;
        assert_eq!(document.decode(), raw);
        assert!(document
            .units()
            .iter()
            .any(|unit| unit.status == TokenStatus::Opaque));
        Ok(())
    }

    #[test]
    fn runtime_proper_unknown_characters_align_without_losing_bytes(
    ) -> Result<(), super::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        for (value, expected) in [("UŔK'A", "u?k"), ("UÁÑ'A", "uan"), ("Â", "a"), ("M1", "m1")]
        {
            let raw = value.as_bytes().to_vec();
            let document = tokenizer.tokenize(raw.clone())?;
            assert_eq!(document.decode(), raw);
            assert!(document.units().iter().any(|unit| {
                unit.analysis
                    .as_ref()
                    .is_some_and(|analysis| analysis.canonical.to_lowercase().contains(expected))
            }));
        }
        Ok(())
    }

    #[test]
    fn punctuation_runs_are_split_for_exact_morphology() -> Result<(), super::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let raw = b"(. ... !?".to_vec();
        let document = tokenizer.tokenize(raw.clone())?;
        assert_eq!(document.decode(), raw);
        let punctuation = document
            .units()
            .iter()
            .filter(|unit| unit.kind == nedo_core::LexicalKind::Punctuation)
            .collect::<Vec<_>>();
        assert_eq!(punctuation.len(), 7);
        assert!(punctuation
            .iter()
            .all(|unit| unit.span.end > unit.span.start));
        Ok(())
    }

    #[test]
    fn native_batch_preserves_order() -> Result<(), super::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let inputs = vec![b"bir".to_vec(), b"iki".to_vec(), b"uc".to_vec()];
        let documents = tokenizer.tokenize_batch(&inputs, 2)?;
        assert_eq!(documents.len(), inputs.len());
        for (document, input) in documents.iter().zip(inputs) {
            assert_eq!(document.decode(), input);
        }
        Ok(())
    }
    #[test]
    fn batch_ranges_are_byte_weighted_and_cover_every_input() -> Result<(), super::TokenizerError> {
        let inputs = vec![
            vec![0_u8; 700_000],
            vec![1_u8; 700_000],
            vec![2_u8; 700_000],
            vec![3_u8; 10],
        ];
        let ranges = super::build_batch_ranges(&inputs, 2)?;
        assert_eq!(ranges, vec![0..2, 2..4]);
        Ok(())
    }

    #[test]
    fn cached_batch_matches_independent_tokenization() -> Result<(), super::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let inputs = vec![
            "Ankara'da evlerimizdekiler konuşuyor.".as_bytes().to_vec(),
            "Ankara'da evlerimizdekiler konuşuyor.".as_bytes().to_vec(),
            "Evlerimizdekiler Ankara'da konuşuyor.".as_bytes().to_vec(),
        ];
        let cached = tokenizer.tokenize_batch(&inputs, 2)?;
        let independent = inputs
            .iter()
            .cloned()
            .map(|raw| tokenizer.tokenize(raw))
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(cached, independent);
        Ok(())
    }

    #[test]
    fn dates_do_not_split_context_and_fallback_has_a_hard_limit(
    ) -> Result<(), super::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig {
            max_sentence_tokens: 512,
            max_fallback_chars: 8,
            ..TokenizerConfig::default()
        })?;
        let document =
            tokenizer.tokenize("23.07.2026 qwxqwxqwxqwxqwxqwxqwxqwx".as_bytes().to_vec())?;
        let date = document
            .units()
            .iter()
            .find(|unit| unit.kind == nedo_core::LexicalKind::Number)
            .ok_or(super::TokenizerError::InvalidConfiguration(
                "missing date unit",
            ))?;
        assert!(date.cuts.len() >= 4);
        let unknowns = document
            .units()
            .iter()
            .filter(|unit| unit.status == TokenStatus::Unknown)
            .collect::<Vec<_>>();
        assert!(unknowns.len() >= 2);
        for unit in &unknowns {
            let start = usize::try_from(unit.span.start)
                .map_err(|_| super::TokenizerError::LengthOverflow("test start"))?;
            let end = usize::try_from(unit.span.end)
                .map_err(|_| super::TokenizerError::LengthOverflow("test end"))?;
            assert!(
                std::str::from_utf8(&document.raw()[start..end])
                    .map_err(|_| super::TokenizerError::InvalidUtf8Unit)?
                    .chars()
                    .count()
                    <= 8
            );
        }

        let pathological = "q".repeat(48 + 17);
        let tokenizer = Tokenizer::embedded(TokenizerConfig {
            mode: TokenizerMode::Turkish,
            ..TokenizerConfig::default()
        })?;
        let pathological_document = tokenizer.tokenize(pathological.as_bytes().to_vec())?;
        let pathological_unknowns = pathological_document
            .units()
            .iter()
            .filter(|unit| {
                matches!(
                    unit.status,
                    TokenStatus::Unknown | TokenStatus::Code | TokenStatus::Opaque
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(pathological_unknowns.len(), 2);
        for unit in pathological_unknowns {
            let start = usize::try_from(unit.span.start)
                .map_err(|_| super::TokenizerError::LengthOverflow("test start"))?;
            let end = usize::try_from(unit.span.end)
                .map_err(|_| super::TokenizerError::LengthOverflow("test end"))?;
            assert!(
                std::str::from_utf8(&pathological_document.raw()[start..end])
                    .map_err(|_| super::TokenizerError::InvalidUtf8Unit)?
                    .chars()
                    .count()
                    <= 48
            );
        }
        Ok(())
    }

    #[test]
    fn completes_numeric_derivations_without_losing_apostrophe() -> Result<(), super::TokenizerError>
    {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        for (raw, expected_id, expected_surface) in [
            (b"20'nci".as_slice(), "Ord", "nci"),
            (b"1'incisini".as_slice(), "Ord", "inci"),
            (b"110'ar".as_slice(), "Dist", "ar"),
            (b"18'erli".as_slice(), "Dist", "er"),
            ("20'şer".as_bytes(), "Dist", "şer"),
            ("11’İNCİ".as_bytes(), "Ord", "inci"),
            ("1’İNCİSİNİ".as_bytes(), "Ord", "inci"),
            ("20’ŞER".as_bytes(), "Dist", "şer"),
            (b"I'inci".as_slice(), "Ord", "inci"),
            ("II’nci".as_bytes(), "Ord", "nci"),
            (b"IX'uncu".as_slice(), "Ord", "uncu"),
        ] {
            let document = tokenizer.tokenize(raw.to_vec())?;
            assert_eq!(document.decode(), raw);
            assert_eq!(document.units().len(), 1);
            let unit = &document.units()[0];
            assert_eq!(unit.status, TokenStatus::Morphological);
            assert!(unit.cuts.len() >= 2);
            let analysis = unit
                .analysis
                .as_ref()
                .ok_or(super::TokenizerError::MissingAnalysis { index: 0 })?;
            assert!(analysis.morphemes.iter().any(|morpheme| {
                morpheme.id == expected_id && morpheme.surface == expected_surface
            }));
        }
        Ok(())
    }
    #[test]
    fn no_context_uses_first_surface_valid_candidate_or_unknown(
    ) -> Result<(), super::TokenizerError> {
        for contextual_disambiguation in [true, false] {
            let tokenizer = Tokenizer::embedded(TokenizerConfig {
                contextual_disambiguation,
                ..TokenizerConfig::default()
            })?;
            for raw in ["Qʼumarkaj'da".as_bytes(), b"normal".as_slice()] {
                let document = tokenizer.tokenize(raw.to_vec())?;
                assert_eq!(document.decode(), raw);
                assert!(!document.units().is_empty());
            }
        }
        Ok(())
    }

    #[test]
    fn cached_batch_preserves_unicode_numeric_apostrophe_semantics(
    ) -> Result<(), super::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let inputs = [
            "Alan 14.000 m²'den büyüktür.",
            "Debi 10 m³/s'dir.",
            "Payın ½’sini kullandı.",
            "FⅧ’ün devamı işlendi.",
            "Kapasite 1700m²'lik binadır.",
            "Muhtemel muhalefeti asgariye indirmenin altyapısını oluşturuyor.",
        ]
        .into_iter()
        .map(|value| value.as_bytes().to_vec())
        .collect::<Vec<_>>();
        let cached = tokenizer.tokenize_batch(&inputs, 4)?;
        let independent = inputs
            .iter()
            .cloned()
            .map(|raw| tokenizer.tokenize(raw))
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(cached, independent);
        Ok(())
    }

    #[test]
    fn flat_training_batch_matches_rich_reference_exactly() -> Result<(), super::TokenizerError> {
        let inputs = vec![
            "Ankara'da evlerimizden mi? 20'nci, 3,14 ve 23.07.2026."
                .as_bytes()
                .to_vec(),
            b"```rust
fn parseHttpRequest_header42() { return 0; }
```"
            .to_vec(),
            "Qʼumarkaj'da 🙂 snake_case camelCase".as_bytes().to_vec(),
            vec![0x00, 0xff, b'a', 0x80, b'Z'],
            "x".repeat(97).into_bytes(),
        ];
        let newline_flags = vec![true, false, true, false, true];
        let vocabulary = CharacterVocabulary::from_sorted(Vec::new())?;
        let option_sets = [
            TrainingEncodingOptions::default(),
            TrainingEncodingOptions {
                emit_unit_boundaries: false,
                emit_morpheme_boundaries: true,
                emit_code_boundaries: true,
            },
            TrainingEncodingOptions {
                emit_unit_boundaries: true,
                emit_morpheme_boundaries: false,
                emit_code_boundaries: false,
            },
        ];
        for mode in [
            TokenizerMode::Auto,
            TokenizerMode::Turkish,
            TokenizerMode::Code,
        ] {
            for contextual_disambiguation in [true, false] {
                let tokenizer = Tokenizer::embedded(TokenizerConfig {
                    mode,
                    contextual_disambiguation,
                    ..TokenizerConfig::default()
                })?;
                let rich = tokenizer.tokenize_batch(&inputs, 2)?;
                for options in option_sets {
                    let reference =
                        vocabulary.encode_training_batch(&rich, &newline_flags, options)?;
                    let flat = tokenizer.encode_training_batch(
                        &inputs,
                        &newline_flags,
                        &vocabulary,
                        2,
                        options,
                    )?;
                    assert_eq!(flat, reference);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn stateful_flat_encoder_reuses_cache_without_changing_output(
    ) -> Result<(), super::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let vocabulary = CharacterVocabulary::from_sorted(Vec::new())?;
        let inputs = vec![
            "Evlerimizden geldik, evlerimizden çıktık."
                .as_bytes()
                .to_vec(),
            "Evlerimizden geldik, evlerimizden çıktık."
                .as_bytes()
                .to_vec(),
        ];
        let newlines = vec![false, true];
        let mut encoder =
            tokenizer.training_encoder(&vocabulary, 1, TrainingEncodingOptions::default())?;
        let first = encoder.encode_batch(&inputs, &newlines)?;
        let first_stats = encoder.cache_stats();
        let second = encoder.encode_batch(&inputs, &newlines)?;
        let second_stats = encoder.cache_stats();
        assert_eq!(first, second);
        assert!(second_stats.hits > first_stats.hits);
        assert_eq!(second_stats.misses, first_stats.misses);
        assert!(second_stats.entries > 0);
        assert!(second_stats.approximate_bytes > 0);
        encoder.clear_caches();
        assert_eq!(encoder.cache_stats(), super::TrainingCacheStats::default());
        Ok(())
    }

    #[test]
    fn persistent_surface_runtime_matches_stateless_and_reuses_cache(
    ) -> Result<(), super::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig {
            mode: TokenizerMode::Turkish,
            ..TokenizerConfig::default()
        })?;
        let vocabulary = SurfaceVocabulary::from_ranked(Vec::new())?;
        let inputs = vec![
            "Evlerimizden geldik, evlerimizden çıktık."
                .as_bytes()
                .to_vec(),
            "Evlerimizden geldik, evlerimizden çıktık."
                .as_bytes()
                .to_vec(),
        ];
        let newlines = vec![false, false];
        let expected = tokenizer.encode_surface_batch(&inputs, &newlines, &vocabulary, 1, true)?;
        let mut runtime =
            tokenizer.surface_runtime_cache(1, SurfaceEncoderOptions::one_pass(true))?;
        let first = tokenizer.encode_surface_batch_with_runtime(
            &inputs,
            &newlines,
            &vocabulary,
            &mut runtime,
        )?;
        let first_stats = runtime.cache_stats();
        let second = tokenizer.encode_surface_batch_with_runtime(
            &inputs,
            &newlines,
            &vocabulary,
            &mut runtime,
        )?;
        let second_stats = runtime.cache_stats();
        assert_eq!(first, expected);
        assert_eq!(second, expected);
        assert!(second_stats.hits > first_stats.hits);
        assert_eq!(second_stats.misses, first_stats.misses);
        runtime.clear();
        assert_eq!(runtime.cache_stats(), super::TrainingCacheStats::default());
        Ok(())
    }

    #[test]
    fn persistent_surface_runtime_rejects_different_tokenizer_config(
    ) -> Result<(), super::TokenizerError> {
        let turkish = Tokenizer::embedded(TokenizerConfig {
            mode: TokenizerMode::Turkish,
            ..TokenizerConfig::default()
        })?;
        let code = Tokenizer::embedded(TokenizerConfig {
            mode: TokenizerMode::Code,
            ..TokenizerConfig::default()
        })?;
        let vocabulary = SurfaceVocabulary::from_ranked(Vec::new())?;
        let mut runtime =
            turkish.surface_runtime_cache(1, SurfaceEncoderOptions::one_pass(true))?;
        let inputs = vec![b"evlerimizden".to_vec()];
        let newlines = vec![false];
        let error = code
            .encode_surface_batch_with_runtime(&inputs, &newlines, &vocabulary, &mut runtime)
            .expect_err("runtime must be bound to the tokenizer configuration that created it");
        assert!(matches!(
            error,
            super::TokenizerError::InvalidConfiguration(_)
        ));
        Ok(())
    }

    #[test]
    fn operators_split_only_in_turkish_mode() -> Result<(), super::TokenizerError> {
        let turkish = Tokenizer::embedded(TokenizerConfig::default())?;
        let turkish_document = turkish.tokenize(b"?.".to_vec())?;
        assert_eq!(turkish_document.decode(), b"?.");
        assert_eq!(turkish_document.units().len(), 2);
        let surfaces = turkish_document
            .units()
            .iter()
            .map(|unit| {
                let start = usize::try_from(unit.span.start)
                    .map_err(|_| super::TokenizerError::LengthOverflow("test start"))?;
                let end = usize::try_from(unit.span.end)
                    .map_err(|_| super::TokenizerError::LengthOverflow("test end"))?;
                Ok(&turkish_document.raw()[start..end])
            })
            .collect::<Result<Vec<_>, super::TokenizerError>>()?;
        assert_eq!(surfaces, [b"?".as_slice(), b".".as_slice()]);

        let code = Tokenizer::embedded(TokenizerConfig {
            mode: TokenizerMode::Code,
            ..TokenizerConfig::default()
        })?;
        let code_document = code.tokenize(b"?.".to_vec())?;
        assert_eq!(code_document.decode(), b"?.");
        assert_eq!(code_document.units().len(), 1);
        assert_eq!(code_document.units()[0].status, TokenStatus::Code);
        Ok(())
    }
}

/// Read-only diagnostics used to build and verify compiled surface tables.
#[cfg(feature = "surface-audit")]
pub mod surface_audit {
    use super::{
        build_flat_analysis_set, explicit_code_spans, scan, split_units, unit_str, LexicalKind,
        TokenMode, Tokenizer, TokenizerError, TokenizerMode,
    };

    /// One distinct tokenizer-visible output among a surface form's analyses.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct SurfaceOutputClassAudit {
        /// Relative byte cuts produced by this class.
        pub cuts: Vec<u32>,
        /// Whether this class represents the explicit unknown analysis.
        pub unknown: bool,
        /// Number of native analyses collapsed into this output class.
        pub analysis_count: usize,
    }

    /// Exact candidate and output-class audit for one surface form.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct SurfaceAnalysisAudit {
        /// Number of surface-valid native analysis candidates.
        pub candidate_count: usize,
        /// Distinct tokenizer-visible `(cuts, unknown)` outputs.
        pub output_classes: Vec<SurfaceOutputClassAudit>,
    }

    impl SurfaceAnalysisAudit {
        /// Whether context can never change this surface form's tokenizer output.
        #[must_use]
        pub fn output_invariant(&self) -> bool {
            self.output_classes.len() == 1
        }
    }

    impl Tokenizer<'_> {
        /// Returns the exact surface strings that enter native morphology for one document.
        ///
        /// # Errors
        ///
        /// Returns an error if scanning, unit splitting, or UTF-8 extraction fails.
        pub fn audit_surface_tokens(&self, raw: Vec<u8>) -> Result<Vec<String>, TokenizerError> {
            Ok(self
                .audit_surface_segments(raw)?
                .into_iter()
                .flatten()
                .collect())
        }

        /// Returns exact sentence/chunk segments entering contextual morphology.
        ///
        /// # Errors
        ///
        /// Returns an error if scanning, unit splitting, UTF-8 extraction, or boundary detection fails.
        pub fn audit_surface_segments(
            &self,
            raw: Vec<u8>,
        ) -> Result<Vec<Vec<String>>, TokenizerError> {
            let scan = scan(raw)?;
            let decoded = scan.document().decode();
            let code_spans = match self.config.mode {
                TokenizerMode::Auto => explicit_code_spans(decoded),
                TokenizerMode::Turkish | TokenizerMode::Code => Vec::new(),
            };
            let units = split_units(&scan, &code_spans, self.config.mode)?;
            let mut output = Vec::new();
            let mut segment = Vec::new();
            for unit in &units {
                if unit.mode != TokenMode::Turkish
                    || matches!(
                        unit.kind,
                        LexicalKind::LineBreak | LexicalKind::Control | LexicalKind::Opaque
                    )
                {
                    if !segment.is_empty() {
                        output.push(std::mem::take(&mut segment));
                    }
                    continue;
                }
                if matches!(unit.kind, LexicalKind::Whitespace) {
                    continue;
                }
                segment.push(unit_str(decoded, unit.span)?.to_owned());
                let boundary = super::is_sentence_boundary(unit.kind, decoded, unit.span)?;
                if boundary || segment.len() >= self.config.max_sentence_tokens {
                    output.push(std::mem::take(&mut segment));
                }
            }
            if !segment.is_empty() {
                output.push(segment);
            }
            Ok(output)
        }

        /// Builds one exact full-table compiler entry in production candidate order.
        ///
        /// # Errors
        ///
        /// Returns an error if native analysis, alignment, or candidate cardinality fails.
        #[cfg(feature = "compiled-surface-table")]
        pub fn audit_compiled_surface_analysis_entry(
            &self,
            token: &str,
        ) -> Result<super::CompiledSurfaceAnalysisEntry, TokenizerError> {
            let analyses = super::analyze_token_with_quality_fallback(&self.morphology, token)?;
            let set = build_flat_analysis_set(token, analyses)?;
            if set.relative_cuts.len() != set.ambiguity.len()
                || set.ambiguity.len() != set.unknown.len()
            {
                return Err(TokenizerError::InvalidTrainingEncoding(
                    "compiled audit candidate cardinalities differ",
                ));
            }
            let candidates = set
                .relative_cuts
                .iter()
                .zip(&set.ambiguity)
                .zip(&set.unknown)
                .map(
                    |((cuts, ambiguity), unknown)| super::CompiledSurfaceCandidateEntry {
                        cuts: cuts.clone(),
                        unknown: *unknown,
                        canonical: ambiguity.canonical.clone(),
                        lemma: ambiguity.lemma.clone(),
                        igs: ambiguity.igs.clone(),
                        java_hash: ambiguity.java_hash,
                    },
                )
                .collect();
            Ok(super::CompiledSurfaceAnalysisEntry {
                surface: token.as_bytes().to_vec(),
                candidates,
            })
        }

        /// Builds one exact NedoFormer shadow-analysis table entry.
        ///
        /// # Errors
        ///
        /// Returns an error if shadow analysis, alignment, or candidate cardinality fails.
        #[cfg(feature = "compiled-surface-table")]
        pub fn audit_nedoformer_compiled_surface_analysis_entry(
            &self,
            token: &str,
        ) -> Result<super::CompiledSurfaceAnalysisEntry, TokenizerError> {
            let analyses = super::analyze_token_with_nedoformer_shadow(&self.morphology, token)?;
            let set = build_flat_analysis_set(token, analyses)?;
            if set.relative_cuts.len() != set.ambiguity.len()
                || set.ambiguity.len() != set.unknown.len()
            {
                return Err(TokenizerError::InvalidTrainingEncoding(
                    "NedoFormer compiled audit candidate cardinalities differ",
                ));
            }
            let candidates = set
                .relative_cuts
                .iter()
                .zip(&set.ambiguity)
                .zip(&set.unknown)
                .map(
                    |((cuts, ambiguity), unknown)| super::CompiledSurfaceCandidateEntry {
                        cuts: cuts.clone(),
                        unknown: *unknown,
                        canonical: ambiguity.canonical.clone(),
                        lemma: ambiguity.lemma.clone(),
                        igs: ambiguity.igs.clone(),
                        java_hash: ambiguity.java_hash,
                    },
                )
                .collect();
            Ok(super::CompiledSurfaceAnalysisEntry {
                surface: token.as_bytes().to_vec(),
                candidates,
            })
        }

        /// Audits all surface-valid analyses using the exact production alignment path.
        ///
        /// # Errors
        ///
        /// Returns an error if native analysis or exact surface alignment fails.
        pub fn audit_surface_analysis(
            &self,
            token: &str,
        ) -> Result<SurfaceAnalysisAudit, TokenizerError> {
            let analyses = self.morphology.analyze_token(token)?;
            let set = build_flat_analysis_set(token, analyses)?;
            let mut output_classes = Vec::<SurfaceOutputClassAudit>::new();
            for (cuts, unknown) in set.relative_cuts.iter().zip(&set.unknown) {
                if let Some(existing) = output_classes.iter_mut().find(|class| {
                    class.cuts.as_slice() == cuts.as_slice() && class.unknown == *unknown
                }) {
                    existing.analysis_count = existing.analysis_count.saturating_add(1);
                } else {
                    output_classes.push(SurfaceOutputClassAudit {
                        cuts: cuts.to_vec(),
                        unknown: *unknown,
                        analysis_count: 1,
                    });
                }
            }
            Ok(SurfaceAnalysisAudit {
                candidate_count: set.relative_cuts.len(),
                output_classes,
            })
        }
    }
}

#[cfg(all(test, feature = "surface-audit", feature = "compiled-surface-table"))]
mod compiled_surface_safety_tests {
    use super::{
        encode_compiled_surface_analysis_table, CompiledSurfaceAnalysisTable,
        SurfaceEncoderOptions, SurfaceVocabulary, Tokenizer, TokenizerConfig, TokenizerError,
    };

    #[test]
    fn semantic_verification_rejects_valid_checksum_with_wrong_cuts() {
        let exact_tokenizer =
            Tokenizer::embedded(TokenizerConfig::default()).expect("embedded tokenizer must load");
        let mut entry = exact_tokenizer
            .audit_compiled_surface_analysis_entry("evler")
            .expect("exact compiler entry must build");
        let first = entry
            .candidates
            .first_mut()
            .expect("surface must have at least one candidate");
        first.cuts = if first.cuts == [1] { vec![2] } else { vec![1] };
        let bytes = encode_compiled_surface_analysis_table(&[entry])
            .expect("wrong-but-structurally-valid table must encode with a valid checksum");
        let table = CompiledSurfaceAnalysisTable::from_bytes(&bytes)
            .expect("wrong-but-checksummed table must parse structurally");
        let verifier =
            Tokenizer::embedded(TokenizerConfig::default()).expect("embedded tokenizer must load");
        match verifier.with_verified_compiled_surface_analysis_table(table) {
            Err(TokenizerError::InvalidCompiledSurfaceTable { reason, .. }) => {
                assert_eq!(reason, "candidate cuts or candidate order differ");
            }
            Err(other) => panic!("unexpected verification error: {other}"),
            Ok(_) => panic!("semantic verification accepted deliberately wrong cuts"),
        }
    }

    #[test]
    fn semantic_verification_accepts_exact_compiler_output() {
        let compiler =
            Tokenizer::embedded(TokenizerConfig::default()).expect("embedded tokenizer must load");
        let entry = compiler
            .audit_compiled_surface_analysis_entry("evler")
            .expect("exact compiler entry must build");
        let bytes =
            encode_compiled_surface_analysis_table(&[entry]).expect("exact table must encode");
        let table =
            CompiledSurfaceAnalysisTable::from_bytes(&bytes).expect("exact table must parse");
        let verifier =
            Tokenizer::embedded(TokenizerConfig::default()).expect("embedded tokenizer must load");
        assert!(verifier
            .with_verified_compiled_surface_analysis_table(table)
            .is_ok());
    }

    #[test]
    fn partial_compiled_table_preserves_surface_output_and_reduces_live_misses() {
        let config = TokenizerConfig {
            mode: super::TokenizerMode::Turkish,
            ..TokenizerConfig::default()
        };
        let compiler = Tokenizer::embedded(config).expect("embedded tokenizer must load");
        let entry = compiler
            .audit_compiled_surface_analysis_entry("evlerimizden")
            .expect("partial compiler entry must build");
        let bytes =
            encode_compiled_surface_analysis_table(&[entry]).expect("partial table must encode");
        let table =
            CompiledSurfaceAnalysisTable::from_bytes(&bytes).expect("partial table must parse");
        let compiled = Tokenizer::embedded(config)
            .expect("embedded tokenizer must load")
            .with_verified_compiled_surface_analysis_table(table)
            .expect("partial table must verify");
        let baseline = Tokenizer::embedded(config).expect("embedded tokenizer must load");
        let vocabulary = SurfaceVocabulary::from_ranked(Vec::new())
            .expect("byte-fallback vocabulary must build");
        let inputs = vec![
            "evlerimizden bugün geldik.".as_bytes().to_vec(),
            "evlerimizden yarın çıkacağız.".as_bytes().to_vec(),
        ];
        let newline_flags = vec![false; inputs.len()];
        let mut baseline_encoder = baseline
            .surface_encoder_with_options(&vocabulary, 1, SurfaceEncoderOptions::one_pass(true))
            .expect("baseline encoder must build");
        let expected = baseline_encoder
            .encode_batch(&inputs, &newline_flags)
            .expect("baseline batch must encode");
        let baseline_misses = baseline_encoder.cache_stats().misses;
        let mut compiled_encoder = compiled
            .surface_encoder_with_options(&vocabulary, 1, SurfaceEncoderOptions::one_pass(true))
            .expect("compiled encoder must build");
        let actual = compiled_encoder
            .encode_batch(&inputs, &newline_flags)
            .expect("compiled batch must encode");
        assert_eq!(actual, expected);
        assert!(compiled_encoder.cache_stats().misses < baseline_misses);
    }

    #[test]
    fn pinned_nedoformer_table_requires_independent_exact_digest() {
        let compiler =
            Tokenizer::embedded(TokenizerConfig::default()).expect("embedded tokenizer must load");
        let entry = compiler
            .audit_nedoformer_compiled_surface_analysis_entry("cocuklarimizdan")
            .expect("NedoFormer shadow entry must compile");
        let bytes = encode_compiled_surface_analysis_table(&[entry])
            .expect("NedoFormer shadow table must encode");
        let table = CompiledSurfaceAnalysisTable::from_bytes(&bytes)
            .expect("NedoFormer shadow table must parse");
        let digest = table.digest();
        let mut wrong = digest;
        wrong[0] ^= 1;
        let verifier =
            Tokenizer::embedded(TokenizerConfig::default()).expect("embedded tokenizer must load");
        assert!(matches!(
            verifier.with_pinned_nedoformer_compiled_surface_analysis_table(table, wrong),
            Err(TokenizerError::CompiledSurfaceTableSealMismatch { .. })
        ));

        let table = CompiledSurfaceAnalysisTable::from_bytes(&bytes)
            .expect("NedoFormer shadow table must parse twice");
        let pinned = Tokenizer::embedded(TokenizerConfig::default())
            .expect("embedded tokenizer must load")
            .with_pinned_nedoformer_compiled_surface_analysis_table(table, digest)
            .expect("trusted exact digest must permit fast attach");
        let input = vec![b"cocuklarimizdan geliyor mu?".to_vec()];
        let baseline =
            Tokenizer::embedded(TokenizerConfig::default()).expect("embedded tokenizer must load");
        assert_eq!(
            pinned
                .nedoformer_sidecar_batch(&input, 1)
                .expect("pinned sidecar must encode"),
            baseline
                .nedoformer_sidecar_batch(&input, 1)
                .expect("baseline sidecar must encode")
        );
    }

    #[test]
    fn nedoformer_shadow_table_is_separately_verified_and_sidecar_exact() {
        let compiler =
            Tokenizer::embedded(TokenizerConfig::default()).expect("embedded tokenizer must load");
        let entry = compiler
            .audit_nedoformer_compiled_surface_analysis_entry("cocuklarimizdan")
            .expect("NedoFormer shadow entry must compile");
        let bytes = encode_compiled_surface_analysis_table(&[entry])
            .expect("NedoFormer shadow table must encode");
        let normal_table = CompiledSurfaceAnalysisTable::from_bytes(&bytes)
            .expect("NedoFormer shadow table must parse");
        let normal =
            Tokenizer::embedded(TokenizerConfig::default()).expect("embedded tokenizer must load");
        assert!(normal
            .with_verified_compiled_surface_analysis_table(normal_table)
            .is_err());

        let ndf_table = CompiledSurfaceAnalysisTable::from_bytes(&bytes)
            .expect("NedoFormer shadow table must parse twice");
        let baseline =
            Tokenizer::embedded(TokenizerConfig::default()).expect("embedded tokenizer must load");
        let accelerated = Tokenizer::embedded(TokenizerConfig::default())
            .expect("embedded tokenizer must load")
            .with_verified_nedoformer_compiled_surface_analysis_table(ndf_table)
            .expect("NedoFormer verifier must accept exact shadow table");
        let input = vec![b"cocuklarimizdan geliyor mu?".to_vec()];
        assert_eq!(
            baseline
                .nedoformer_sidecar_batch(&input, 1)
                .expect("baseline sidecar must encode"),
            accelerated
                .nedoformer_sidecar_batch(&input, 1)
                .expect("compiled NedoFormer sidecar must encode")
        );
    }
}
