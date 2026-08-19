//! Native parser and Viterbi decoder for the pinned Zemberek compressed ambiguity model.

use std::error::Error;
use std::fmt;
use std::sync::{Arc, OnceLock};

use crate::{NativeAnalysis, NativeMorpheme, NativeMorphology};

const LOSSY_MAGIC: u32 = 0xcafe_beef;
const HASH_MULTIPLIER: u32 = 16_777_619;
const INITIAL_HASH_SEED: u32 = 0x811c_9dc5;

/// Native compressed perceptron model.
#[derive(Clone, Debug)]
pub struct PerceptronModel {
    data: Vec<i32>,
    levels: Vec<HashLevel>,
}

/// Best contextual sequence and its accumulated perceptron score.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeDisambiguation {
    /// Selected analysis for every token.
    pub best: Vec<NativeAnalysis>,
    /// Accumulated f32 perceptron score.
    pub score: f32,
}

/// Exact feature metadata consumed by the pinned ambiguity perceptron.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AmbiguityWordData {
    /// Stable full-analysis identity used for exact Viterbi state deduplication.
    pub canonical: String,
    /// Dictionary lemma used by lexical features.
    pub lemma: String,
    /// Inflectional-group lexical forms, including the secondary-POS prefix on group zero.
    pub igs: Vec<String>,
    /// Java-compatible `SingleAnalysis.hashCode()` value used by the Viterbi active list.
    pub java_hash: i32,
}

/// Exact worker-local identity of all perceptron-visible analysis content.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AmbiguityScoringCode {
    /// Interned `(lemma, complete IG sequence)` identity.
    pub signature: u32,
    /// Interned exact canonical-analysis identity.
    pub canonical: u32,
    /// Original Java-compatible analysis hash used for active-list slot order.
    pub java_hash: i32,
    cache_signature: u32,
}

impl AmbiguityScoringCode {
    /// Creates an exact scoring identity and precomputes its optional cache signature.
    #[must_use]
    pub const fn new(signature: u32, canonical: u32, java_hash: i32) -> Self {
        let cache_signature = if signature < TRIGRAM_END_SIGNATURE as u32 {
            signature
        } else {
            INVALID_CACHE_SIGNATURE
        };
        Self {
            signature,
            canonical,
            java_hash,
            cache_signature,
        }
    }

    const fn boundary(signature: u32, canonical: u32, java_hash: i32) -> Self {
        Self {
            signature,
            canonical,
            java_hash,
            cache_signature: signature,
        }
    }
}

/// Shared immutable prepared-feature bank for stable compiled analysis signatures.
///
/// Each slot is initialized at most once. Callers must size this bank only for a
/// globally stable signature namespace; worker-local/dynamic signatures must use
/// the per-worker fallback cache instead.
pub struct SharedPreparedWordCache {
    entries: Box<[OnceLock<PreparedWordFeatures>]>,
}

impl SharedPreparedWordCache {
    /// Creates an empty exact bank with one slot per globally stable signature.
    #[must_use]
    pub fn with_slots(slots: usize) -> Self {
        let entries = std::iter::repeat_with(OnceLock::new)
            .take(slots)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self { entries }
    }

    /// Returns the stable signature capacity of this bank.
    #[must_use]
    pub const fn slots(&self) -> usize {
        self.entries.len()
    }

    fn ensure(
        &self,
        index: usize,
        model: &PerceptronModel,
        word: &AmbiguityWordData,
    ) -> Option<()> {
        let slot = self.entries.get(index)?;
        slot.get_or_init(|| PreparedWordFeatures::new(model, word));
        Some(())
    }

    fn get(&self, index: usize) -> Option<&PreparedWordFeatures> {
        self.entries.get(index)?.get()
    }
}

#[derive(Clone, Copy)]
enum PreparedWordLocation {
    Shared(usize),
    Local(usize),
}

/// Persistent exact trigram-score cache owned by one tokenizer worker.
#[derive(Default)]
pub struct DisambiguationScoreCache {
    trigrams: TrigramScoreCache,
    decoder: CompactDecodeWorkspace,
    shared_prepared_words: Option<Arc<SharedPreparedWordCache>>,
    prepared_words: Vec<Option<Box<PreparedWordFeatures>>>,
    dense_attempts: u64,
    dense_successes: u64,
    dense_duplicate_fallbacks: u64,
    dense_state_fallbacks: u64,
    dense_tie_fallbacks: u64,
}

impl DisambiguationScoreCache {
    /// Creates a balanced exact cache sized for high-shard corpus preprocessing.
    ///
    /// Capacity exhaustion only disables further score retention; transition
    /// scores are still computed exactly, so tokenization semantics are unchanged.
    #[must_use]
    pub fn balanced() -> Self {
        Self {
            trigrams: TrigramScoreCache::with_capacities(
                BALANCED_TRIGRAM_PAIR_SLOTS,
                BALANCED_TRIGRAM_ROW_CAPACITY,
                BALANCED_TRIGRAM_OVERFLOW_SLOTS,
            ),
            decoder: CompactDecodeWorkspace::default(),
            shared_prepared_words: None,
            prepared_words: Vec::new(),
            dense_attempts: 0,
            dense_successes: 0,
            dense_duplicate_fallbacks: 0,
            dense_state_fallbacks: 0,
            dense_tie_fallbacks: 0,
        }
    }

    /// Creates a smaller exact cache for independent one-pass corpus shards.
    ///
    /// Capacity exhaustion only disables further score retention; transition
    /// scores are still computed exactly, so tokenization semantics are unchanged.
    #[must_use]
    pub fn compact() -> Self {
        Self {
            trigrams: TrigramScoreCache::with_capacities(
                COMPACT_TRIGRAM_PAIR_SLOTS,
                COMPACT_TRIGRAM_ROW_CAPACITY,
                COMPACT_TRIGRAM_OVERFLOW_SLOTS,
            ),
            decoder: CompactDecodeWorkspace::default(),
            shared_prepared_words: None,
            prepared_words: Vec::new(),
            dense_attempts: 0,
            dense_successes: 0,
            dense_duplicate_fallbacks: 0,
            dense_state_fallbacks: 0,
            dense_tie_fallbacks: 0,
        }
    }

    /// Shares a prepared-feature bank whose indices are stable compiled signatures.
    pub fn set_shared_prepared_words(&mut self, cache: Arc<SharedPreparedWordCache>) {
        self.shared_prepared_words = Some(cache);
        self.prepared_words.clear();
    }

    /// Removes all compiled transition scores and counters while preserving capacity policy.
    pub fn clear(&mut self) {
        self.trigrams.clear();
        self.decoder = CompactDecodeWorkspace::default();
        for entry in &mut self.prepared_words {
            *entry = None;
        }
        self.dense_attempts = 0;
        self.dense_successes = 0;
        self.dense_duplicate_fallbacks = 0;
        self.dense_state_fallbacks = 0;
        self.dense_tie_fallbacks = 0;
    }

    fn ensure_prepared_word(
        &mut self,
        signature: u32,
        model: &PerceptronModel,
        word: &AmbiguityWordData,
    ) -> Option<PreparedWordLocation> {
        let index = usize::try_from(signature).ok()?;
        // Boundary sentinels use the top u32 values; they are kept segment-local.
        if index > 2_000_000 {
            return None;
        }
        let local_index = if let Some(shared) = self.shared_prepared_words.as_ref() {
            if index < shared.slots() {
                shared.ensure(index, model, word)?;
                return Some(PreparedWordLocation::Shared(index));
            }
            index.checked_sub(shared.slots())?
        } else {
            index
        };
        if self.prepared_words.len() <= local_index {
            self.prepared_words.resize_with(local_index + 1, || None);
        }
        if self.prepared_words[local_index].is_none() {
            self.prepared_words[local_index] =
                Some(Box::new(PreparedWordFeatures::new(model, word)));
        }
        Some(PreparedWordLocation::Local(local_index))
    }

    fn prepared_word(&self, location: PreparedWordLocation) -> Option<&PreparedWordFeatures> {
        match location {
            PreparedWordLocation::Shared(index) => self.shared_prepared_words.as_ref()?.get(index),
            PreparedWordLocation::Local(index) => self.prepared_words.get(index)?.as_deref(),
        }
    }

    /// Returns `(hits, misses, retained transitions, allocated slots)`.
    #[must_use]
    pub fn stats(&self) -> (u64, u64, usize, usize) {
        (
            self.trigrams.hits,
            self.trigrams.misses,
            self.trigrams.entries,
            self.trigrams.pair_keys.len() + self.trigrams.overflow.entries.len(),
        )
    }

    /// Returns row/table usage for cache-capacity telemetry.
    #[must_use]
    pub fn layout_stats(&self) -> (usize, usize, usize, usize, usize) {
        (
            self.trigrams.rows.len(),
            self.trigrams.row_capacity,
            self.trigrams.pair_slots,
            self.trigrams.overflow.size,
            self.trigrams.overflow.slots,
        )
    }

    /// Returns dense-decoder attempts, successes, and fallback counts.
    #[must_use]
    pub const fn dense_stats(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.dense_attempts,
            self.dense_successes,
            self.dense_duplicate_fallbacks,
            self.dense_state_fallbacks,
            self.dense_tie_fallbacks,
        )
    }

    /// Approximate bytes retained by pair rows and overflow scores.
    #[must_use]
    pub fn approximate_bytes(&self) -> u64 {
        let bytes = self
            .trigrams
            .pair_keys
            .capacity()
            .saturating_mul(std::mem::size_of::<u64>())
            .saturating_add(
                self.trigrams
                    .pair_rows
                    .capacity()
                    .saturating_mul(std::mem::size_of::<u32>()),
            )
            .saturating_add(
                self.trigrams
                    .rows
                    .capacity()
                    .saturating_mul(std::mem::size_of::<TransitionRow>()),
            )
            .saturating_add(self.trigrams.overflow.approximate_bytes())
            .saturating_add(self.decoder.approximate_bytes());
        u64::try_from(bytes).unwrap_or(u64::MAX)
    }
}

/// Computes the exact ambiguity feature metadata for one native analysis.
#[must_use]
pub fn ambiguity_word_data(analysis: &NativeAnalysis) -> AmbiguityWordData {
    let word = WordData::from_analysis(analysis);
    AmbiguityWordData {
        canonical: analysis.canonical.clone(),
        lemma: word.lemma,
        igs: word.igs,
        java_hash: java_analysis_hash(analysis),
    }
}

/// Native Sak-style trigram morphological disambiguator.
#[derive(Clone, Debug)]
pub struct NativeDisambiguator {
    model: PerceptronModel,
    begin: AmbiguityWordData,
    end: AmbiguityWordData,
}

/// Native ambiguity-model or decoding failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DisambiguationError {
    message: String,
}

impl PerceptronModel {
    /// Parses the pinned big-endian `LossyIntLookup` model.
    ///
    /// # Errors
    ///
    /// Returns an error for bad magic, malformed lengths, invalid MPHF levels, or trailing bytes.
    pub fn parse(bytes: &[u8]) -> Result<Self, DisambiguationError> {
        let mut reader = Reader::new(bytes);
        if reader.read_u32()? != LOSSY_MAGIC {
            return Err(failure("ambiguity model has invalid magic"));
        }
        let data_length = reader.read_usize("weight data length")?;
        if data_length == 0 || data_length % 2 != 0 {
            return Err(failure(
                "ambiguity weight data length must be positive and even",
            ));
        }
        let mut data = Vec::with_capacity(data_length);
        for _ in 0..data_length {
            data.push(reader.read_i32()?);
        }
        let level_count = reader.read_usize("MPHF level count")?;
        if level_count == 0 {
            return Err(failure("ambiguity MPHF has no levels"));
        }
        let mut levels = Vec::with_capacity(level_count);
        for _ in 0..level_count {
            let key_amount = reader.read_usize("MPHF key amount")?;
            let bucket_amount = reader.read_usize("MPHF bucket amount")?;
            if key_amount == 0 || bucket_amount == 0 {
                return Err(failure("ambiguity MPHF level has an empty dimension"));
            }
            let seeds = reader.read_bytes(bucket_amount)?.to_vec();
            let failed_count = reader.read_usize("MPHF failed index count")?;
            let mut failed_indexes = Vec::with_capacity(failed_count);
            for _ in 0..failed_count {
                let value = reader.read_usize("MPHF failed index")?;
                if value >= data_length / 2 {
                    return Err(failure("ambiguity MPHF failed index exceeds weight count"));
                }
                failed_indexes.push(value);
            }
            levels.push(HashLevel {
                key_amount,
                bucket_amount,
                seeds,
                failed_indexes,
            });
        }
        if !reader.is_finished() {
            return Err(failure("ambiguity model has trailing bytes"));
        }
        if levels[0].key_amount != data_length / 2 {
            return Err(failure("ambiguity MPHF size differs from weight count"));
        }
        Ok(Self { data, levels })
    }

    /// Number of compressed model entries.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len() / 2
    }

    /// Whether the compressed model has no entries.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Looks up one feature weight with the same lossy fingerprint check as Java.
    #[must_use]
    pub fn get(&self, key: &str) -> f32 {
        let (initial_hash, java_hash) = initial_mphf_and_java_hash(key);
        self.get_prehashed(key, initial_hash, java_hash)
    }

    fn get_prehashed(&self, key: &str, initial_hash: usize, java_hash: i32) -> f32 {
        let Some(index) = self.mphf_index(key, initial_hash) else {
            return 0.0;
        };
        let data_index = index * 2;
        if self.data.get(data_index).copied() != Some(java_hash & 0x07ff_ffff) {
            return 0.0;
        }
        self.data
            .get(data_index + 1)
            .map_or(0.0, |bits| f32::from_bits(bits.cast_unsigned()))
    }

    fn get_virtual(&self, key: &FeatureView<'_>, initial_hash: usize, java_hash: i32) -> f32 {
        let mut index = None;
        for (level_index, level) in self.levels.iter().enumerate() {
            let Some(seed) = level
                .seeds
                .get(initial_hash % level.bucket_amount)
                .copied()
                .map(u32::from)
            else {
                return 0.0;
            };
            if seed == 0 {
                continue;
            }
            let slot = key.mphf_hash(seed) % level.key_amount;
            index = if level_index == 0 {
                Some(slot)
            } else {
                self.levels
                    .get(level_index - 1)
                    .and_then(|previous| previous.failed_indexes.get(slot))
                    .copied()
            };
            break;
        }
        let Some(index) = index else {
            return 0.0;
        };
        let data_index = index * 2;
        if self.data.get(data_index).copied() != Some(java_hash & 0x07ff_ffff) {
            return 0.0;
        }
        self.data
            .get(data_index + 1)
            .map_or(0.0, |bits| f32::from_bits(bits.cast_unsigned()))
    }

    fn mphf_index(&self, key: &str, initial_hash: usize) -> Option<usize> {
        for (level_index, level) in self.levels.iter().enumerate() {
            let seed = u32::from(*level.seeds.get(initial_hash % level.bucket_amount)?);
            if seed == 0 {
                continue;
            }
            let slot = mphf_hash(key, seed) % level.key_amount;
            return if level_index == 0 {
                Some(slot)
            } else {
                self.levels
                    .get(level_index - 1)?
                    .failed_indexes
                    .get(slot)
                    .copied()
            };
        }
        None
    }
}

impl NativeDisambiguator {
    /// Creates a decoder from a parsed model.
    #[must_use]
    pub fn new(model: PerceptronModel) -> Self {
        Self {
            model,
            begin: boundary_word_data("<s>"),
            end: boundary_word_data("</s>"),
        }
    }

    /// Parses model bytes and creates a decoder.
    ///
    /// # Errors
    ///
    /// Returns an error when model bytes are malformed.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DisambiguationError> {
        Ok(Self::new(PerceptronModel::parse(bytes)?))
    }

    /// Returns the parsed compressed model.
    #[must_use]
    pub const fn model(&self) -> &PerceptronModel {
        &self.model
    }

    /// Analyzes tokens and selects the highest-scoring contextual path.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty sentence or morphology failures.
    pub fn analyze_and_disambiguate(
        &self,
        morphology: &NativeMorphology<'_>,
        tokens: &[&str],
    ) -> Result<NativeDisambiguation, DisambiguationError> {
        if tokens.is_empty() {
            return Err(failure("disambiguation cannot run on an empty sentence"));
        }
        let mut candidates = Vec::with_capacity(tokens.len());
        for token in tokens {
            candidates.push(
                morphology
                    .analyze_token(token)
                    .map_err(|error| failure(format!("morphology analysis failed: {error}")))?,
            );
        }
        self.disambiguate(tokens, &candidates)
    }

    /// Selects the highest-scoring path from precomputed token candidates.
    ///
    /// # Errors
    ///
    /// Returns an error for empty or length-mismatched inputs.
    pub fn disambiguate(
        &self,
        tokens: &[&str],
        candidates: &[Vec<NativeAnalysis>],
    ) -> Result<NativeDisambiguation, DisambiguationError> {
        if tokens.is_empty() {
            return Err(failure("disambiguation cannot run on an empty sentence"));
        }
        if tokens.len() != candidates.len() {
            return Err(failure("token and candidate counts differ"));
        }
        Decoder::new(&self.model, tokens, candidates).decode()
    }

    /// Selects exact candidate indexes from precomputed ambiguity metadata.
    ///
    /// This metadata-free training hot path preserves the same Viterbi
    /// features, Java hashes, state deduplication, and tie behavior as
    /// [`Self::disambiguate`] without cloning full [`NativeAnalysis`] graphs.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty sentence or an empty candidate list.
    pub fn disambiguate_indices(
        &self,
        candidates: &[&[AmbiguityWordData]],
    ) -> Result<Vec<usize>, DisambiguationError> {
        CompactDecoder::new(&self.model, &self.begin, &self.end, candidates, None)?.decode(None)
    }

    /// Selects exact candidate indexes while reusing complete trigram scores.
    ///
    /// # Errors
    ///
    /// Returns an error for empty, mismatched, or malformed candidate metadata.
    pub fn disambiguate_indices_scored(
        &self,
        candidates: &[&[AmbiguityWordData]],
        scoring_codes: &[&[AmbiguityScoringCode]],
        cache: &mut DisambiguationScoreCache,
    ) -> Result<Vec<usize>, DisambiguationError> {
        CompactDecoder::new(
            &self.model,
            &self.begin,
            &self.end,
            candidates,
            Some(scoring_codes),
        )?
        .decode(Some(cache))
    }

    /// Selects candidate indexes causally from left to right.
    ///
    /// Deterministic tokens with exactly one candidate bypass perceptron scoring.
    /// Ambiguous tokens are scored only from the two already-selected analyses
    /// immediately to their left and the current candidate. No future token is
    /// inspected, so this path is safe for autoregressive tokenization.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty sentence or an empty candidate list.
    pub fn disambiguate_indices_causal(
        &self,
        candidates: &[&[AmbiguityWordData]],
    ) -> Result<Vec<usize>, DisambiguationError> {
        CompactDecoder::new(&self.model, &self.begin, &self.end, candidates, None)?
            .decode_causal(None)
    }

    /// Selects candidate indexes causally while reusing exact trigram scores.
    ///
    /// Single-candidate tokens do not invoke the perceptron. For every ambiguous
    /// token, only `(selected[t-2], selected[t-1], candidate[t])` is scored.
    ///
    /// # Errors
    ///
    /// Returns an error for empty, mismatched, or malformed candidate metadata.
    pub fn disambiguate_indices_scored_causal(
        &self,
        candidates: &[&[AmbiguityWordData]],
        scoring_codes: &[&[AmbiguityScoringCode]],
        cache: &mut DisambiguationScoreCache,
    ) -> Result<Vec<usize>, DisambiguationError> {
        CompactDecoder::new(
            &self.model,
            &self.begin,
            &self.end,
            candidates,
            Some(scoring_codes),
        )?
        .decode_causal(Some(cache))
    }

    /// Scores current-token candidates from selected left context only.
    ///
    /// Rows for deterministic single-candidate tokens contain `0.0`, because
    /// their choice is fixed and no contextual scorer call is needed.
    ///
    /// # Errors
    ///
    /// Returns an error for empty inputs, cardinality mismatches, or a selected
    /// candidate index outside its token candidate list.
    pub fn causal_candidate_scores(
        &self,
        candidates: &[&[AmbiguityWordData]],
        selected: &[usize],
    ) -> Result<Vec<Vec<f32>>, DisambiguationError> {
        let decoder = CompactDecoder::new(&self.model, &self.begin, &self.end, candidates, None)?;
        decoder.causal_candidate_scores(selected)
    }

    /// Scores every candidate conditionally on the selected Viterbi path.
    ///
    /// For candidate `c` at position `i`, this returns the sum of every trigram
    /// factor whose value would change if only position `i` were replaced by
    /// `c` while all other positions remained on `selected`. These are exact
    /// perceptron log-scores, not calibrated probabilities; callers may apply
    /// a temperature softmax for segmentation sampling.
    ///
    /// # Errors
    ///
    /// Returns an error for empty inputs, cardinality mismatches, or a selected
    /// candidate index outside its token candidate list.
    pub fn conditional_candidate_scores(
        &self,
        candidates: &[&[AmbiguityWordData]],
        selected: &[usize],
    ) -> Result<Vec<Vec<f32>>, DisambiguationError> {
        if candidates.is_empty() {
            return Err(failure(
                "conditional scoring cannot run on an empty sentence",
            ));
        }
        if candidates.len() != selected.len() {
            return Err(failure("conditional scoring selected cardinality differs"));
        }
        for (values, &choice) in candidates.iter().zip(selected) {
            if values.is_empty() || choice >= values.len() {
                return Err(failure(
                    "conditional scoring selected candidate is out of range",
                ));
            }
        }

        let n = candidates.len();
        let selected_word =
            |index: usize| -> &AmbiguityWordData { &candidates[index][selected[index]] };
        let mut scratch = FeatureScratch::new();
        let mut rows = Vec::with_capacity(n);
        for (index, values) in candidates.iter().enumerate() {
            let mut row = Vec::with_capacity(values.len());
            for candidate in *values {
                let older = if index >= 2 {
                    selected_word(index - 2)
                } else {
                    &self.begin
                };
                let previous = if index >= 1 {
                    selected_word(index - 1)
                } else {
                    &self.begin
                };
                let mut score =
                    trigram_score_ambiguity(&self.model, older, previous, candidate, &mut scratch);

                if index + 1 < n {
                    score += trigram_score_ambiguity(
                        &self.model,
                        previous,
                        candidate,
                        selected_word(index + 1),
                        &mut scratch,
                    );
                }
                if index + 2 < n {
                    score += trigram_score_ambiguity(
                        &self.model,
                        candidate,
                        selected_word(index + 1),
                        selected_word(index + 2),
                        &mut scratch,
                    );
                }

                if index + 1 == n {
                    score += trigram_score_ambiguity(
                        &self.model,
                        previous,
                        candidate,
                        &self.end,
                        &mut scratch,
                    );
                } else if index + 2 == n {
                    score += trigram_score_ambiguity(
                        &self.model,
                        candidate,
                        selected_word(index + 1),
                        &self.end,
                        &mut scratch,
                    );
                }
                row.push(score);
            }
            rows.push(row);
        }
        Ok(rows)
    }
}

impl fmt::Display for DisambiguationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for DisambiguationError {}

struct CompactDecoder<'a> {
    model: &'a PerceptronModel,
    begin: &'a AmbiguityWordData,
    end: &'a AmbiguityWordData,
    analyses: Vec<&'a AmbiguityWordData>,
    scoring_codes: Option<Vec<&'a AmbiguityScoringCode>>,
    token_starts: Vec<usize>,
    begin_code: AmbiguityScoringCode,
    end_code: AmbiguityScoringCode,
}

impl<'a> CompactDecoder<'a> {
    fn new(
        model: &'a PerceptronModel,
        begin: &'a AmbiguityWordData,
        end: &'a AmbiguityWordData,
        candidates: &[&'a [AmbiguityWordData]],
        scoring_codes: Option<&[&'a [AmbiguityScoringCode]]>,
    ) -> Result<Self, DisambiguationError> {
        if candidates.is_empty() {
            return Err(failure("disambiguation cannot run on an empty sentence"));
        }
        if let Some(codes) = scoring_codes {
            if codes.len() != candidates.len()
                || codes
                    .iter()
                    .zip(candidates)
                    .any(|(left, right)| left.len() != right.len())
            {
                return Err(failure("compact scoring-code cardinalities differ"));
            }
        }
        let candidate_count = candidates.iter().try_fold(0_usize, |total, values| {
            if values.is_empty() {
                return Err(failure("compact disambiguation candidate list is empty"));
            }
            total
                .checked_add(values.len())
                .ok_or_else(|| failure("compact disambiguation candidate count overflow"))
        })?;
        let mut analyses = Vec::with_capacity(candidate_count);
        let mut flattened_codes = scoring_codes.map(|_| Vec::with_capacity(candidate_count));
        let mut token_starts = Vec::with_capacity(candidates.len() + 1);
        token_starts.push(2);
        for (token_index, values) in candidates.iter().enumerate() {
            for (candidate_index, value) in values.iter().enumerate() {
                analyses.push(value);
                if let (Some(codes), Some(flattened)) = (scoring_codes, flattened_codes.as_mut()) {
                    flattened.push(&codes[token_index][candidate_index]);
                }
            }
            token_starts.push(
                analyses
                    .len()
                    .checked_add(2)
                    .ok_or_else(|| failure("compact analysis index overflow"))?,
            );
        }
        let begin_code = AmbiguityScoringCode::boundary(
            TRIGRAM_SIGNATURE_MASK as u32,
            u32::MAX,
            begin.java_hash,
        );
        let end_code = AmbiguityScoringCode::boundary(
            TRIGRAM_END_SIGNATURE as u32,
            u32::MAX - 1,
            end.java_hash,
        );
        Ok(Self {
            model,
            begin,
            end,
            analyses,
            scoring_codes: flattened_codes,
            token_starts,
            begin_code,
            end_code,
        })
    }

    fn word(&self, index: usize) -> &AmbiguityWordData {
        match index {
            0 => self.begin,
            1 => self.end,
            _ => self.analyses[index - 2],
        }
    }

    fn scoring_code(&self, index: usize) -> Option<AmbiguityScoringCode> {
        match index {
            0 => Some(self.begin_code),
            1 => Some(self.end_code),
            _ => self.scoring_codes.as_ref().map(|codes| *codes[index - 2]),
        }
    }

    #[inline(always)]
    fn token_count(&self) -> usize {
        self.token_starts.len() - 1
    }

    #[inline(always)]
    fn token_bounds(&self, index: usize) -> (usize, usize) {
        (self.token_starts[index], self.token_starts[index + 1])
    }

    #[inline(always)]
    fn token_len(&self, index: usize) -> usize {
        let (start, end) = self.token_bounds(index);
        end - start
    }

    fn initial_score_row(&self, score_cache: &mut Option<&mut DisambiguationScoreCache>) -> u32 {
        let Some(runtime) = score_cache.as_deref_mut() else {
            return INVALID_SCORE_ROW;
        };
        let Some(begin) = self.scoring_code(0) else {
            return INVALID_SCORE_ROW;
        };
        let Some(pair_key) =
            pack_cached_trigram_pair_key([begin.cache_signature, begin.cache_signature])
        else {
            return INVALID_SCORE_ROW;
        };
        runtime
            .trigrams
            .ensure_row(pair_key)
            .unwrap_or(INVALID_SCORE_ROW)
    }

    fn compact_state_identity(&self, previous: usize, current: usize) -> Option<(u64, i32)> {
        let previous = self.scoring_code(previous)?;
        let current = self.scoring_code(current)?;
        let key = (u64::from(previous.canonical) << 32) | u64::from(current.canonical);
        let hash = previous
            .java_hash
            .wrapping_mul(31)
            .wrapping_add(current.java_hash);
        Some((key, hash))
    }

    fn decoder_capacities(&self) -> (usize, usize, usize) {
        let mut hypothesis_capacity = 1_usize;
        let mut max_active_states = 1_usize;
        for index in 0..self.token_count() {
            let current = self.token_len(index);
            let previous = index
                .checked_sub(1)
                .map_or(1, |value| self.token_len(value));
            let older = index
                .checked_sub(2)
                .map_or(1, |value| self.token_len(value));
            let transitions = older.saturating_mul(previous).saturating_mul(current);
            hypothesis_capacity = hypothesis_capacity.saturating_add(transitions.saturating_mul(2));
            max_active_states = max_active_states.max(previous.saturating_mul(current));
        }
        let active_capacity = max_active_states
            .saturating_mul(4)
            .max(8)
            .checked_next_power_of_two()
            .unwrap_or(usize::MAX / 2 + 1);
        (
            hypothesis_capacity.max(8),
            active_capacity,
            max_active_states,
        )
    }

    fn decode_causal(
        &self,
        mut score_cache: Option<&mut DisambiguationScoreCache>,
    ) -> Result<Vec<usize>, DisambiguationError> {
        let mut selected = Vec::with_capacity(self.token_count());
        let mut selected_global = Vec::with_capacity(self.token_count());
        let mut prepared = None;
        let mut feature_scratch = FeatureScratch::new();

        for token_index in 0..self.token_count() {
            let (analysis_start, analysis_end) = self.token_bounds(token_index);
            if analysis_start == analysis_end {
                return Err(failure("compact disambiguation candidate list is empty"));
            }
            if analysis_end - analysis_start == 1 {
                selected.push(0);
                selected_global.push(analysis_start);
                continue;
            }

            let older = token_index
                .checked_sub(2)
                .map_or(0, |index| selected_global[index]);
            let previous = token_index
                .checked_sub(1)
                .map_or(0, |index| selected_global[index]);
            let mut best_analysis = analysis_start;
            let mut best_score = None::<f32>;
            for analysis in analysis_start..analysis_end {
                let score = score_compact_transition(
                    self,
                    &mut prepared,
                    token_index,
                    older,
                    previous,
                    analysis,
                    INVALID_SCORE_ROW,
                    &mut score_cache,
                    false,
                    &mut feature_scratch,
                )
                .score;
                if best_score.is_none_or(|current| current < score) {
                    best_analysis = analysis;
                    best_score = Some(score);
                }
            }
            selected.push(best_analysis - analysis_start);
            selected_global.push(best_analysis);
        }
        Ok(selected)
    }

    fn causal_candidate_scores(
        &self,
        selected: &[usize],
    ) -> Result<Vec<Vec<f32>>, DisambiguationError> {
        if selected.len() != self.token_count() {
            return Err(failure("causal scoring selected cardinality differs"));
        }
        let mut selected_global = Vec::with_capacity(self.token_count());
        for (token_index, &choice) in selected.iter().enumerate() {
            let (start, end) = self.token_bounds(token_index);
            if choice >= end - start {
                return Err(failure("causal scoring selected candidate is out of range"));
            }
            selected_global.push(start + choice);
        }

        let mut rows = Vec::with_capacity(self.token_count());
        let mut prepared = None;
        let mut feature_scratch = FeatureScratch::new();
        let mut no_cache = None;
        for token_index in 0..self.token_count() {
            let (analysis_start, analysis_end) = self.token_bounds(token_index);
            if analysis_end - analysis_start == 1 {
                rows.push(vec![0.0]);
                continue;
            }
            let older = token_index
                .checked_sub(2)
                .map_or(0, |index| selected_global[index]);
            let previous = token_index
                .checked_sub(1)
                .map_or(0, |index| selected_global[index]);
            let mut row = Vec::with_capacity(analysis_end - analysis_start);
            for analysis in analysis_start..analysis_end {
                row.push(
                    score_compact_transition(
                        self,
                        &mut prepared,
                        token_index,
                        older,
                        previous,
                        analysis,
                        INVALID_SCORE_ROW,
                        &mut no_cache,
                        false,
                        &mut feature_scratch,
                    )
                    .score,
                );
            }
            rows.push(row);
        }
        Ok(rows)
    }

    fn decode(
        self,
        mut score_cache: Option<&mut DisambiguationScoreCache>,
    ) -> Result<Vec<usize>, DisambiguationError> {
        let mut workspace = score_cache
            .as_deref_mut()
            .map_or_else(CompactDecodeWorkspace::default, |runtime| {
                std::mem::take(&mut runtime.decoder)
            });
        let dense_enabled = self.scoring_codes.is_some() && !force_hash_decoder();
        let result = if dense_enabled {
            if let Some(runtime) = score_cache.as_deref_mut() {
                runtime.dense_attempts = runtime.dense_attempts.saturating_add(1);
            }
            match self.decode_dense_with_workspace(&mut score_cache, &mut workspace)? {
                DenseDecodeOutcome::Selected(selected) => {
                    if let Some(runtime) = score_cache.as_deref_mut() {
                        runtime.dense_successes = runtime.dense_successes.saturating_add(1);
                    }
                    Ok(selected)
                }
                DenseDecodeOutcome::Fallback(reason) => {
                    if let Some(runtime) = score_cache.as_deref_mut() {
                        match reason {
                            DenseFallback::DuplicateIdentity => {
                                runtime.dense_duplicate_fallbacks =
                                    runtime.dense_duplicate_fallbacks.saturating_add(1);
                            }
                            DenseFallback::StateSpace => {
                                runtime.dense_state_fallbacks =
                                    runtime.dense_state_fallbacks.saturating_add(1);
                            }
                            DenseFallback::Tie => {
                                runtime.dense_tie_fallbacks =
                                    runtime.dense_tie_fallbacks.saturating_add(1);
                            }
                        }
                    }
                    self.decode_hash_with_workspace(&mut score_cache, &mut workspace)
                }
            }
        } else {
            self.decode_hash_with_workspace(&mut score_cache, &mut workspace)
        };
        if let Some(runtime) = score_cache.as_deref_mut() {
            runtime.decoder = workspace;
        }
        result
    }

    fn decode_hash_with_workspace(
        &self,
        score_cache: &mut Option<&mut DisambiguationScoreCache>,
        workspace: &mut CompactDecodeWorkspace,
    ) -> Result<Vec<usize>, DisambiguationError> {
        let (hypothesis_capacity, active_capacity, _) = self.decoder_capacities();
        workspace.prepare(hypothesis_capacity, active_capacity);
        let root_score_row = self.initial_score_row(score_cache);
        workspace
            .hypotheses
            .push(CompactHypothesis::root(root_score_row));
        workspace.current.add(0, &workspace.hypotheses, self);
        let mut prepared = None;
        let mut feature_scratch = FeatureScratch::new();

        for token_index in 0..self.token_count() {
            workspace.next.reset();
            let active_count = workspace.current.snapshot();
            let (analysis_start, analysis_end) = self.token_bounds(token_index);
            for analysis in analysis_start..analysis_end {
                for offset in 0..active_count {
                    let hypothesis_index = usize::try_from(workspace.current.scratch[offset])
                        .expect("u32 hypothesis index must fit usize");
                    let current_hypothesis = workspace.hypotheses[hypothesis_index];
                    let transition = score_compact_transition(
                        self,
                        &mut prepared,
                        token_index,
                        current_hypothesis.previous_analysis(),
                        current_hypothesis.current_analysis(),
                        analysis,
                        current_hypothesis.score_row,
                        score_cache,
                        true,
                        &mut feature_scratch,
                    );
                    let score = current_hypothesis.score + transition.score;
                    let candidate = CompactHypothesis::new(
                        current_hypothesis.current_analysis(),
                        analysis,
                        hypothesis_index,
                        transition.next_row,
                        score,
                    )?;
                    workspace
                        .next
                        .add_candidate(candidate, &mut workspace.hypotheses, self);
                }
            }
            std::mem::swap(&mut workspace.current, &mut workspace.next);
        }

        for hypothesis_index in workspace.current.iter() {
            let terminal = workspace.hypotheses[hypothesis_index];
            workspace.hypotheses[hypothesis_index].score += score_compact_transition(
                self,
                &mut prepared,
                self.token_count(),
                terminal.previous_analysis(),
                terminal.current_analysis(),
                1,
                terminal.score_row,
                score_cache,
                true,
                &mut feature_scratch,
            )
            .score;
        }
        let best_index = workspace
            .current
            .best(&workspace.hypotheses)
            .ok_or_else(|| failure("compact disambiguation produced no hypothesis"))?;
        self.reconstruct_selected(&workspace.hypotheses, best_index)
    }

    fn decode_dense_with_workspace(
        &self,
        score_cache: &mut Option<&mut DisambiguationScoreCache>,
        workspace: &mut CompactDecodeWorkspace,
    ) -> Result<DenseDecodeOutcome, DisambiguationError> {
        const MAX_DENSE_STATES: usize = 4_096;
        for token_index in 0..self.token_count() {
            let (start, end) = self.token_bounds(token_index);
            let count = end - start;
            let previous_count = token_index
                .checked_sub(1)
                .map_or(1, |previous| self.token_len(previous));
            if previous_count.saturating_mul(count) > MAX_DENSE_STATES {
                return Ok(DenseDecodeOutcome::Fallback(DenseFallback::StateSpace));
            }
            for left in start..end {
                let left_code = self
                    .scoring_code(left)
                    .ok_or_else(|| failure("dense decoder requires scoring codes"))?;
                for right in left + 1..end {
                    let right_code = self
                        .scoring_code(right)
                        .ok_or_else(|| failure("dense decoder requires scoring codes"))?;
                    if left_code.canonical == right_code.canonical
                        && left_code.java_hash == right_code.java_hash
                    {
                        return Ok(DenseDecodeOutcome::Fallback(
                            DenseFallback::DuplicateIdentity,
                        ));
                    }
                }
            }
        }

        let (hypothesis_capacity, _, max_active_states) = self.decoder_capacities();
        workspace.prepare_dense(hypothesis_capacity, max_active_states);
        let root_score_row = self.initial_score_row(score_cache);
        workspace
            .hypotheses
            .push(CompactHypothesis::root(root_score_row));
        workspace.dense_current.push(0);
        let mut prepared = None;
        let mut feature_scratch = FeatureScratch::new();

        for token_index in 0..self.token_count() {
            let (analysis_start, analysis_end) = self.token_bounds(token_index);
            let current_count = analysis_end - analysis_start;
            let previous_start = token_index
                .checked_sub(1)
                .map_or(0, |previous| self.token_bounds(previous).0);
            let previous_count = token_index
                .checked_sub(1)
                .map_or(1, |previous| self.token_len(previous));
            let state_count = previous_count.saturating_mul(current_count);
            workspace.prepare_dense_layer(state_count);

            for analysis in analysis_start..analysis_end {
                let current_local = analysis - analysis_start;
                for offset in 0..workspace.dense_current.len() {
                    let hypothesis_index = usize::try_from(workspace.dense_current[offset])
                        .expect("u32 hypothesis index must fit usize");
                    let current_hypothesis = workspace.hypotheses[hypothesis_index];
                    let previous_local = if token_index == 0 {
                        0
                    } else {
                        current_hypothesis
                            .current_analysis()
                            .checked_sub(previous_start)
                            .ok_or_else(|| failure("dense previous candidate underflow"))?
                    };
                    if previous_local >= previous_count {
                        return Err(failure("dense previous candidate is outside token range"));
                    }
                    let slot = previous_local * current_count + current_local;
                    let transition = score_compact_transition(
                        self,
                        &mut prepared,
                        token_index,
                        current_hypothesis.previous_analysis(),
                        current_hypothesis.current_analysis(),
                        analysis,
                        current_hypothesis.score_row,
                        score_cache,
                        true,
                        &mut feature_scratch,
                    );
                    let candidate = CompactHypothesis::new(
                        current_hypothesis.current_analysis(),
                        analysis,
                        hypothesis_index,
                        transition.next_row,
                        current_hypothesis.score + transition.score,
                    )?;
                    let stored = workspace.dense_slots[slot];
                    if stored == EMPTY_ACTIVE_HYPOTHESIS {
                        let new_index = workspace.hypotheses.len();
                        workspace.hypotheses.push(candidate);
                        let packed = u32::try_from(new_index)
                            .map_err(|_| failure("dense hypothesis index exceeds u32"))?;
                        workspace.dense_slots[slot] = packed;
                        workspace.dense_next.push(packed);
                    } else {
                        let existing =
                            usize::try_from(stored).expect("u32 hypothesis index must fit usize");
                        let existing_score = workspace.hypotheses[existing].score;
                        if existing_score.is_nan()
                            || candidate.score.is_nan()
                            || existing_score == candidate.score
                        {
                            return Ok(DenseDecodeOutcome::Fallback(DenseFallback::Tie));
                        }
                        if existing_score < candidate.score {
                            workspace.hypotheses[existing] = candidate;
                        }
                    }
                }
            }
            std::mem::swap(&mut workspace.dense_current, &mut workspace.dense_next);
        }

        let mut best = None::<usize>;
        for packed in workspace.dense_current.iter().copied() {
            let hypothesis_index =
                usize::try_from(packed).expect("u32 hypothesis index must fit usize");
            let terminal = workspace.hypotheses[hypothesis_index];
            workspace.hypotheses[hypothesis_index].score += score_compact_transition(
                self,
                &mut prepared,
                self.token_count(),
                terminal.previous_analysis(),
                terminal.current_analysis(),
                1,
                terminal.score_row,
                score_cache,
                true,
                &mut feature_scratch,
            )
            .score;
            if let Some(current_best) = best {
                let best_score = workspace.hypotheses[current_best].score;
                let candidate_score = workspace.hypotheses[hypothesis_index].score;
                if best_score.is_nan() || candidate_score.is_nan() || best_score == candidate_score
                {
                    return Ok(DenseDecodeOutcome::Fallback(DenseFallback::Tie));
                }
                if best_score < candidate_score {
                    best = Some(hypothesis_index);
                }
            } else {
                best = Some(hypothesis_index);
            }
        }
        let best = best.ok_or_else(|| failure("dense decoder produced no hypothesis"))?;
        Ok(DenseDecodeOutcome::Selected(
            self.reconstruct_selected(&workspace.hypotheses, best)?,
        ))
    }

    fn reconstruct_selected(
        &self,
        hypotheses: &[CompactHypothesis],
        best_index: usize,
    ) -> Result<Vec<usize>, DisambiguationError> {
        let mut selected = Vec::with_capacity(self.token_count());
        let mut cursor = best_index;
        while let Some(previous) = hypotheses[cursor].backpointer() {
            let global = hypotheses[cursor].current_analysis();
            let reverse_token = self
                .token_count()
                .checked_sub(selected.len() + 1)
                .ok_or_else(|| failure("compact backpointer length mismatch"))?;
            let (start, end) = self.token_bounds(reverse_token);
            if global < start || global >= end {
                return Err(failure("compact selected candidate is not in token list"));
            }
            selected.push(global - start);
            cursor = previous;
        }
        selected.reverse();
        if selected.len() != self.token_count() {
            return Err(failure("compact disambiguation output length mismatch"));
        }
        Ok(selected)
    }
}

#[derive(Clone, Copy)]
enum DenseFallback {
    DuplicateIdentity,
    StateSpace,
    Tie,
}

enum DenseDecodeOutcome {
    Selected(Vec<usize>),
    Fallback(DenseFallback),
}

fn force_hash_decoder() -> bool {
    static FORCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FORCE.get_or_init(|| std::env::var_os("NEDO_FORCE_HASH_DECODER").is_some())
}

#[derive(Clone, Copy)]
struct PreparedFeature {
    identity: u32,
    java_hash: i32,
    weight: f32,
}

impl PreparedFeature {
    fn new(model: &PerceptronModel, key: &FeatureView<'_>, identity: u32) -> Self {
        let (initial_hash, java_hash) = key.initial_and_java_hash();
        Self {
            identity,
            java_hash,
            weight: model.get_virtual(key, initial_hash, java_hash),
        }
    }
}

// Exact worker-local transition automaton. Each active Viterbi state carries
// the row for its last two perceptron signatures. The common first four
// continuations stay inline; only larger rows use the overflow hash table.
const TRIGRAM_PAIR_SLOTS: usize = 1_048_576;
const TRIGRAM_ROW_CAPACITY: usize = 655_360;
const TRIGRAM_INLINE_TRANSITIONS: usize = 8;
const TRIGRAM_OVERFLOW_SLOTS: usize = 524_288;
const BALANCED_TRIGRAM_PAIR_SLOTS: usize = 524_288;
const BALANCED_TRIGRAM_ROW_CAPACITY: usize = 262_144;
const BALANCED_TRIGRAM_OVERFLOW_SLOTS: usize = 262_144;
const COMPACT_TRIGRAM_PAIR_SLOTS: usize = 131_072;
const COMPACT_TRIGRAM_ROW_CAPACITY: usize = 16_384;
const COMPACT_TRIGRAM_OVERFLOW_SLOTS: usize = 262_144;
const TRIGRAM_SIGNATURE_BITS: u32 = 21;
const TRIGRAM_SIGNATURE_MASK: u64 = (1_u64 << TRIGRAM_SIGNATURE_BITS) - 1;
const TRIGRAM_PAIR_MASK: u64 = (1_u64 << (TRIGRAM_SIGNATURE_BITS * 2)) - 1;
const TRIGRAM_END_SIGNATURE: u64 = TRIGRAM_SIGNATURE_MASK - 1;
const EMPTY_TRIGRAM_SCORE_KEY: u64 = u64::MAX;
const EMPTY_TRIGRAM_PAIR_KEY: u64 = u64::MAX;
const INVALID_SCORE_ROW: u32 = u32::MAX;
const EMPTY_TRANSITION_SIGNATURE: u32 = u32::MAX;
const INVALID_CACHE_SIGNATURE: u32 = TRIGRAM_SIGNATURE_MASK as u32 + 1;

#[derive(Clone, Copy)]
struct InlineTransition {
    third: u32,
    next_row: u32,
    score: f32,
}

impl InlineTransition {
    const EMPTY: Self = Self {
        third: EMPTY_TRANSITION_SIGNATURE,
        next_row: INVALID_SCORE_ROW,
        score: 0.0,
    };
}

#[derive(Clone, Copy)]
struct TransitionRow {
    entries: [InlineTransition; TRIGRAM_INLINE_TRANSITIONS],
    overflow: u32,
}

impl TransitionRow {
    const EMPTY: Self = Self {
        entries: [InlineTransition::EMPTY; TRIGRAM_INLINE_TRANSITIONS],
        overflow: 0,
    };

    #[inline(always)]
    fn get(&self, third: u32) -> Option<(f32, u32)> {
        for entry in &self.entries {
            if entry.third == third {
                return Some((entry.score, entry.next_row));
            }
            if entry.third == EMPTY_TRANSITION_SIGNATURE {
                return None;
            }
        }
        None
    }

    #[inline]
    fn insert_inline(&mut self, third: u32, score: f32, next_row: u32) -> bool {
        for entry in &mut self.entries {
            if entry.third == third {
                entry.score = score;
                entry.next_row = next_row;
                return false;
            }
            if entry.third == EMPTY_TRANSITION_SIGNATURE {
                *entry = InlineTransition {
                    third,
                    next_row,
                    score,
                };
                return true;
            }
        }
        self.overflow = 1;
        false
    }
}

#[derive(Clone, Copy)]
struct OverflowScoreEntry {
    key: u64,
    score: f32,
    next_row: u32,
}

impl OverflowScoreEntry {
    const EMPTY: Self = Self {
        key: EMPTY_TRIGRAM_SCORE_KEY,
        score: 0.0,
        next_row: INVALID_SCORE_ROW,
    };
}

struct OverflowScoreCache {
    entries: Vec<OverflowScoreEntry>,
    size: usize,
    slots: usize,
    slot_shift: u32,
    max_entries: usize,
}

impl Default for OverflowScoreCache {
    fn default() -> Self {
        Self::with_slots(TRIGRAM_OVERFLOW_SLOTS)
    }
}

impl OverflowScoreCache {
    fn with_slots(slots: usize) -> Self {
        debug_assert!(slots.is_power_of_two());
        Self {
            entries: Vec::new(),
            size: 0,
            slots,
            slot_shift: 64 - slots.ilog2(),
            max_entries: slots.saturating_mul(4) / 5,
        }
    }

    fn clear(&mut self) {
        self.entries = Vec::new();
        self.size = 0;
    }

    #[inline]
    fn get(&self, key: u64) -> Option<(f32, u32)> {
        if self.entries.is_empty() {
            return None;
        }
        let modulo = self.entries.len() - 1;
        let mut slot = trigram_overflow_slot(key, self.slot_shift);
        loop {
            let stored = self.entries[slot];
            if stored.key == key {
                return Some((stored.score, stored.next_row));
            }
            if stored.key == EMPTY_TRIGRAM_SCORE_KEY {
                return None;
            }
            slot = (slot + 1) & modulo;
        }
    }

    #[inline]
    fn insert(&mut self, key: u64, score: f32, next_row: u32) -> bool {
        if self.size >= self.max_entries {
            return false;
        }
        self.ensure_allocated();
        let modulo = self.entries.len() - 1;
        let mut slot = trigram_overflow_slot(key, self.slot_shift);
        loop {
            let stored = self.entries[slot];
            if stored.key == EMPTY_TRIGRAM_SCORE_KEY {
                self.entries[slot] = OverflowScoreEntry {
                    key,
                    score,
                    next_row,
                };
                self.size += 1;
                return true;
            }
            if stored.key == key {
                self.entries[slot].score = score;
                self.entries[slot].next_row = next_row;
                return false;
            }
            slot = (slot + 1) & modulo;
        }
    }

    fn ensure_allocated(&mut self) {
        if !self.entries.is_empty() {
            return;
        }
        self.entries = vec![OverflowScoreEntry::EMPTY; self.slots];
    }

    fn approximate_bytes(&self) -> usize {
        self.entries
            .capacity()
            .saturating_mul(std::mem::size_of::<OverflowScoreEntry>())
    }
}

struct TrigramScoreCache {
    pair_keys: Vec<u64>,
    pair_rows: Vec<u32>,
    rows: Vec<TransitionRow>,
    overflow: OverflowScoreCache,
    pair_slots: usize,
    pair_slot_shift: u32,
    row_capacity: usize,
    entries: usize,
    hits: u64,
    misses: u64,
}

impl Default for TrigramScoreCache {
    fn default() -> Self {
        Self::with_capacities(
            TRIGRAM_PAIR_SLOTS,
            TRIGRAM_ROW_CAPACITY,
            TRIGRAM_OVERFLOW_SLOTS,
        )
    }
}

impl TrigramScoreCache {
    fn with_capacities(pair_slots: usize, row_capacity: usize, overflow_slots: usize) -> Self {
        debug_assert!(pair_slots.is_power_of_two());
        debug_assert!(overflow_slots.is_power_of_two());
        Self {
            pair_keys: Vec::new(),
            pair_rows: Vec::new(),
            rows: Vec::new(),
            overflow: OverflowScoreCache::with_slots(overflow_slots),
            pair_slots,
            pair_slot_shift: 64 - pair_slots.ilog2(),
            row_capacity,
            entries: 0,
            hits: 0,
            misses: 0,
        }
    }

    fn clear(&mut self) {
        self.pair_keys = Vec::new();
        self.pair_rows = Vec::new();
        self.rows = Vec::new();
        self.overflow.clear();
        self.entries = 0;
        self.hits = 0;
        self.misses = 0;
    }

    fn ensure_row(&mut self, pair_key: u64) -> Option<u32> {
        self.ensure_allocated();
        let modulo = self.pair_keys.len() - 1;
        let mut slot = trigram_pair_slot(pair_key, self.pair_slot_shift);
        loop {
            let stored = self.pair_keys[slot];
            if stored == pair_key {
                return Some(self.pair_rows[slot]);
            }
            if stored == EMPTY_TRIGRAM_PAIR_KEY {
                if self.rows.len() == self.row_capacity {
                    return None;
                }
                let row = u32::try_from(self.rows.len()).ok()?;
                self.rows.push(TransitionRow::EMPTY);
                self.pair_keys[slot] = pair_key;
                self.pair_rows[slot] = row;
                return Some(row);
            }
            slot = (slot + 1) & modulo;
        }
    }

    #[inline(always)]
    fn get(&mut self, row: u32, third: u32, full_key: u64) -> Option<(f32, u32)> {
        let row_index = usize::try_from(row).ok()?;
        let row = self.rows.get(row_index)?;
        if let Some(value) = row.get(third) {
            self.hits = self.hits.saturating_add(1);
            return Some(value);
        }
        if row.overflow != 0 {
            if let Some(value) = self.overflow.get(full_key) {
                self.hits = self.hits.saturating_add(1);
                return Some(value);
            }
        }
        self.misses = self.misses.saturating_add(1);
        None
    }

    #[inline]
    fn insert(&mut self, row: u32, third: u32, full_key: u64, score: f32, next_row: u32) {
        let Ok(row_index) = usize::try_from(row) else {
            return;
        };
        let Some(target) = self.rows.get_mut(row_index) else {
            return;
        };
        if target.insert_inline(third, score, next_row) {
            self.entries += 1;
            return;
        }
        if target.get(third).is_some() {
            return;
        }
        target.overflow = 1;
        if self.overflow.insert(full_key, score, next_row) {
            self.entries += 1;
        }
    }

    fn ensure_allocated(&mut self) {
        if !self.pair_keys.is_empty() {
            return;
        }
        debug_assert!(self.pair_slots.is_power_of_two());
        self.pair_keys = vec![EMPTY_TRIGRAM_PAIR_KEY; self.pair_slots];
        self.pair_rows = vec![INVALID_SCORE_ROW; self.pair_slots];
        self.rows = Vec::with_capacity(self.row_capacity);
    }
}

#[inline(always)]
fn trigram_pair_slot(key: u64, shift: u32) -> usize {
    let mixed = key.wrapping_mul(0xd6e8_feb8_6659_fd93);
    usize::try_from(mixed >> shift).unwrap_or(0)
}

#[inline(always)]
fn trigram_overflow_slot(key: u64, shift: u32) -> usize {
    let mixed = key.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    usize::try_from(mixed >> shift).unwrap_or(0)
}

#[inline(always)]
fn pack_cached_trigram_score_key(signatures: [u32; 3]) -> Option<u64> {
    if signatures[0] > TRIGRAM_SIGNATURE_MASK as u32
        || signatures[1] > TRIGRAM_SIGNATURE_MASK as u32
        || signatures[2] > TRIGRAM_SIGNATURE_MASK as u32
    {
        return None;
    }
    Some(
        u64::from(signatures[0])
            | (u64::from(signatures[1]) << TRIGRAM_SIGNATURE_BITS)
            | (u64::from(signatures[2]) << (TRIGRAM_SIGNATURE_BITS * 2)),
    )
}

#[inline(always)]
fn pack_cached_trigram_pair_key(signatures: [u32; 2]) -> Option<u64> {
    if signatures[0] > TRIGRAM_SIGNATURE_MASK as u32
        || signatures[1] > TRIGRAM_SIGNATURE_MASK as u32
    {
        return None;
    }
    Some(u64::from(signatures[0]) | (u64::from(signatures[1]) << TRIGRAM_SIGNATURE_BITS))
}

#[inline(always)]
#[cfg(test)]
fn pack_trigram_score_key(signatures: [u32; 3]) -> Option<u64> {
    let first = pack_trigram_signature(signatures[0])?;
    let second = pack_trigram_signature(signatures[1])?;
    let third = pack_trigram_signature(signatures[2])?;
    Some(first | (second << TRIGRAM_SIGNATURE_BITS) | (third << (TRIGRAM_SIGNATURE_BITS * 2)))
}

#[inline(always)]
#[cfg(test)]
fn pack_trigram_pair_key(signatures: [u32; 2]) -> Option<u64> {
    let first = pack_trigram_signature(signatures[0])?;
    let second = pack_trigram_signature(signatures[1])?;
    Some(first | (second << TRIGRAM_SIGNATURE_BITS))
}

#[inline(always)]
#[cfg(test)]
fn pack_trigram_signature(signature: u32) -> Option<u64> {
    if signature == u32::MAX {
        Some(TRIGRAM_SIGNATURE_MASK)
    } else if signature == u32::MAX - 1 {
        Some(TRIGRAM_END_SIGNATURE)
    } else if u64::from(signature) < TRIGRAM_END_SIGNATURE {
        Some(u64::from(signature))
    } else {
        None
    }
}

struct PreparedWordFeatures {
    f4: PreparedFeature,
    f10: PreparedFeature,
    f10b: PreparedFeature,
    f10c: PreparedFeature,
    f20: Vec<PreparedFeature>,
    f22: PreparedFeature,
}

impl PreparedWordFeatures {
    fn new(model: &PerceptronModel, word: &AmbiguityWordData) -> Self {
        let f20 = word
            .igs
            .iter()
            .enumerate()
            .map(|(index, ig)| {
                let identity = u32::try_from(index).unwrap_or(u32::MAX) & 0x00ff_ffff;
                PreparedFeature::new(
                    model,
                    &FeatureView::F20 { index, ig },
                    0x2000_0000 | identity,
                )
            })
            .collect();
        Self {
            f4: PreparedFeature::new(
                model,
                &FeatureView::F4 {
                    lemma3: &word.lemma,
                    igs3: &word.igs,
                },
                0x0400_0000,
            ),
            f10: PreparedFeature::new(
                model,
                &FeatureView::F10 {
                    lemma3: &word.lemma,
                },
                0x0a00_0000,
            ),
            f10b: PreparedFeature::new(
                model,
                &FeatureView::F10b {
                    lemma2: &word.lemma,
                },
                0x0b00_0000,
            ),
            f10c: PreparedFeature::new(
                model,
                &FeatureView::F10c {
                    lemma1: &word.lemma,
                },
                0x0c00_0000,
            ),
            f20,
            f22: PreparedFeature::new(
                model,
                &FeatureView::F22 {
                    count: word.igs.len(),
                },
                0x2200_0000,
            ),
        }
    }
}

struct PreparedPairFeatures {
    f3: PreparedFeature,
    f9: PreparedFeature,
    f17: Vec<PreparedFeature>,
}

impl PreparedPairFeatures {
    fn new(model: &PerceptronModel, word2: &AmbiguityWordData, word3: &AmbiguityWordData) -> Self {
        let last2 = word2.igs.last().map_or("", String::as_str);
        let f17 = word3
            .igs
            .iter()
            .enumerate()
            .map(|(index, ig)| {
                let representative = word3.igs[..index]
                    .iter()
                    .position(|previous| previous == ig)
                    .unwrap_or(index);
                let representative =
                    u32::try_from(representative).unwrap_or(u32::MAX) & 0x00ff_ffff;
                PreparedFeature::new(
                    model,
                    &FeatureView::F17 { last2, ig },
                    0x1700_0000 | representative,
                )
            })
            .collect();
        Self {
            f3: PreparedFeature::new(
                model,
                &FeatureView::F3 {
                    lemma2: &word2.lemma,
                    igs2: &word2.igs,
                    lemma3: &word3.lemma,
                    igs3: &word3.igs,
                },
                0x0300_0000,
            ),
            f9: PreparedFeature::new(
                model,
                &FeatureView::F9 {
                    lemma2: &word2.lemma,
                    lemma3: &word3.lemma,
                },
                0x0900_0000,
            ),
            f17,
        }
    }
}

struct PreparedPairLayer {
    previous_base: usize,
    current_base: usize,
    current_count: usize,
    pairs: Vec<Option<PreparedPairFeatures>>,
}

impl PreparedPairLayer {
    fn slot(&self, previous_analysis: usize, current_analysis: usize) -> usize {
        let previous = previous_analysis - self.previous_base;
        let current = current_analysis - self.current_base;
        previous * self.current_count + current
    }
}

struct CompactPreparedFeatures {
    words: Vec<Option<PreparedWordFeatures>>,
    pair_layers: Vec<PreparedPairLayer>,
}

impl CompactPreparedFeatures {
    fn new(decoder: &CompactDecoder<'_>) -> Self {
        let words = std::iter::repeat_with(|| None)
            .take(decoder.analyses.len() + 2)
            .collect();
        let token_count = decoder.token_count();
        let mut pair_layers = Vec::with_capacity(token_count + 1);
        for position in 0..=token_count {
            let (previous_start, previous_end) = if position == 0 {
                (0, 1)
            } else {
                decoder.token_bounds(position - 1)
            };
            let (current_start, current_end) = if position == token_count {
                (1, 2)
            } else {
                decoder.token_bounds(position)
            };
            let current_count = current_end - current_start;
            let pair_count = (previous_end - previous_start).saturating_mul(current_count);
            let pairs = std::iter::repeat_with(|| None).take(pair_count).collect();
            pair_layers.push(PreparedPairLayer {
                previous_base: previous_start,
                current_base: current_start,
                current_count,
                pairs,
            });
        }
        Self { words, pair_layers }
    }

    fn ensure_word(&mut self, decoder: &CompactDecoder<'_>, index: usize) {
        if self.words[index].is_none() {
            self.words[index] = Some(PreparedWordFeatures::new(
                decoder.model,
                decoder.word(index),
            ));
        }
    }

    fn ensure_pair(
        &mut self,
        decoder: &CompactDecoder<'_>,
        layer: usize,
        previous_analysis: usize,
        current_analysis: usize,
    ) -> usize {
        let slot = self.pair_layers[layer].slot(previous_analysis, current_analysis);
        if self.pair_layers[layer].pairs[slot].is_none() {
            self.pair_layers[layer].pairs[slot] = Some(PreparedPairFeatures::new(
                decoder.model,
                decoder.word(previous_analysis),
                decoder.word(current_analysis),
            ));
        }
        slot
    }

    fn score_uncached(
        &mut self,
        decoder: &CompactDecoder<'_>,
        layer: usize,
        word1_index: usize,
        word2_index: usize,
        word3_index: usize,
        score_cache: &mut Option<&mut DisambiguationScoreCache>,
        scratch: &mut FeatureScratch,
    ) -> f32 {
        let pair_slot = self.ensure_pair(decoder, layer, word2_index, word3_index);
        let word1 = decoder.word(word1_index);
        let word2 = decoder.word(word2_index);
        let word3 = decoder.word(word3_index);

        let cached = if let Some(runtime) = score_cache.as_deref_mut() {
            match (
                decoder.scoring_code(word1_index),
                decoder.scoring_code(word2_index),
                decoder.scoring_code(word3_index),
            ) {
                (Some(code1), Some(code2), Some(code3)) => {
                    let first = runtime.ensure_prepared_word(code1.signature, decoder.model, word1);
                    let second =
                        runtime.ensure_prepared_word(code2.signature, decoder.model, word2);
                    let third = runtime.ensure_prepared_word(code3.signature, decoder.model, word3);
                    match (first, second, third) {
                        (Some(first), Some(second), Some(third)) => Some((first, second, third)),
                        _ => None,
                    }
                }
                _ => None,
            }
        } else {
            None
        };

        if let Some((first, second, third)) = cached {
            let runtime = score_cache.as_deref().expect("prepared-word cache runtime");
            let prepared1 = runtime.prepared_word(first).expect("cached prepared word1");
            let prepared2 = runtime
                .prepared_word(second)
                .expect("cached prepared word2");
            let prepared3 = runtime.prepared_word(third).expect("cached prepared word3");
            let pair = self.pair_layers[layer].pairs[pair_slot]
                .as_ref()
                .expect("prepared pair");
            return score_prepared_trigram(
                decoder, word1, word2, word3, prepared1, prepared2, prepared3, pair, scratch,
            );
        }

        self.ensure_word(decoder, word1_index);
        self.ensure_word(decoder, word2_index);
        self.ensure_word(decoder, word3_index);
        let prepared1 = self.words[word1_index].as_ref().expect("prepared word1");
        let prepared2 = self.words[word2_index].as_ref().expect("prepared word2");
        let prepared3 = self.words[word3_index].as_ref().expect("prepared word3");
        let pair = self.pair_layers[layer].pairs[pair_slot]
            .as_ref()
            .expect("prepared pair");
        score_prepared_trigram(
            decoder, word1, word2, word3, prepared1, prepared2, prepared3, pair, scratch,
        )
    }
}

fn score_prepared_trigram(
    decoder: &CompactDecoder<'_>,
    word1: &AmbiguityWordData,
    word2: &AmbiguityWordData,
    word3: &AmbiguityWordData,
    prepared1: &PreparedWordFeatures,
    prepared2: &PreparedWordFeatures,
    prepared3: &PreparedWordFeatures,
    pair: &PreparedPairFeatures,
    scratch: &mut FeatureScratch,
) -> f32 {
    scratch.reset();
    scratch.add_feature(
        decoder.model,
        &FeatureView::F2 {
            lemma1: &word1.lemma,
            igs2: &word2.igs,
            lemma3: &word3.lemma,
            igs3: &word3.igs,
        },
        0x0200_0000,
    );
    scratch.add_prepared(pair.f3);
    scratch.add_prepared(prepared3.f4);
    scratch.add_prepared(pair.f9);
    scratch.add_prepared(prepared3.f10);
    scratch.add_prepared(prepared2.f10b);
    scratch.add_prepared(prepared1.f10c);
    let last1 = word1.igs.last().map_or("", String::as_str);
    let last2 = word2.igs.last().map_or("", String::as_str);
    for (index, ig) in word3.igs.iter().enumerate() {
        let representative = word3.igs[..index]
            .iter()
            .position(|previous| previous == ig)
            .unwrap_or(index);
        let representative = u32::try_from(representative).unwrap_or(u32::MAX) & 0x00ff_ffff;
        scratch.add_feature(
            decoder.model,
            &FeatureView::F15 { last1, last2, ig },
            0x1500_0000 | representative,
        );
    }
    for feature in &pair.f17 {
        scratch.add_prepared(*feature);
    }
    for feature in &prepared3.f20 {
        scratch.add_prepared(*feature);
    }
    scratch.add_prepared(prepared3.f22);
    scratch.score()
}

#[derive(Clone, Copy)]
struct CompactTransitionScore {
    score: f32,
    next_row: u32,
}

#[inline(always)]
fn score_compact_transition(
    decoder: &CompactDecoder<'_>,
    prepared: &mut Option<CompactPreparedFeatures>,
    layer: usize,
    word1_index: usize,
    word2_index: usize,
    word3_index: usize,
    current_row: u32,
    score_cache: &mut Option<&mut DisambiguationScoreCache>,
    need_next_row: bool,
    scratch: &mut FeatureScratch,
) -> CompactTransitionScore {
    let score_key = match (
        decoder.scoring_code(word1_index),
        decoder.scoring_code(word2_index),
        decoder.scoring_code(word3_index),
    ) {
        (Some(word1), Some(word2), Some(word3)) => pack_cached_trigram_score_key([
            word1.cache_signature,
            word2.cache_signature,
            word3.cache_signature,
        ]),
        _ => None,
    };

    let mut resolved_row = current_row;
    if let (Some(runtime), Some(key)) = (score_cache.as_deref_mut(), score_key) {
        if resolved_row == INVALID_SCORE_ROW {
            resolved_row = runtime
                .trigrams
                .ensure_row(key & TRIGRAM_PAIR_MASK)
                .unwrap_or(INVALID_SCORE_ROW);
        }
        if resolved_row != INVALID_SCORE_ROW {
            let third =
                u32::try_from((key >> (TRIGRAM_SIGNATURE_BITS * 2)) & TRIGRAM_SIGNATURE_MASK)
                    .unwrap_or(EMPTY_TRANSITION_SIGNATURE);
            if let Some((score, next_row)) = runtime.trigrams.get(resolved_row, third, key) {
                if !need_next_row || next_row != INVALID_SCORE_ROW {
                    return CompactTransitionScore { score, next_row };
                }
            }
        }
    }

    let prepared = prepared.get_or_insert_with(|| CompactPreparedFeatures::new(decoder));
    let score = prepared.score_uncached(
        decoder,
        layer,
        word1_index,
        word2_index,
        word3_index,
        score_cache,
        scratch,
    );
    let mut next_row = INVALID_SCORE_ROW;
    if let (Some(runtime), Some(key)) = (score_cache.as_deref_mut(), score_key) {
        if resolved_row == INVALID_SCORE_ROW {
            resolved_row = runtime
                .trigrams
                .ensure_row(key & TRIGRAM_PAIR_MASK)
                .unwrap_or(INVALID_SCORE_ROW);
        }
        if need_next_row {
            next_row = runtime
                .trigrams
                .ensure_row((key >> TRIGRAM_SIGNATURE_BITS) & TRIGRAM_PAIR_MASK)
                .unwrap_or(INVALID_SCORE_ROW);
        }
        if resolved_row != INVALID_SCORE_ROW {
            let third =
                u32::try_from((key >> (TRIGRAM_SIGNATURE_BITS * 2)) & TRIGRAM_SIGNATURE_MASK)
                    .unwrap_or(EMPTY_TRANSITION_SIGNATURE);
            runtime
                .trigrams
                .insert(resolved_row, third, key, score, next_row);
        }
    }
    CompactTransitionScore { score, next_row }
}

fn boundary_word_data(surface: &str) -> AmbiguityWordData {
    let (analysis, word) = WordData::unknown(surface);
    let java_hash = java_analysis_hash(&analysis);
    AmbiguityWordData {
        canonical: analysis.canonical,
        lemma: word.lemma,
        igs: word.igs,
        java_hash,
    }
}

#[derive(Default)]
struct CompactDecodeWorkspace {
    hypotheses: Vec<CompactHypothesis>,
    current: CompactActiveHypotheses,
    next: CompactActiveHypotheses,
    dense_slots: Vec<u32>,
    dense_current: Vec<u32>,
    dense_next: Vec<u32>,
}

impl CompactDecodeWorkspace {
    fn prepare(&mut self, hypothesis_capacity: usize, active_capacity: usize) {
        self.hypotheses.clear();
        if self.hypotheses.capacity() < hypothesis_capacity {
            self.hypotheses
                .reserve(hypothesis_capacity - self.hypotheses.capacity());
        }
        self.current.ensure_capacity(active_capacity);
        self.next.ensure_capacity(active_capacity);
        self.current.reset();
        self.next.reset();
    }

    fn prepare_dense(&mut self, hypothesis_capacity: usize, max_active_states: usize) {
        self.hypotheses.clear();
        if self.hypotheses.capacity() < hypothesis_capacity {
            self.hypotheses
                .reserve(hypothesis_capacity - self.hypotheses.capacity());
        }
        if self.dense_slots.len() < max_active_states {
            self.dense_slots
                .resize(max_active_states, EMPTY_ACTIVE_HYPOTHESIS);
        }
        self.dense_current.clear();
        self.dense_next.clear();
        if self.dense_current.capacity() < max_active_states {
            self.dense_current
                .reserve(max_active_states - self.dense_current.capacity());
        }
        if self.dense_next.capacity() < max_active_states {
            self.dense_next
                .reserve(max_active_states - self.dense_next.capacity());
        }
    }

    fn prepare_dense_layer(&mut self, state_count: usize) {
        self.dense_slots[..state_count].fill(EMPTY_ACTIVE_HYPOTHESIS);
        self.dense_next.clear();
    }

    fn approximate_bytes(&self) -> usize {
        self.hypotheses
            .capacity()
            .saturating_mul(std::mem::size_of::<CompactHypothesis>())
            .saturating_add(self.current.approximate_bytes())
            .saturating_add(self.next.approximate_bytes())
            .saturating_add(
                self.dense_slots
                    .capacity()
                    .saturating_add(self.dense_current.capacity())
                    .saturating_add(self.dense_next.capacity())
                    .saturating_mul(std::mem::size_of::<u32>()),
            )
    }
}

const NO_COMPACT_BACKPOINTER: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct CompactHypothesis {
    previous_analysis: u32,
    current_analysis: u32,
    backpointer: u32,
    score_row: u32,
    score: f32,
}

impl CompactHypothesis {
    const fn root(score_row: u32) -> Self {
        Self {
            previous_analysis: 0,
            current_analysis: 0,
            backpointer: NO_COMPACT_BACKPOINTER,
            score_row,
            score: 0.0,
        }
    }

    fn new(
        previous_analysis: usize,
        current_analysis: usize,
        backpointer: usize,
        score_row: u32,
        score: f32,
    ) -> Result<Self, DisambiguationError> {
        Ok(Self {
            previous_analysis: u32::try_from(previous_analysis)
                .map_err(|_| failure("compact previous-analysis index exceeds u32"))?,
            current_analysis: u32::try_from(current_analysis)
                .map_err(|_| failure("compact current-analysis index exceeds u32"))?,
            backpointer: u32::try_from(backpointer)
                .map_err(|_| failure("compact backpointer index exceeds u32"))?,
            score_row,
            score,
        })
    }

    #[inline(always)]
    fn previous_analysis(self) -> usize {
        usize::try_from(self.previous_analysis).expect("u32 analysis index must fit usize")
    }

    #[inline(always)]
    fn current_analysis(self) -> usize {
        usize::try_from(self.current_analysis).expect("u32 analysis index must fit usize")
    }

    #[inline(always)]
    fn backpointer(self) -> Option<usize> {
        (self.backpointer != NO_COMPACT_BACKPOINTER).then(|| {
            usize::try_from(self.backpointer).expect("u32 backpointer index must fit usize")
        })
    }
}

const EMPTY_ACTIVE_HYPOTHESIS: u32 = u32::MAX;

struct CompactActiveHypotheses {
    slots: Vec<u32>,
    scratch: Vec<u32>,
    active_len: usize,
    modulo: usize,
    size: usize,
    expand_limit: usize,
}

impl Default for CompactActiveHypotheses {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            scratch: Vec::new(),
            active_len: 0,
            modulo: 0,
            size: 0,
            expand_limit: 0,
        }
    }
}

impl CompactActiveHypotheses {
    fn ensure_capacity(&mut self, reserve: usize) {
        let reserve = reserve.max(8).next_power_of_two();
        if self.slots.len() < reserve {
            self.slots.resize(reserve, EMPTY_ACTIVE_HYPOTHESIS);
            self.scratch.resize(reserve, EMPTY_ACTIVE_HYPOTHESIS);
        }
    }

    fn approximate_bytes(&self) -> usize {
        self.slots
            .capacity()
            .saturating_add(self.scratch.capacity())
            .saturating_mul(std::mem::size_of::<u32>())
    }

    fn reset(&mut self) {
        self.slots[..self.active_len].fill(EMPTY_ACTIVE_HYPOTHESIS);
        self.active_len = 8;
        self.modulo = 7;
        self.size = 0;
        self.expand_limit = load_limit(8);
    }

    #[inline(always)]
    fn snapshot(&mut self) -> usize {
        let mut count = 0_usize;
        for index in self.slots[..self.active_len].iter().copied() {
            if index != EMPTY_ACTIVE_HYPOTHESIS {
                self.scratch[count] = index;
                count += 1;
            }
        }
        debug_assert_eq!(count, self.size);
        count
    }

    #[inline(always)]
    fn add(
        &mut self,
        index: usize,
        hypotheses: &[CompactHypothesis],
        decoder: &CompactDecoder<'_>,
    ) {
        let packed_index = u32::try_from(index).expect("compact hypothesis index exceeds u32");
        let location = self.locate(index, hypotheses, decoder);
        if let Some(slot) = location.existing {
            let existing = self.slots[slot];
            if existing == EMPTY_ACTIVE_HYPOTHESIS {
                return;
            }
            let existing = usize::try_from(existing).expect("u32 hypothesis index must fit usize");
            if hypotheses[existing].score < hypotheses[index].score {
                self.slots[slot] = packed_index;
            }
        } else {
            self.slots[location.slot] = packed_index;
            self.size += 1;
        }
        if self.size == self.expand_limit {
            self.expand(hypotheses, decoder);
        }
    }

    #[inline(always)]
    fn add_candidate(
        &mut self,
        candidate: CompactHypothesis,
        hypotheses: &mut Vec<CompactHypothesis>,
        decoder: &CompactDecoder<'_>,
    ) {
        if let Some((candidate_key, candidate_hash)) = decoder
            .compact_state_identity(candidate.previous_analysis(), candidate.current_analysis())
        {
            let mut slot = hash_slot(candidate_hash, self.modulo);
            loop {
                let stored = self.slots[slot];
                if stored == EMPTY_ACTIVE_HYPOTHESIS {
                    self.insert_candidate(slot, candidate, hypotheses, decoder);
                    return;
                }
                let existing =
                    usize::try_from(stored).expect("u32 hypothesis index must fit usize");
                let existing_hypothesis = hypotheses[existing];
                let existing_key = decoder
                    .compact_state_identity(
                        existing_hypothesis.previous_analysis(),
                        existing_hypothesis.current_analysis(),
                    )
                    .expect("scored active hypothesis must have compact identity")
                    .0;
                if existing_key == candidate_key {
                    if existing_hypothesis.score < candidate.score {
                        hypotheses[existing] = candidate;
                    }
                    return;
                }
                slot = (slot + 1) & self.modulo;
            }
        }

        let mut slot = hash_slot(compact_hypothesis_hash(&candidate, decoder), self.modulo);
        loop {
            let stored = self.slots[slot];
            if stored == EMPTY_ACTIVE_HYPOTHESIS {
                self.insert_candidate(slot, candidate, hypotheses, decoder);
                return;
            }
            let existing = usize::try_from(stored).expect("u32 hypothesis index must fit usize");
            if compact_hypothesis_equal(&hypotheses[existing], &candidate, decoder) {
                if hypotheses[existing].score < candidate.score {
                    hypotheses[existing] = candidate;
                }
                return;
            }
            slot = (slot + 1) & self.modulo;
        }
    }

    #[inline(always)]
    fn insert_candidate(
        &mut self,
        slot: usize,
        candidate: CompactHypothesis,
        hypotheses: &mut Vec<CompactHypothesis>,
        decoder: &CompactDecoder<'_>,
    ) {
        let index = hypotheses.len();
        hypotheses.push(candidate);
        self.slots[slot] = u32::try_from(index).expect("compact hypothesis index exceeds u32");
        self.size += 1;
        if self.size == self.expand_limit {
            self.expand(hypotheses, decoder);
        }
    }

    fn locate(
        &self,
        index: usize,
        hypotheses: &[CompactHypothesis],
        decoder: &CompactDecoder<'_>,
    ) -> ActiveLocation {
        let mut slot = hash_slot(
            compact_hypothesis_hash(&hypotheses[index], decoder),
            self.modulo,
        );
        loop {
            let stored = self.slots[slot];
            if stored == EMPTY_ACTIVE_HYPOTHESIS {
                return ActiveLocation {
                    slot,
                    existing: None,
                };
            }
            let existing = usize::try_from(stored).expect("u32 hypothesis index must fit usize");
            if compact_hypothesis_equal(&hypotheses[existing], &hypotheses[index], decoder) {
                return ActiveLocation {
                    slot,
                    existing: Some(slot),
                };
            }
            slot = (slot + 1) & self.modulo;
        }
    }

    fn expand(&mut self, hypotheses: &[CompactHypothesis], decoder: &CompactDecoder<'_>) {
        let old_len = self.active_len;
        let old_modulo = self.modulo;
        let expanded_len = old_len.saturating_mul(2);
        if expanded_len > self.slots.len() {
            self.slots.resize(expanded_len, EMPTY_ACTIVE_HYPOTHESIS);
            self.scratch.resize(expanded_len, EMPTY_ACTIVE_HYPOTHESIS);
        }
        self.scratch[..expanded_len].fill(EMPTY_ACTIVE_HYPOTHESIS);
        let mut expanded_size = 0_usize;
        for stored in self.slots[..old_len]
            .iter()
            .copied()
            .filter(|index| *index != EMPTY_ACTIVE_HYPOTHESIS)
        {
            let index = usize::try_from(stored).expect("u32 hypothesis index must fit usize");
            let mut slot = hash_slot(
                compact_hypothesis_hash(&hypotheses[index], decoder),
                old_modulo,
            );
            while self.scratch[slot] != EMPTY_ACTIVE_HYPOTHESIS {
                slot = (slot + 1) & old_modulo;
            }
            self.scratch[slot] = stored;
            expanded_size += 1;
        }
        std::mem::swap(&mut self.slots, &mut self.scratch);
        self.active_len = expanded_len;
        self.modulo = expanded_len - 1;
        self.size = expanded_size;
        self.expand_limit = load_limit(expanded_len);
    }

    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.slots[..self.active_len]
            .iter()
            .copied()
            .filter(|index| *index != EMPTY_ACTIVE_HYPOTHESIS)
            .map(|index| usize::try_from(index).expect("u32 hypothesis index must fit usize"))
    }

    fn best(&self, hypotheses: &[CompactHypothesis]) -> Option<usize> {
        let mut best: Option<usize> = None;
        for index in self.iter() {
            if best.is_none_or(|current| hypotheses[current].score < hypotheses[index].score) {
                best = Some(index);
            }
        }
        best
    }
}

fn compact_hypothesis_hash(hypothesis: &CompactHypothesis, decoder: &CompactDecoder<'_>) -> i32 {
    if let (Some(previous), Some(current)) = (
        decoder.scoring_code(hypothesis.previous_analysis()),
        decoder.scoring_code(hypothesis.current_analysis()),
    ) {
        previous
            .java_hash
            .wrapping_mul(31)
            .wrapping_add(current.java_hash)
    } else {
        decoder
            .word(hypothesis.previous_analysis())
            .java_hash
            .wrapping_mul(31)
            .wrapping_add(decoder.word(hypothesis.current_analysis()).java_hash)
    }
}

fn compact_hypothesis_equal(
    left: &CompactHypothesis,
    right: &CompactHypothesis,
    decoder: &CompactDecoder<'_>,
) -> bool {
    match (
        decoder.scoring_code(left.previous_analysis()),
        decoder.scoring_code(left.current_analysis()),
        decoder.scoring_code(right.previous_analysis()),
        decoder.scoring_code(right.current_analysis()),
    ) {
        (Some(left_previous), Some(left_current), Some(right_previous), Some(right_current)) => {
            left_previous.canonical == right_previous.canonical
                && left_current.canonical == right_current.canonical
        }
        _ => {
            decoder.word(left.previous_analysis()).canonical
                == decoder.word(right.previous_analysis()).canonical
                && decoder.word(left.current_analysis()).canonical
                    == decoder.word(right.current_analysis()).canonical
        }
    }
}

#[derive(Clone, Debug)]
struct HashLevel {
    key_amount: usize,
    bucket_amount: usize,
    seeds: Vec<u8>,
    failed_indexes: Vec<usize>,
}

#[derive(Clone)]
struct AnalysisData {
    analysis: NativeAnalysis,
    word: WordData,
    java_hash: i32,
}

#[derive(Clone)]
struct WordData {
    lemma: String,
    igs: Vec<String>,
}

impl WordData {
    fn from_analysis(analysis: &NativeAnalysis) -> Self {
        let group_count = 1 + analysis
            .morphemes
            .iter()
            .filter(|morpheme| morpheme.derivational)
            .count();
        let mut first_group = String::new();
        for morpheme in &analysis.morphemes {
            if morpheme.derivational {
                break;
            }
            first_group.push_str(morpheme.mapped_id.as_deref().unwrap_or(&morpheme.id));
        }
        let prefix = secondary_pos_name(&analysis.secondary_pos);
        let mut igs = Vec::with_capacity(group_count);
        let mut first = String::with_capacity(prefix.len() + first_group.len());
        first.push_str(prefix);
        first.push_str(&first_group);
        igs.push(first);
        for _ in 1..group_count {
            igs.push(first_group.clone());
        }
        Self {
            lemma: analysis.lemma.clone(),
            igs,
        }
    }

    fn unknown(surface: &str) -> (NativeAnalysis, Self) {
        let morpheme = NativeMorpheme {
            id: "Unknown".to_owned(),
            name: "Unknown".to_owned(),
            surface: surface.to_owned(),
            derivational: false,
            informal: false,
            pos: None,
            mapped_id: None,
        };
        let analysis = NativeAnalysis {
            canonical: format!("UNK_Unk_Unk\u{1}Unknown={surface}\u{2}"),
            dictionary_id: "UNK_Unk_Unk".to_owned(),
            lemma: "UNK".to_owned(),
            primary_pos: "Unk".to_owned(),
            secondary_pos: "Unk".to_owned(),
            surface_form: surface.to_owned(),
            stem: surface.to_owned(),
            ending: String::new(),
            morphemes: vec![morpheme],
        };
        let word = Self {
            lemma: "UNK".to_owned(),
            igs: vec!["UnknownSecUnknown".to_owned()],
        };
        (analysis, word)
    }
}

struct Decoder<'a> {
    model: &'a PerceptronModel,
    analyses: Vec<AnalysisData>,
    token_candidates: Vec<Vec<usize>>,
}

impl<'a> Decoder<'a> {
    fn new(
        model: &'a PerceptronModel,
        tokens: &[&str],
        candidates: &[Vec<NativeAnalysis>],
    ) -> Self {
        let mut analyses = Vec::new();
        let (begin, begin_word) = WordData::unknown("<s>");
        analyses.push(AnalysisData::new(begin, begin_word));
        let (end, end_word) = WordData::unknown("</s>");
        analyses.push(AnalysisData::new(end, end_word));
        let mut token_candidates = Vec::with_capacity(tokens.len());
        for (token, word_candidates) in tokens.iter().zip(candidates) {
            let mut ids = Vec::new();
            if word_candidates.is_empty() {
                let (unknown_analysis, word) = WordData::unknown(token);
                ids.push(analyses.len());
                analyses.push(AnalysisData::new(unknown_analysis, word));
            } else {
                for analysis in word_candidates {
                    ids.push(analyses.len());
                    analyses.push(AnalysisData::new(
                        analysis.clone(),
                        WordData::from_analysis(analysis),
                    ));
                }
            }
            token_candidates.push(ids);
        }
        Self {
            model,
            analyses,
            token_candidates,
        }
    }

    fn decode(self) -> Result<NativeDisambiguation, DisambiguationError> {
        let mut hypotheses = vec![Hypothesis {
            previous_analysis: 0,
            current_analysis: 0,
            backpointer: None,
            score: 0.0,
        }];
        let mut current = ActiveHypotheses::new();
        current.add(0, &hypotheses, &self.analyses);
        let mut feature_scratch = FeatureScratch::new();

        for analyses_for_token in &self.token_candidates {
            let mut next = ActiveHypotheses::new();
            for &analysis in analyses_for_token {
                for hypothesis_index in current.iter() {
                    let current_hypothesis = &hypotheses[hypothesis_index];
                    let score = current_hypothesis.score
                        + trigram_score(
                            self.model,
                            &self.analyses[current_hypothesis.previous_analysis].word,
                            &self.analyses[current_hypothesis.current_analysis].word,
                            &self.analyses[analysis].word,
                            &mut feature_scratch,
                        );
                    let new_index = hypotheses.len();
                    hypotheses.push(Hypothesis {
                        previous_analysis: current_hypothesis.current_analysis,
                        current_analysis: analysis,
                        backpointer: Some(hypothesis_index),
                        score,
                    });
                    next.add(new_index, &hypotheses, &self.analyses);
                }
            }
            current = next;
        }

        for hypothesis_index in current.iter() {
            let terminal_hypothesis = &mut hypotheses[hypothesis_index];
            terminal_hypothesis.score += trigram_score(
                self.model,
                &self.analyses[terminal_hypothesis.previous_analysis].word,
                &self.analyses[terminal_hypothesis.current_analysis].word,
                &self.analyses[1].word,
                &mut feature_scratch,
            );
        }
        let best_index = current
            .best(&hypotheses)
            .ok_or_else(|| failure("disambiguation produced no hypothesis"))?;
        let score = hypotheses[best_index].score;
        let mut selected = Vec::with_capacity(self.token_candidates.len());
        let mut cursor = best_index;
        while let Some(previous) = hypotheses[cursor].backpointer {
            selected.push(
                self.analyses[hypotheses[cursor].current_analysis]
                    .analysis
                    .clone(),
            );
            cursor = previous;
        }
        selected.reverse();
        Ok(NativeDisambiguation {
            best: selected,
            score,
        })
    }
}

impl AnalysisData {
    fn new(analysis: NativeAnalysis, word: WordData) -> Self {
        let java_hash = java_analysis_hash(&analysis);
        Self {
            analysis,
            word,
            java_hash,
        }
    }
}

#[derive(Clone, Copy)]
struct Hypothesis {
    previous_analysis: usize,
    current_analysis: usize,
    backpointer: Option<usize>,
    score: f32,
}

struct ActiveHypotheses {
    slots: Vec<Option<usize>>,
    modulo: usize,
    size: usize,
    expand_limit: usize,
}

impl ActiveHypotheses {
    fn new() -> Self {
        Self::with_capacity(8)
    }

    fn with_capacity(capacity: usize) -> Self {
        let actual = capacity.next_power_of_two();
        Self {
            slots: vec![None; actual],
            modulo: actual - 1,
            size: 0,
            expand_limit: load_limit(actual),
        }
    }

    fn add(&mut self, index: usize, hypotheses: &[Hypothesis], analyses: &[AnalysisData]) {
        let location = self.locate(index, hypotheses, analyses);
        if let Some(slot) = location.existing {
            let Some(existing) = self.slots[slot] else {
                return;
            };
            if hypotheses[existing].score < hypotheses[index].score {
                self.slots[slot] = Some(index);
            }
        } else {
            self.slots[location.slot] = Some(index);
            self.size += 1;
        }
        if self.size == self.expand_limit {
            self.expand(hypotheses, analyses);
        }
    }

    fn locate(
        &self,
        index: usize,
        hypotheses: &[Hypothesis],
        analyses: &[AnalysisData],
    ) -> ActiveLocation {
        let mut slot = hash_slot(hypothesis_hash(&hypotheses[index], analyses), self.modulo);
        loop {
            match self.slots[slot] {
                None => {
                    return ActiveLocation {
                        slot,
                        existing: None,
                    }
                }
                Some(existing)
                    if hypothesis_equal(&hypotheses[existing], &hypotheses[index], analyses) =>
                {
                    return ActiveLocation {
                        slot,
                        existing: Some(slot),
                    };
                }
                Some(_) => slot = (slot + 1) & self.modulo,
            }
        }
    }

    fn expand(&mut self, hypotheses: &[Hypothesis], analyses: &[AnalysisData]) {
        let old_modulo = self.modulo;
        let mut expanded = Self::with_capacity(self.slots.len() * 2);
        for index in self.slots.iter().flatten().copied() {
            let mut slot = hash_slot(hypothesis_hash(&hypotheses[index], analyses), old_modulo);
            while expanded.slots[slot].is_some() {
                slot = (slot + 1) & old_modulo;
            }
            expanded.slots[slot] = Some(index);
            expanded.size += 1;
        }
        self.slots = expanded.slots;
        self.modulo = expanded.modulo;
        self.expand_limit = expanded.expand_limit;
    }

    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.slots.iter().filter_map(|slot| *slot)
    }

    fn best(&self, hypotheses: &[Hypothesis]) -> Option<usize> {
        let mut best: Option<usize> = None;
        for index in self.iter() {
            if best.is_none_or(|current| hypotheses[current].score < hypotheses[index].score) {
                best = Some(index);
            }
        }
        best
    }
}

struct ActiveLocation {
    slot: usize,
    existing: Option<usize>,
}

fn trigram_score(
    model: &PerceptronModel,
    w1: &WordData,
    w2: &WordData,
    w3: &WordData,
    scratch: &mut FeatureScratch,
) -> f32 {
    trigram_score_parts(
        model, &w1.lemma, &w1.igs, &w2.lemma, &w2.igs, &w3.lemma, &w3.igs, scratch,
    )
}

fn trigram_score_ambiguity(
    model: &PerceptronModel,
    w1: &AmbiguityWordData,
    w2: &AmbiguityWordData,
    w3: &AmbiguityWordData,
    scratch: &mut FeatureScratch,
) -> f32 {
    trigram_score_parts(
        model, &w1.lemma, &w1.igs, &w2.lemma, &w2.igs, &w3.lemma, &w3.igs, scratch,
    )
}

fn trigram_score_parts(
    model: &PerceptronModel,
    lemma1: &str,
    igs1: &[String],
    lemma2: &str,
    igs2: &[String],
    lemma3: &str,
    igs3: &[String],
    scratch: &mut FeatureScratch,
) -> f32 {
    scratch.reset();
    scratch.add_feature(
        model,
        &FeatureView::F2 {
            lemma1,
            igs2,
            lemma3,
            igs3,
        },
        0x0200_0000,
    );
    scratch.add_feature(
        model,
        &FeatureView::F3 {
            lemma2,
            igs2,
            lemma3,
            igs3,
        },
        0x0300_0000,
    );
    scratch.add_feature(model, &FeatureView::F4 { lemma3, igs3 }, 0x0400_0000);
    scratch.add_feature(model, &FeatureView::F9 { lemma2, lemma3 }, 0x0900_0000);
    scratch.add_feature(model, &FeatureView::F10 { lemma3 }, 0x0a00_0000);
    scratch.add_feature(model, &FeatureView::F10b { lemma2 }, 0x0b00_0000);
    scratch.add_feature(model, &FeatureView::F10c { lemma1 }, 0x0c00_0000);
    let last1 = igs1.last().map_or("", String::as_str);
    let last2 = igs2.last().map_or("", String::as_str);
    for (index, ig) in igs3.iter().enumerate() {
        let representative = igs3[..index]
            .iter()
            .position(|previous| previous == ig)
            .unwrap_or(index);
        let representative = u32::try_from(representative).unwrap_or(u32::MAX) & 0x00ff_ffff;
        scratch.add_feature(
            model,
            &FeatureView::F15 { last1, last2, ig },
            0x1500_0000 | representative,
        );
        scratch.add_feature(
            model,
            &FeatureView::F17 { last2, ig },
            0x1700_0000 | representative,
        );
    }
    for (index, ig) in igs3.iter().enumerate() {
        let identity = u32::try_from(index).unwrap_or(u32::MAX) & 0x00ff_ffff;
        scratch.add_feature(
            model,
            &FeatureView::F20 { index, ig },
            0x2000_0000 | identity,
        );
    }
    scratch.add_feature(model, &FeatureView::F22 { count: igs3.len() }, 0x2200_0000);
    scratch.score()
}

#[derive(Clone, Copy)]
enum FeatureView<'a> {
    F2 {
        lemma1: &'a str,
        igs2: &'a [String],
        lemma3: &'a str,
        igs3: &'a [String],
    },
    F3 {
        lemma2: &'a str,
        igs2: &'a [String],
        lemma3: &'a str,
        igs3: &'a [String],
    },
    F4 {
        lemma3: &'a str,
        igs3: &'a [String],
    },
    F9 {
        lemma2: &'a str,
        lemma3: &'a str,
    },
    F10 {
        lemma3: &'a str,
    },
    F10b {
        lemma2: &'a str,
    },
    F10c {
        lemma1: &'a str,
    },
    F15 {
        last1: &'a str,
        last2: &'a str,
        ig: &'a str,
    },
    F17 {
        last2: &'a str,
        ig: &'a str,
    },
    F20 {
        index: usize,
        ig: &'a str,
    },
    F22 {
        count: usize,
    },
}

impl FeatureView<'_> {
    #[inline(always)]
    fn for_each_utf16(&self, mut consume: impl FnMut(u16)) {
        match *self {
            Self::F2 {
                lemma1,
                igs2,
                lemma3,
                igs3,
            } => {
                emit_ascii(b"2:", &mut consume);
                emit_str(lemma1, &mut consume);
                emit_joined(igs2, &mut consume);
                emit_str(lemma3, &mut consume);
                consume(u16::from(b'+'));
                emit_joined(igs3, &mut consume);
            }
            Self::F3 {
                lemma2,
                igs2,
                lemma3,
                igs3,
            } => {
                emit_ascii(b"3:", &mut consume);
                emit_str(lemma2, &mut consume);
                consume(u16::from(b'+'));
                emit_joined(igs2, &mut consume);
                consume(u16::from(b'-'));
                emit_str(lemma3, &mut consume);
                consume(u16::from(b'+'));
                emit_joined(igs3, &mut consume);
            }
            Self::F4 { lemma3, igs3 } => {
                emit_ascii(b"4:", &mut consume);
                emit_str(lemma3, &mut consume);
                consume(u16::from(b'+'));
                emit_joined(igs3, &mut consume);
            }
            Self::F9 { lemma2, lemma3 } => {
                emit_ascii(b"9:", &mut consume);
                emit_str(lemma2, &mut consume);
                consume(u16::from(b'-'));
                emit_str(lemma3, &mut consume);
            }
            Self::F10 { lemma3 } => {
                emit_ascii(b"10:", &mut consume);
                emit_str(lemma3, &mut consume);
            }
            Self::F10b { lemma2 } => {
                emit_ascii(b"10b:", &mut consume);
                emit_str(lemma2, &mut consume);
            }
            Self::F10c { lemma1 } => {
                emit_ascii(b"10c:", &mut consume);
                emit_str(lemma1, &mut consume);
            }
            Self::F15 { last1, last2, ig } => {
                emit_ascii(b"15:", &mut consume);
                emit_str(last1, &mut consume);
                consume(u16::from(b'-'));
                emit_str(last2, &mut consume);
                consume(u16::from(b'-'));
                emit_str(ig, &mut consume);
            }
            Self::F17 { last2, ig } => {
                emit_ascii(b"17:", &mut consume);
                emit_str(last2, &mut consume);
                emit_str(ig, &mut consume);
            }
            Self::F20 { index, ig } => {
                emit_ascii(b"20:", &mut consume);
                emit_decimal(index, &mut consume);
                consume(u16::from(b'-'));
                emit_str(ig, &mut consume);
            }
            Self::F22 { count } => {
                emit_ascii(b"22:", &mut consume);
                emit_decimal(count, &mut consume);
            }
        }
    }

    #[inline(always)]
    fn initial_and_java_hash(&self) -> (usize, i32) {
        let mut mphf = INITIAL_HASH_SEED;
        let mut java = 0_i32;
        self.for_each_utf16(|unit| {
            mphf = (mphf ^ u32::from(unit)).wrapping_mul(HASH_MULTIPLIER);
            java = java.wrapping_mul(31).wrapping_add(i32::from(unit));
        });
        (usize::try_from(mphf & 0x7fff_ffff).unwrap_or(0), java)
    }

    #[inline(always)]
    fn mphf_hash(&self, seed: u32) -> usize {
        let mut hash = if seed == 0 { INITIAL_HASH_SEED } else { seed };
        self.for_each_utf16(|unit| {
            hash = (hash ^ u32::from(unit)).wrapping_mul(HASH_MULTIPLIER);
        });
        usize::try_from(hash & 0x7fff_ffff).unwrap_or(0)
    }
}

#[inline(always)]
fn emit_ascii(value: &[u8], consume: &mut impl FnMut(u16)) {
    for byte in value {
        consume(u16::from(*byte));
    }
}

#[inline(always)]
fn emit_str(value: &str, consume: &mut impl FnMut(u16)) {
    for_each_utf16(value, consume);
}

#[inline(always)]
fn emit_joined(values: &[String], consume: &mut impl FnMut(u16)) {
    if let Some((first, rest)) = values.split_first() {
        emit_str(first, consume);
        for value in rest {
            consume(u16::from(b'+'));
            emit_str(value, consume);
        }
    }
}

#[inline(always)]
fn emit_decimal(mut value: usize, consume: &mut impl FnMut(u16)) {
    if value == 0 {
        consume(u16::from(b'0'));
        return;
    }
    let mut digits = [0_u8; 20];
    let mut length = 0_usize;
    while value != 0 {
        digits[length] = u8::try_from(value % 10).unwrap_or(0);
        length += 1;
        value /= 10;
    }
    for index in (0..length).rev() {
        consume(u16::from(b'0' + digits[index]));
    }
}

#[derive(Clone, Copy)]
struct FeatureEntry {
    identity: u32,
    count: i16,
    java_hash: i32,
    initial_hash: usize,
    weight: f32,
}

struct FeatureScratch {
    slots: Vec<Option<FeatureEntry>>,
    rehash_order: Vec<FeatureEntry>,
    active_len: usize,
    key_count: usize,
    threshold: usize,
}

impl FeatureScratch {
    fn new() -> Self {
        Self {
            slots: vec![None; 32],
            rehash_order: Vec::with_capacity(24),
            active_len: 32,
            key_count: 0,
            threshold: load_limit(32),
        }
    }

    fn reset(&mut self) {
        self.slots[..self.active_len].fill(None);
        self.key_count = 0;
        self.threshold = load_limit(self.active_len);
    }

    fn add_feature(&mut self, model: &PerceptronModel, key: &FeatureView<'_>, identity: u32) {
        if self.key_count == self.threshold {
            self.expand();
        }
        let (initial_hash, java_hash) = key.initial_and_java_hash();
        let mut slot = hash_slot(java_hash, self.active_len - 1);
        loop {
            match self.slots[slot] {
                None => {
                    let weight = model.get_virtual(key, initial_hash, java_hash);
                    self.slots[slot] = Some(FeatureEntry {
                        identity,
                        count: 1,
                        java_hash,
                        initial_hash,
                        weight,
                    });
                    self.key_count += 1;
                    return;
                }
                Some(existing) => {
                    if existing.identity == identity {
                        debug_assert_eq!(existing.java_hash, java_hash);
                        debug_assert_eq!(existing.initial_hash, initial_hash);
                        let mut updated = existing;
                        updated.count = existing.count.checked_add(1).unwrap_or(i16::MAX);
                        self.slots[slot] = Some(updated);
                        return;
                    }
                    slot = (slot + 1) & (self.active_len - 1);
                }
            }
        }
    }

    fn add_prepared(&mut self, feature: PreparedFeature) {
        if self.key_count == self.threshold {
            self.expand();
        }
        let mut slot = hash_slot(feature.java_hash, self.active_len - 1);
        loop {
            match self.slots[slot] {
                None => {
                    self.slots[slot] = Some(FeatureEntry {
                        identity: feature.identity,
                        count: 1,
                        java_hash: feature.java_hash,
                        initial_hash: 0,
                        weight: feature.weight,
                    });
                    self.key_count += 1;
                    return;
                }
                Some(existing) => {
                    if existing.identity == feature.identity {
                        debug_assert_eq!(existing.java_hash, feature.java_hash);
                        debug_assert_eq!(existing.weight.to_bits(), feature.weight.to_bits());
                        let mut updated = existing;
                        updated.count = existing.count.checked_add(1).unwrap_or(i16::MAX);
                        self.slots[slot] = Some(updated);
                        return;
                    }
                    slot = (slot + 1) & (self.active_len - 1);
                }
            }
        }
    }

    fn expand(&mut self) {
        let old_len = self.active_len;
        let new_len = old_len * 2;
        if self.slots.len() < new_len {
            self.slots.resize(new_len, None);
        }
        self.rehash_order.clear();
        self.rehash_order
            .extend(self.slots[..old_len].iter().flatten().copied());
        self.slots[..new_len].fill(None);
        for entry in self.rehash_order.iter().copied() {
            let mut slot = hash_slot(entry.java_hash, new_len - 1);
            while self.slots[slot].is_some() {
                slot = (slot + 1) & (new_len - 1);
            }
            self.slots[slot] = Some(entry);
        }
        self.active_len = new_len;
        self.threshold = load_limit(new_len);
    }

    fn score(&self) -> f32 {
        let mut score = 0.0_f32;
        for entry in self.slots[..self.active_len].iter().flatten() {
            score = std::ops::Add::add(
                score,
                std::ops::Mul::mul(entry.weight, f32::from(entry.count)),
            );
        }
        score
    }
}

fn java_analysis_hash(analysis: &NativeAnalysis) -> i32 {
    let mut list_hash = 1_i32;
    for morpheme in &analysis.morphemes {
        let morpheme_data_hash = java_string_hash(&morpheme.id)
            .wrapping_mul(31)
            .wrapping_add(java_string_hash(&morpheme.surface));
        list_hash = list_hash.wrapping_mul(31).wrapping_add(morpheme_data_hash);
    }
    java_string_hash(&analysis.dictionary_id)
        .wrapping_mul(31)
        .wrapping_add(list_hash)
        .wrapping_mul(31)
}

fn hypothesis_hash(hypothesis: &Hypothesis, analyses: &[AnalysisData]) -> i32 {
    analyses[hypothesis.previous_analysis]
        .java_hash
        .wrapping_mul(31)
        .wrapping_add(analyses[hypothesis.current_analysis].java_hash)
}

fn hypothesis_equal(left: &Hypothesis, right: &Hypothesis, analyses: &[AnalysisData]) -> bool {
    analyses[left.previous_analysis].analysis == analyses[right.previous_analysis].analysis
        && analyses[left.current_analysis].analysis == analyses[right.current_analysis].analysis
}

#[inline(always)]
fn for_each_utf16(value: &str, mut consume: impl FnMut(u16)) {
    let bytes = value.as_bytes();
    let mut index = 0_usize;
    while index < bytes.len() {
        let first = bytes[index];
        if first < 0x80 {
            consume(u16::from(first));
            index += 1;
        } else if first < 0xe0 {
            let codepoint = (u32::from(first & 0x1f) << 6) | u32::from(bytes[index + 1] & 0x3f);
            consume(u16::try_from(codepoint).unwrap_or(0));
            index += 2;
        } else if first < 0xf0 {
            let codepoint = (u32::from(first & 0x0f) << 12)
                | (u32::from(bytes[index + 1] & 0x3f) << 6)
                | u32::from(bytes[index + 2] & 0x3f);
            consume(u16::try_from(codepoint).unwrap_or(0));
            index += 3;
        } else {
            let codepoint = (u32::from(first & 0x07) << 18)
                | (u32::from(bytes[index + 1] & 0x3f) << 12)
                | (u32::from(bytes[index + 2] & 0x3f) << 6)
                | u32::from(bytes[index + 3] & 0x3f);
            let supplementary = codepoint - 0x1_0000;
            consume(0xd800 | u16::try_from(supplementary >> 10).unwrap_or(0));
            consume(0xdc00 | u16::try_from(supplementary & 0x03ff).unwrap_or(0));
            index += 4;
        }
    }
}

#[inline(always)]
fn initial_mphf_and_java_hash(key: &str) -> (usize, i32) {
    let mut mphf = INITIAL_HASH_SEED;
    let mut java = 0_i32;
    for_each_utf16(key, |unit| {
        mphf = (mphf ^ u32::from(unit)).wrapping_mul(HASH_MULTIPLIER);
        java = java.wrapping_mul(31).wrapping_add(i32::from(unit));
    });
    (usize::try_from(mphf & 0x7fff_ffff).unwrap_or(0), java)
}

#[inline(always)]
fn mphf_hash(key: &str, seed: u32) -> usize {
    let mut hash = if seed == 0 { INITIAL_HASH_SEED } else { seed };
    for_each_utf16(key, |unit| {
        hash = (hash ^ u32::from(unit)).wrapping_mul(HASH_MULTIPLIER);
    });
    usize::try_from(hash & 0x7fff_ffff).unwrap_or(0)
}

#[inline(always)]
fn java_string_hash(value: &str) -> i32 {
    let mut hash = 0_i32;
    for_each_utf16(value, |unit| {
        hash = hash.wrapping_mul(31).wrapping_add(i32::from(unit));
    });
    hash
}

#[cfg(test)]
fn java_fingerprint(value: &str) -> i32 {
    java_string_hash(value) & 0x07ff_ffff
}

const fn rehash(value: i32) -> u32 {
    let hash = value.wrapping_mul(0x9e37_79b9_u32.cast_signed());
    (hash ^ (hash >> 16)).cast_unsigned() & 0x7fff_ffff
}

const fn load_limit(capacity: usize) -> usize {
    capacity.saturating_mul(55) / 100
}

fn hash_slot(value: i32, modulo: usize) -> usize {
    usize::try_from(rehash(value)).map_or(0, |hash| hash & modulo)
}

fn secondary_pos_name(short: &str) -> &'static str {
    match short {
        "Unk" => "UnknownSec",
        "Demons" => "DemonstrativePron",
        "Time" => "Time",
        "Quant" => "QuantitivePron",
        "Ques" => "QuestionPron",
        "Prop" => "ProperNoun",
        "Pers" => "PersonalPron",
        "Reflex" => "ReflexivePron",
        "Ord" => "Ordinal",
        "Card" => "Cardinal",
        "Percent" => "Percentage",
        "Ratio" => "Ratio",
        "Range" => "Range",
        "Real" => "Real",
        "Dist" => "Distribution",
        "Clock" => "Clock",
        "Date" => "Date",
        "Email" => "Email",
        "Url" => "Url",
        "Mention" => "Mention",
        "HashTag" => "HashTag",
        "Emoticon" => "Emoticon",
        "RomanNumeral" => "RomanNumeral",
        "RegAbbrv" => "RegularAbbreviation",
        "Abbrv" => "Abbreviation",
        "PCDat" => "PCDat",
        "PCAcc" => "PCAcc",
        "PCIns" => "PCIns",
        "PCNom" => "PCNom",
        "PCGen" => "PCGen",
        "PCAbl" => "PCAbl",
        _ => "",
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn read_u32(&mut self) -> Result<u32, DisambiguationError> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i32(&mut self) -> Result<i32, DisambiguationError> {
        Ok(self.read_u32()?.cast_signed())
    }

    fn read_usize(&mut self, label: &str) -> Result<usize, DisambiguationError> {
        let value = self.read_i32()?;
        usize::try_from(value).map_err(|_| failure(format!("{label} is negative")))
    }

    fn read_bytes(&mut self, length: usize) -> Result<&'a [u8], DisambiguationError> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| failure("ambiguity model offset overflow"))?;
        let result = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| failure("ambiguity model is truncated"))?;
        self.position = end;
        Ok(result)
    }

    const fn is_finished(&self) -> bool {
        self.position == self.bytes.len()
    }
}

fn failure(message: impl Into<String>) -> DisambiguationError {
    DisambiguationError {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ambiguity_word_data, java_fingerprint, java_string_hash, mphf_hash, pack_trigram_pair_key,
        pack_trigram_score_key, rehash, TrigramScoreCache, TRIGRAM_END_SIGNATURE,
        TRIGRAM_SIGNATURE_MASK,
    };
    use crate::{NativeAnalysis, NativeMorpheme};

    #[test]
    fn packed_trigram_keys_preserve_order_and_boundary_identity() {
        let forward = pack_trigram_score_key([1, 2, 3]).expect("small signatures must pack");
        let reverse = pack_trigram_score_key([3, 2, 1]).expect("small signatures must pack");
        let begin = pack_trigram_score_key([u32::MAX, 2, 3])
            .expect("the reserved begin signature must pack");
        let end = pack_trigram_score_key([u32::MAX - 1, 2, 3])
            .expect("the reserved end signature must pack");
        assert_ne!(forward, reverse);
        assert_ne!(forward, begin);
        assert_ne!(begin, end);
        assert_eq!(begin & TRIGRAM_SIGNATURE_MASK, TRIGRAM_SIGNATURE_MASK);
        assert_eq!(end & TRIGRAM_SIGNATURE_MASK, TRIGRAM_END_SIGNATURE);
        assert!(pack_trigram_score_key([TRIGRAM_END_SIGNATURE as u32, 2, 3]).is_none());
    }

    #[test]
    fn trigram_score_cache_reuses_exact_bits_and_clears() {
        let key = pack_trigram_score_key([17, 23, 42]).expect("test key must pack");
        let row = pack_trigram_pair_key([17, 23]).expect("test pair must pack");
        let next = pack_trigram_pair_key([23, 42]).expect("next test pair must pack");
        let score = f32::from_bits(0xc248_0001);
        let mut cache = TrigramScoreCache::default();
        let row = cache.ensure_row(row).expect("test row must be allocated");
        let next = cache
            .ensure_row(next)
            .expect("next test row must be allocated");
        assert!(cache.get(row, 42, key).is_none());
        cache.insert(row, 42, key, score, next);
        assert_eq!(
            cache
                .get(row, 42, key)
                .map(|(value, next_row)| (value.to_bits(), next_row)),
            Some((score.to_bits(), next)),
        );
        assert_eq!(cache.entries, 1);
        cache = TrigramScoreCache::default();
        assert_eq!(cache.entries, 0);
        assert!(cache.pair_keys.is_empty());
    }

    #[test]
    fn java_hashes_match_known_values() {
        assert_eq!(java_string_hash("abc"), 96_354);
        assert_eq!(java_fingerprint("abc"), 96_354);
        assert_eq!(rehash(96_354), 201_612_502);
        assert_eq!(mphf_hash("abc", 0), 440_920_331);
    }

    #[test]
    fn custom_utf16_hashing_matches_standard_iteration() {
        fn reference_mphf(value: &str, seed: u32) -> usize {
            let mut hash = if seed == 0 {
                super::INITIAL_HASH_SEED
            } else {
                seed
            };
            for unit in value.encode_utf16() {
                hash = (hash ^ u32::from(unit)).wrapping_mul(super::HASH_MULTIPLIER);
            }
            usize::try_from(hash & 0x7fff_ffff).unwrap_or(0)
        }
        fn reference_java(value: &str) -> i32 {
            value.encode_utf16().fold(0_i32, |hash, unit| {
                hash.wrapping_mul(31).wrapping_add(i32::from(unit))
            })
        }
        for value in [
            "",
            "ascii-feature-20:3=A3sg",
            "İstanbul'da",
            "çığöşüı",
            "e\u{301}",
            "🙂",
            "𐐷Türkçe🙂",
        ] {
            assert_eq!(super::mphf_hash(value, 0), reference_mphf(value, 0));
            assert_eq!(super::mphf_hash(value, 47), reference_mphf(value, 47));
            assert_eq!(super::java_string_hash(value), reference_java(value));
            let (initial, java) = super::initial_mphf_and_java_hash(value);
            assert_eq!(initial, reference_mphf(value, 0));
            assert_eq!(java, reference_java(value));
        }
    }

    #[test]
    fn virtual_feature_keys_match_materialized_strings() {
        fn materialize(view: &super::FeatureView<'_>) -> String {
            let mut units = Vec::new();
            view.for_each_utf16(|unit| units.push(unit));
            String::from_utf16(&units).expect("virtual feature must be valid UTF-16")
        }
        let igs1 = vec!["Noun+A3sg".to_owned()];
        let igs2 = vec!["Noun+A3sg+Pnon+Nom".to_owned(), "Verb+Past".to_owned()];
        let igs3 = vec!["Adj".to_owned(), "Verb+A1sg".to_owned()];
        let cases = [
            (
                super::FeatureView::F2 {
                    lemma1: "önce",
                    igs2: &igs2,
                    lemma3: "gel",
                    igs3: &igs3,
                },
                "2:önceNoun+A3sg+Pnon+Nom+Verb+Pastgel+Adj+Verb+A1sg".to_owned(),
            ),
            (
                super::FeatureView::F3 {
                    lemma2: "ara",
                    igs2: &igs2,
                    lemma3: "gel",
                    igs3: &igs3,
                },
                "3:ara+Noun+A3sg+Pnon+Nom+Verb+Past-gel+Adj+Verb+A1sg".to_owned(),
            ),
            (
                super::FeatureView::F4 {
                    lemma3: "gel",
                    igs3: &igs3,
                },
                "4:gel+Adj+Verb+A1sg".to_owned(),
            ),
            (
                super::FeatureView::F9 {
                    lemma2: "ara",
                    lemma3: "gel",
                },
                "9:ara-gel".to_owned(),
            ),
            (
                super::FeatureView::F10 { lemma3: "gel" },
                "10:gel".to_owned(),
            ),
            (
                super::FeatureView::F10b { lemma2: "ara" },
                "10b:ara".to_owned(),
            ),
            (
                super::FeatureView::F10c { lemma1: "önce" },
                "10c:önce".to_owned(),
            ),
            (
                super::FeatureView::F15 {
                    last1: igs1.last().unwrap(),
                    last2: igs2.last().unwrap(),
                    ig: &igs3[1],
                },
                "15:Noun+A3sg-Verb+Past-Verb+A1sg".to_owned(),
            ),
            (
                super::FeatureView::F17 {
                    last2: igs2.last().unwrap(),
                    ig: &igs3[1],
                },
                "17:Verb+PastVerb+A1sg".to_owned(),
            ),
            (
                super::FeatureView::F20 {
                    index: 12,
                    ig: &igs3[1],
                },
                "20:12-Verb+A1sg".to_owned(),
            ),
            (super::FeatureView::F22 { count: 2 }, "22:2".to_owned()),
        ];
        for (view, expected) in cases {
            assert_eq!(materialize(&view), expected);
            let (initial, java) = view.initial_and_java_hash();
            assert_eq!(initial, super::mphf_hash(&expected, 0));
            assert_eq!(java, super::java_string_hash(&expected));
            assert_eq!(view.mphf_hash(61), super::mphf_hash(&expected, 61));
        }
    }

    #[test]
    fn informal_morphemes_use_formal_ids_for_perceptron_features() {
        let analysis = NativeAnalysis {
            canonical: "gitmek_Verb\u{1}Verb=gid\u{2}Fut_Informal=ice\u{2}A1sg=m\u{2}".to_owned(),
            dictionary_id: "gitmek_Verb".to_owned(),
            lemma: "gitmek".to_owned(),
            primary_pos: "Verb".to_owned(),
            secondary_pos: "None".to_owned(),
            surface_form: "gidicem".to_owned(),
            stem: "gid".to_owned(),
            ending: "icem".to_owned(),
            morphemes: vec![
                NativeMorpheme {
                    id: "Verb".to_owned(),
                    name: "Verb".to_owned(),
                    surface: "gid".to_owned(),
                    derivational: false,
                    informal: false,
                    pos: Some("Verb".to_owned()),
                    mapped_id: None,
                },
                NativeMorpheme {
                    id: "Fut_Informal".to_owned(),
                    name: "Fut_Informal".to_owned(),
                    surface: "ice".to_owned(),
                    derivational: false,
                    informal: true,
                    pos: None,
                    mapped_id: Some("Fut".to_owned()),
                },
                NativeMorpheme {
                    id: "A1sg".to_owned(),
                    name: "FirstPersonSingular".to_owned(),
                    surface: "m".to_owned(),
                    derivational: false,
                    informal: false,
                    pos: None,
                    mapped_id: None,
                },
            ],
        };

        let data = ambiguity_word_data(&analysis);
        assert_eq!(data.igs, vec!["VerbFutA1sg"]);
    }
    #[test]
    fn compact_score_cache_preserves_capacity_policy_after_clear() {
        let mut standard = super::DisambiguationScoreCache::default();
        let mut compact = super::DisambiguationScoreCache::compact();
        assert_eq!(standard.trigrams.ensure_row(1), Some(0));
        assert_eq!(compact.trigrams.ensure_row(1), Some(0));
        assert_eq!(standard.stats().3, super::TRIGRAM_PAIR_SLOTS);
        assert_eq!(compact.stats().3, super::COMPACT_TRIGRAM_PAIR_SLOTS);
        assert!(compact.approximate_bytes() < standard.approximate_bytes());

        compact.clear();
        assert_eq!(compact.stats(), (0, 0, 0, 0));
        assert_eq!(compact.trigrams.ensure_row(1), Some(0));
        assert_eq!(compact.stats().3, super::COMPACT_TRIGRAM_PAIR_SLOTS);
    }

    #[test]
    fn causal_decoder_skips_scoring_for_single_candidate_tokens(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let model = super::PerceptronModel::parse(include_bytes!(
            "../../../assets/native/model-compressed"
        ))?;
        let decoder = super::NativeDisambiguator::new(model);
        let words = [
            vec![super::AmbiguityWordData {
                canonical: "a".to_owned(),
                lemma: "a".to_owned(),
                igs: vec!["NounA3sg".to_owned()],
                java_hash: 11,
            }],
            vec![super::AmbiguityWordData {
                canonical: "b".to_owned(),
                lemma: "b".to_owned(),
                igs: vec!["VerbA3sg".to_owned()],
                java_hash: 22,
            }],
            vec![super::AmbiguityWordData {
                canonical: "c".to_owned(),
                lemma: "c".to_owned(),
                igs: vec!["Adj".to_owned()],
                java_hash: 33,
            }],
        ];
        let codes = [
            vec![super::AmbiguityScoringCode::new(1, 1, 11)],
            vec![super::AmbiguityScoringCode::new(2, 2, 22)],
            vec![super::AmbiguityScoringCode::new(3, 3, 33)],
        ];
        let word_refs = words.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let code_refs = codes.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let mut cache = super::DisambiguationScoreCache::default();
        let selected =
            decoder.disambiguate_indices_scored_causal(&word_refs, &code_refs, &mut cache)?;
        assert_eq!(selected, vec![0, 0, 0]);
        assert_eq!(cache.stats(), (0, 0, 0, 0));
        Ok(())
    }

    #[test]
    fn causal_decoder_choice_is_independent_of_future_tokens(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let morphology = super::NativeMorphology::parse(include_bytes!(
            "../../../assets/native/nedo-morph-v1.bin"
        ))?;
        let model = super::PerceptronModel::parse(include_bytes!(
            "../../../assets/native/model-compressed"
        ))?;
        let decoder = super::NativeDisambiguator::new(model);

        let make_words = |tokens: &[&str]| -> Result<
            Vec<Vec<super::AmbiguityWordData>>,
            Box<dyn std::error::Error>,
        > {
            tokens
                .iter()
                .map(|token| {
                    Ok(morphology
                        .analyze_token(token)?
                        .iter()
                        .map(super::ambiguity_word_data)
                        .collect::<Vec<_>>())
                })
                .collect()
        };
        let left = make_words(&["bir", "koyun", "geldi"])?;
        let right = make_words(&["bir", "koyun", "yazar"])?;
        assert!(left[1].len() > 1, "koyun must exercise real ambiguity");
        assert_eq!(left[0], right[0]);
        assert_eq!(left[1], right[1]);
        let left_refs = left.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let right_refs = right.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let left_selected = decoder.disambiguate_indices_causal(&left_refs)?;
        let right_selected = decoder.disambiguate_indices_causal(&right_refs)?;
        assert_eq!(&left_selected[..2], &right_selected[..2]);
        Ok(())
    }

    #[test]
    fn causal_lazy_rows_can_be_completed_by_viterbi_exactly(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let morphology = super::NativeMorphology::parse(include_bytes!(
            "../../../assets/native/nedo-morph-v1.bin"
        ))?;
        let model = super::PerceptronModel::parse(include_bytes!(
            "../../../assets/native/model-compressed"
        ))?;
        let decoder = super::NativeDisambiguator::new(model);
        let tokens = ["bir", "koyun", "evleri", "geldi"];
        let words = tokens
            .iter()
            .map(|token| {
                Ok::<_, Box<dyn std::error::Error>>(
                    morphology
                        .analyze_token(token)?
                        .iter()
                        .map(super::ambiguity_word_data)
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut next_signature = 1_u32;
        let codes = words
            .iter()
            .map(|candidates| {
                candidates
                    .iter()
                    .map(|word| {
                        let signature = next_signature;
                        next_signature = next_signature.saturating_add(1);
                        super::AmbiguityScoringCode::new(signature, signature, word.java_hash)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let word_refs = words.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let code_refs = codes.iter().map(Vec::as_slice).collect::<Vec<_>>();

        let mut mixed_cache = super::DisambiguationScoreCache::default();
        let _ =
            decoder.disambiguate_indices_scored_causal(&word_refs, &code_refs, &mut mixed_cache)?;
        let mixed =
            decoder.disambiguate_indices_scored(&word_refs, &code_refs, &mut mixed_cache)?;

        let mut fresh_cache = super::DisambiguationScoreCache::default();
        let fresh =
            decoder.disambiguate_indices_scored(&word_refs, &code_refs, &mut fresh_cache)?;
        assert_eq!(mixed, fresh);
        Ok(())
    }

    #[test]
    fn dense_decoder_matches_hash_decoder_on_real_candidate_lattices(
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::collections::HashMap;

        let morphology = super::NativeMorphology::parse(include_bytes!(
            "../../../assets/native/nedo-morph-v1.bin"
        ))?;
        let model = super::PerceptronModel::parse(include_bytes!(
            "../../../assets/native/model-compressed"
        ))?;
        let begin = super::boundary_word_data("<s>");
        let end = super::boundary_word_data("</s>");
        let sentences: &[&[&str]] = &[
            &["koyun", "evleri"],
            &["yazar", "kitabı", "okur"],
            &["ben", "yarın", "gelirim"],
            &["dolar", "yükseldi"],
            &["yüz", "koyun", "gördüm"],
            &["Ankara'ya", "gidiyorum"],
            &["geliyom", "gidiyom"],
        ];
        let mut dense_successes = 0_usize;

        for sentence in sentences {
            let analyses = sentence
                .iter()
                .map(|token| morphology.analyze_token(token))
                .collect::<Result<Vec<_>, _>>()?;
            let words = analyses
                .iter()
                .map(|candidates| {
                    candidates
                        .iter()
                        .map(super::ambiguity_word_data)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mut canonical_ids = HashMap::<String, u32>::new();
            let mut signature_ids = HashMap::<(String, Vec<String>), u32>::new();
            let mut codes = Vec::with_capacity(words.len());
            for candidates in &words {
                let mut token_codes = Vec::with_capacity(candidates.len());
                for word in candidates {
                    let next_canonical = u32::try_from(canonical_ids.len())?;
                    let canonical = *canonical_ids
                        .entry(word.canonical.clone())
                        .or_insert(next_canonical);
                    let signature_key = (word.lemma.clone(), word.igs.clone());
                    let next_signature = u32::try_from(signature_ids.len())?;
                    let signature = *signature_ids.entry(signature_key).or_insert(next_signature);
                    token_codes.push(super::AmbiguityScoringCode::new(
                        signature,
                        canonical,
                        word.java_hash,
                    ));
                }
                codes.push(token_codes);
            }
            let word_refs = words.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let code_refs = codes.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let decoder =
                super::CompactDecoder::new(&model, &begin, &end, &word_refs, Some(&code_refs))?;

            let mut hash_runtime = super::DisambiguationScoreCache::default();
            let mut hash_cache = Some(&mut hash_runtime);
            let mut hash_workspace = super::CompactDecodeWorkspace::default();
            let expected =
                decoder.decode_hash_with_workspace(&mut hash_cache, &mut hash_workspace)?;

            let mut dense_runtime = super::DisambiguationScoreCache::default();
            let mut dense_cache = Some(&mut dense_runtime);
            let mut dense_workspace = super::CompactDecodeWorkspace::default();
            if let super::DenseDecodeOutcome::Selected(actual) =
                decoder.decode_dense_with_workspace(&mut dense_cache, &mut dense_workspace)?
            {
                assert_eq!(actual, expected, "dense/hash mismatch for {sentence:?}");
                dense_successes += 1;
            }
        }
        assert!(
            dense_successes > 0,
            "real lattices did not exercise dense success"
        );
        Ok(())
    }
}
