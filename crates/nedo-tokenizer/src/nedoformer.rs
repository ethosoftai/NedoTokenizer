//! `NedoFormer`-specific segmentation lattice and sampling contract.
//!
//! The released NedoTokenizer surface tokenizer remains a standalone path.  This module
//! exposes the richer byte-exact segmentation representation required by
//! `NedoFormer`: one selected segmentation plus every surface-valid alternative,
//! with contextual perceptron scores that can drive deterministic training-time
//! segmentation sampling.

use std::cmp::Ordering;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
#[cfg(feature = "compiled-surface-table")]
use std::sync::Arc;
use std::thread;

use sha2::{Digest, Sha256};

use super::{
    analysis_cache_entries_for_parallelism, assign_inner_groups, auto_code_spans,
    build_batch_ranges, chunk_span, explicit_code_spans, is_sentence_boundary, split_units,
    unit_str, unknown_chunk_analysis, AlignedMorpheme, AnalysisCache, AnalysisMetadata, ByteSpan,
    ContextCachePolicy, FlatAnalysisCache, LexicalKind, NedoFormerSidecarCandidate,
    NedoFormerSidecarUnit, TokenMode, TokenStatus, TokenizedDocument, TokenizedUnit, Tokenizer,
    TokenizerError, TokenizerMode, BATCH_ANALYSIS_CACHE_ENTRIES, MODEL_SHA256, MORPHOLOGY_SHA256,
    NEDOFORMER_LATTICE_SCHEMA_VERSION,
};

const LATTICE_MAGIC: &[u8; 8] = b"NDFLAT01";
const LATTICE_CODEC_SCHEMA: u32 = NEDOFORMER_LATTICE_SCHEMA_VERSION;
const NONE_GROUP_ID: u32 = u32::MAX;

#[derive(Clone, Debug)]
struct CompactSidecarClass {
    cuts: Vec<u64>,
    unknown: bool,
    conditional_log_score: f32,
    selected: bool,
}

#[derive(Clone, Debug)]
struct CompactSidecarWorkUnit {
    selected_unit: TokenizedUnit,
    candidates: Vec<CompactSidecarClass>,
}

/// Persistent worker-local runtime for bounded fast-best preprocessing batches.
///
/// Morphology and contextual-score caches survive between calls. The runtime is
/// bound to the tokenizer configuration and exact `NedoFormer` compiled-table
/// digest that created it; cross-tokenizer reuse is rejected.
pub struct NedoFormerBestRuntime {
    config: super::TokenizerConfig,
    compiled_table_digest: Option<[u8; 32]>,
    threads: usize,
    caches: Vec<FlatAnalysisCache>,
}

impl NedoFormerBestRuntime {
    /// Number of persistent worker caches.
    #[must_use]
    pub const fn threads(&self) -> usize {
        self.threads
    }
}

#[cfg(feature = "compiled-surface-table")]
enum NedoFormerFlatAnalysisSource<'a> {
    Compiled(&'a super::FlatAnalysisSet),
    Live(Arc<super::FlatAnalysisSet>),
}

#[cfg(feature = "compiled-surface-table")]
impl NedoFormerFlatAnalysisSource<'_> {
    fn set(&self) -> &super::FlatAnalysisSet {
        match self {
            Self::Compiled(set) => set,
            Self::Live(set) => set.as_ref(),
        }
    }
}

/// One distinct segmentation hypothesis for a surface unit.
#[derive(Clone, Debug, PartialEq)]
pub struct NedoFormerSegmentationCandidate {
    /// Strictly increasing cuts in original document byte offsets.
    pub cuts: Vec<u64>,
    /// Resulting unit status if this hypothesis is selected.
    pub status: TokenStatus,
    /// Representative analysis metadata for this cut class, when morphological.
    pub analysis: Option<AnalysisMetadata>,
    /// Number of native analyses collapsed into this identical output class.
    pub analysis_count: usize,
    /// Conditional perceptron log-score, aggregated by log-sum-exp within the class.
    ///
    /// This is an exact model score under fixed neighboring Viterbi choices, not a
    /// calibrated probability.
    pub conditional_log_score: f32,
    /// Whether the deterministic Viterbi path selects this output class.
    pub selected: bool,
}

/// One byte-exact unit and all distinct segmentation hypotheses for it.
#[derive(Clone, Debug, PartialEq)]
pub struct NedoFormerLatticeUnit {
    /// Selected/reference unit metadata, including byte span and phonological group.
    pub selected_unit: TokenizedUnit,
    /// Distinct surface segmentation candidates.
    pub candidates: Vec<NedoFormerSegmentationCandidate>,
}

/// Complete byte-exact `NedoFormer` segmentation lattice.
#[derive(Clone, Debug, PartialEq)]
pub struct NedoFormerLatticeDocument {
    raw: Vec<u8>,
    units: Vec<NedoFormerLatticeUnit>,
}

/// Policy used when materializing one segmentation from a lattice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NedoFormerSamplingPolicy {
    /// Use the deterministic contextual Viterbi path.
    Best,
    /// Sample uniformly over distinct cut classes.
    Uniform,
    /// Temperature-softmax the exact conditional perceptron log-scores.
    ContextWeighted {
        /// Positive softmax temperature.  Lower values approach the local argmax.
        temperature: f32,
    },
}

impl NedoFormerLatticeDocument {
    /// Original bytes.  They are the only text source of truth.
    #[must_use]
    pub fn raw(&self) -> &[u8] {
        &self.raw
    }

    /// Lattice units in exact document order.
    #[must_use]
    pub fn units(&self) -> &[NedoFormerLatticeUnit] {
        &self.units
    }

    /// Stable checksum-protected lattice serialization.
    ///
    /// The payload retains original bytes, unit/mode/group metadata, every distinct
    /// cut class, representative rich analysis, conditional f32 score bits, and the
    /// deterministic selected class.  Sampling therefore has identical semantics
    /// after reload.
    ///
    /// # Errors
    ///
    /// Returns an error if a collection or string length cannot fit the codec.
    pub fn to_bytes(&self) -> Result<Vec<u8>, TokenizerError> {
        let _ = self.selected_document()?;
        let mut payload = Vec::new();
        write_u64(
            &mut payload,
            u64::try_from(self.raw.len())
                .map_err(|_| TokenizerError::LengthOverflow("NedoFormer lattice raw length"))?,
        );
        payload.extend_from_slice(&self.raw);
        write_u32(
            &mut payload,
            u32::try_from(self.units.len())
                .map_err(|_| TokenizerError::LengthOverflow("NedoFormer lattice unit count"))?,
        );
        for lattice in &self.units {
            let unit = &lattice.selected_unit;
            write_u64(&mut payload, unit.span.start);
            write_u64(&mut payload, unit.span.end);
            payload.push(unit.kind as u8);
            payload.push(unit.mode as u8);
            write_u32(&mut payload, unit.group_id.unwrap_or(NONE_GROUP_ID));
            write_u32(
                &mut payload,
                u32::try_from(lattice.candidates.len()).map_err(|_| {
                    TokenizerError::LengthOverflow("NedoFormer lattice candidate count")
                })?,
            );
            for candidate in &lattice.candidates {
                payload.push(candidate.status as u8);
                payload.push(u8::from(candidate.selected));
                payload.extend_from_slice(&[0, 0]);
                write_u32(
                    &mut payload,
                    u32::try_from(candidate.analysis_count).map_err(|_| {
                        TokenizerError::LengthOverflow("NedoFormer lattice analysis count")
                    })?,
                );
                payload.extend_from_slice(&candidate.conditional_log_score.to_bits().to_le_bytes());
                write_u32(
                    &mut payload,
                    u32::try_from(candidate.cuts.len()).map_err(|_| {
                        TokenizerError::LengthOverflow("NedoFormer lattice cut count")
                    })?,
                );
                for cut in &candidate.cuts {
                    write_u64(&mut payload, *cut);
                }
                payload.push(u8::from(candidate.analysis.is_some()));
                if let Some(analysis) = &candidate.analysis {
                    write_analysis(&mut payload, analysis)?;
                }
            }
        }

        let mut output = Vec::with_capacity(8 + 4 + 64 + 64 + 32 + payload.len());
        output.extend_from_slice(LATTICE_MAGIC);
        output.extend_from_slice(&LATTICE_CODEC_SCHEMA.to_le_bytes());
        output.extend_from_slice(MORPHOLOGY_SHA256.as_bytes());
        output.extend_from_slice(MODEL_SHA256.as_bytes());
        output.extend_from_slice(&Sha256::digest(&payload));
        output.extend_from_slice(&payload);
        Ok(output)
    }

    /// Loads and fully validates a stable `NedoFormer` segmentation lattice.
    ///
    /// # Errors
    ///
    /// Returns an error for identity/checksum/count/UTF-8/span/candidate failures.
    #[allow(clippy::too_many_lines)] // Codec validation is intentionally linear and explicit.
    pub fn from_bytes(input: &[u8]) -> Result<Self, TokenizerError> {
        const HEADER: usize = 8 + 4 + 64 + 64 + 32;
        if input.len() < HEADER {
            return Err(TokenizerError::TruncatedCodec);
        }
        if input.get(..8) != Some(LATTICE_MAGIC) {
            return Err(TokenizerError::BadCodecMagic);
        }
        let mut reader = LatticeReader::new(&input[8..]);
        let schema = reader.u32()?;
        if schema != LATTICE_CODEC_SCHEMA {
            return Err(TokenizerError::UnsupportedCodecVersion(schema));
        }
        let morphology = reader.bytes(64)?;
        let model = reader.bytes(64)?;
        if morphology != MORPHOLOGY_SHA256.as_bytes() || model != MODEL_SHA256.as_bytes() {
            return Err(TokenizerError::AssetIdentityMismatch);
        }
        let expected: [u8; 32] = reader
            .bytes(32)?
            .try_into()
            .map_err(|_| TokenizerError::TruncatedCodec)?;
        let payload = reader.remaining_bytes();
        let actual: [u8; 32] = Sha256::digest(payload).into();
        if actual != expected {
            return Err(TokenizerError::CodecChecksumMismatch);
        }

        let mut reader = LatticeReader::new(payload);
        let raw_len = reader.usize64("NedoFormer lattice raw length")?;
        let raw = reader.bytes(raw_len)?.to_vec();
        let unit_count = reader.usize32("NedoFormer lattice unit count")?;
        if unit_count > reader.remaining() / 28 {
            return Err(TokenizerError::ImpossibleCodecCount(
                "NedoFormer lattice unit count",
            ));
        }
        let mut units = Vec::with_capacity(unit_count);
        for _ in 0..unit_count {
            let span = ByteSpan {
                start: reader.u64()?,
                end: reader.u64()?,
            };
            let kind = lattice_lexical_kind(reader.u8()?)?;
            let mode = TokenMode::try_from(reader.u8()?)?;
            let group_raw = reader.u32()?;
            let candidate_count = reader.usize32("NedoFormer lattice candidate count")?;
            if candidate_count == 0 || candidate_count > reader.remaining() / 17 {
                return Err(TokenizerError::ImpossibleCodecCount(
                    "NedoFormer lattice candidate count",
                ));
            }
            let mut candidates = Vec::with_capacity(candidate_count);
            for _ in 0..candidate_count {
                let status = TokenStatus::try_from(reader.u8()?)?;
                let selected = reader.boolean("NedoFormer lattice selected")?;
                if reader.bytes(2)? != [0, 0] {
                    return Err(TokenizerError::InvalidTrainingEncoding(
                        "NedoFormer lattice reserved bytes are nonzero",
                    ));
                }
                let analysis_count = reader.usize32("NedoFormer lattice analysis count")?;
                if analysis_count == 0 {
                    return Err(TokenizerError::InvalidTrainingEncoding(
                        "NedoFormer lattice analysis count is zero",
                    ));
                }
                let score = f32::from_bits(reader.u32()?);
                if !score.is_finite() {
                    return Err(TokenizerError::InvalidTrainingEncoding(
                        "NedoFormer lattice conditional score is non-finite",
                    ));
                }
                let cut_count = reader.usize32("NedoFormer lattice cut count")?;
                if cut_count > reader.remaining() / 8 {
                    return Err(TokenizerError::ImpossibleCodecCount(
                        "NedoFormer lattice cut count",
                    ));
                }
                let mut cuts = Vec::with_capacity(cut_count);
                for _ in 0..cut_count {
                    cuts.push(reader.u64()?);
                }
                nedo_format::SurfaceUnit::new(span, cuts.clone())?;
                let analysis = if reader.boolean("NedoFormer lattice has analysis")? {
                    Some(read_analysis(&mut reader)?)
                } else {
                    None
                };
                candidates.push(NedoFormerSegmentationCandidate {
                    cuts,
                    status,
                    analysis,
                    analysis_count,
                    conditional_log_score: score,
                    selected,
                });
            }
            if candidates
                .iter()
                .filter(|candidate| candidate.selected)
                .count()
                != 1
            {
                return Err(TokenizerError::InvalidTrainingEncoding(
                    "NedoFormer lattice codec requires exactly one selected candidate",
                ));
            }
            let selected_candidate = candidates
                .iter()
                .find(|candidate| candidate.selected)
                .ok_or(TokenizerError::InvalidTrainingEncoding(
                    "NedoFormer lattice selected candidate disappeared",
                ))?;
            let selected_unit = TokenizedUnit {
                span,
                kind,
                mode,
                status: selected_candidate.status,
                group_id: (group_raw != NONE_GROUP_ID).then_some(group_raw),
                cuts: selected_candidate.cuts.clone(),
                analysis: selected_candidate.analysis.clone(),
            };
            units.push(NedoFormerLatticeUnit {
                selected_unit,
                candidates,
            });
        }
        if reader.remaining() != 0 {
            return Err(TokenizerError::TrailingCodecBytes(reader.remaining()));
        }
        let lattice = Self { raw, units };
        let _ = lattice.selected_document()?;
        // Every candidate must itself be valid in the surrounding document schema.
        for unit_index in 0..lattice.units.len() {
            for candidate_index in 0..lattice.units[unit_index].candidates.len() {
                let mut units = lattice
                    .units
                    .iter()
                    .map(|entry| entry.selected_unit.clone())
                    .collect::<Vec<_>>();
                let candidate = &lattice.units[unit_index].candidates[candidate_index];
                units[unit_index].cuts.clone_from(&candidate.cuts);
                units[unit_index].status = candidate.status;
                units[unit_index].analysis.clone_from(&candidate.analysis);
                TokenizedDocument::new(lattice.raw.clone(), units)?;
            }
        }
        Ok(lattice)
    }

    /// Materializes the deterministic selected path as a normal tokenized document.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored lattice violates the base tokenized-document
    /// coverage or metadata invariants.
    pub fn selected_document(&self) -> Result<TokenizedDocument, TokenizerError> {
        self.materialize(NedoFormerSamplingPolicy::Best, 0)
    }

    /// Deterministically samples one valid segmentation path.
    ///
    /// `seed` is consumed only for units with more than one distinct output class.
    /// Identical `(lattice, policy, seed)` inputs produce identical cuts.
    ///
    /// # Errors
    ///
    /// Returns an error for non-positive/non-finite temperature, malformed lattice
    /// probabilities, or a materialized document that violates byte-span invariants.
    pub fn sample(
        &self,
        policy: NedoFormerSamplingPolicy,
        seed: u64,
    ) -> Result<TokenizedDocument, TokenizerError> {
        self.materialize(policy, seed)
    }

    fn materialize(
        &self,
        policy: NedoFormerSamplingPolicy,
        seed: u64,
    ) -> Result<TokenizedDocument, TokenizerError> {
        if let NedoFormerSamplingPolicy::ContextWeighted { temperature } = policy {
            if !temperature.is_finite() || temperature <= 0.0 {
                return Err(TokenizerError::InvalidConfiguration(
                    "NedoFormer sampling temperature must be positive and finite",
                ));
            }
        }
        let mut rng = SplitMix64::new(seed);
        let mut units = Vec::with_capacity(self.units.len());
        for lattice in &self.units {
            let candidate_index = choose_candidate(&lattice.candidates, policy, &mut rng)?;
            let candidate = lattice.candidates.get(candidate_index).ok_or(
                TokenizerError::InvalidTrainingEncoding(
                    "NedoFormer lattice candidate index is outside the unit",
                ),
            )?;
            let mut unit = lattice.selected_unit.clone();
            unit.cuts.clone_from(&candidate.cuts);
            unit.status = candidate.status;
            unit.analysis.clone_from(&candidate.analysis);
            units.push(unit);
        }
        TokenizedDocument::new(self.raw.clone(), units)
    }
}

impl Tokenizer<'_> {
    /// Builds the byte-exact `NedoFormer` segmentation lattice for one document.
    ///
    /// The base tokenizer first establishes exact unit/code/fallback boundaries.
    /// Turkish contextual segments are then re-scored with the same pinned native
    /// morphology and perceptron model.  Native analyses that imply identical cuts
    /// collapse into one output class so segmentation sampling is not biased merely
    /// because a cut pattern has multiple morphology labels.
    ///
    /// # Errors
    ///
    /// Returns an error for scanning/morphology/disambiguation/alignment failures or
    /// if the reconstructed best lattice path differs from normal tokenization.
    pub fn nedoformer_lattice(
        &self,
        raw: Vec<u8>,
    ) -> Result<NedoFormerLatticeDocument, TokenizerError> {
        let mut cache = AnalysisCache::new(BATCH_ANALYSIS_CACHE_ENTRIES);
        self.nedoformer_lattice_with_cache(raw, &mut cache)
    }

    /// Builds `NedoFormer` lattices for a byte-weighted document batch while reusing
    /// one morphology-analysis cache per worker.
    ///
    /// Output order is identical to input order. `threads` must be positive.
    ///
    /// # Errors
    ///
    /// Returns any single-document lattice error, invalid thread configuration,
    /// batch length overflow, or worker panic.
    pub fn nedoformer_lattice_batch(
        &self,
        inputs: &[Vec<u8>],
        threads: usize,
    ) -> Result<Vec<NedoFormerLatticeDocument>, TokenizerError> {
        if threads == 0 {
            return Err(TokenizerError::InvalidConfiguration(
                "NedoFormer lattice threads must be positive",
            ));
        }
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let ranges = build_batch_ranges(inputs, threads)?;
        let worker_count = threads.min(ranges.len()).max(1);
        if worker_count == 1 {
            let mut cache = AnalysisCache::new(analysis_cache_entries_for_parallelism(1));
            return inputs
                .iter()
                .cloned()
                .map(|raw| self.nedoformer_lattice_with_cache(raw, &mut cache))
                .collect();
        }

        let next = AtomicUsize::new(0);
        let completed = thread::scope(|scope| {
            let handles = (0..worker_count)
                .map(|_| {
                    let ranges = &ranges;
                    let next = &next;
                    scope.spawn(move || {
                        let mut cache = AnalysisCache::new(analysis_cache_entries_for_parallelism(
                            worker_count,
                        ));
                        let mut chunks = Vec::new();
                        loop {
                            let chunk_index = next.fetch_add(1, AtomicOrdering::Relaxed);
                            let Some(range) = ranges.get(chunk_index) else {
                                break;
                            };
                            let mut rows = Vec::with_capacity(range.len());
                            for index in range.clone() {
                                rows.push(self.nedoformer_lattice_with_cache(
                                    inputs[index].clone(),
                                    &mut cache,
                                )?);
                            }
                            chunks.push((chunk_index, rows));
                        }
                        Ok::<_, TokenizerError>(chunks)
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().map_err(|_| TokenizerError::WorkerPanicked)?)
                .collect::<Result<Vec<_>, _>>()
        })?;
        let mut chunks = completed.into_iter().flatten().collect::<Vec<_>>();
        chunks.sort_unstable_by_key(|entry| entry.0);
        let documents = chunks
            .into_iter()
            .flat_map(|(_, rows)| rows)
            .collect::<Vec<_>>();
        if documents.len() != inputs.len() {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "NedoFormer lattice batch output cardinality differs",
            ));
        }
        Ok(documents)
    }

    /// Creates persistent worker-local caches for repeated fast-best batches.
    ///
    /// # Errors
    ///
    /// Returns an error if `threads` is zero.
    pub fn nedoformer_best_runtime(
        &self,
        threads: usize,
        context_cache_policy: ContextCachePolicy,
    ) -> Result<NedoFormerBestRuntime, TokenizerError> {
        if threads == 0 {
            return Err(TokenizerError::InvalidConfiguration(
                "NedoFormer best runtime threads must be positive",
            ));
        }
        let entries = analysis_cache_entries_for_parallelism(threads);
        let shared_prepared_words = self.new_nedoformer_shared_prepared_word_cache();
        Ok(NedoFormerBestRuntime {
            config: self.config,
            compiled_table_digest: self.nedoformer_compiled_surface_table_digest(),
            threads,
            caches: (0..threads)
                .map(|_| {
                    self.new_nedoformer_flat_analysis_cache(
                        entries,
                        context_cache_policy,
                        shared_prepared_words.clone(),
                    )
                })
                .collect(),
        })
    }

    /// Encodes one bounded fast-best batch while preserving runtime caches.
    ///
    /// Output order matches input order exactly.
    ///
    /// # Errors
    ///
    /// Rejects a runtime created for another tokenizer/table, zero-cardinality
    /// internal scheduling, worker panics, and all normal fast-best errors.
    pub fn nedoformer_best_sidecar_batch_with_runtime(
        &self,
        inputs: &[Vec<u8>],
        runtime: &mut NedoFormerBestRuntime,
    ) -> Result<Vec<Vec<u8>>, TokenizerError> {
        if runtime.config != self.config
            || runtime.compiled_table_digest != self.nedoformer_compiled_surface_table_digest()
        {
            return Err(TokenizerError::InvalidConfiguration(
                "NedoFormer best runtime belongs to a different tokenizer configuration or table",
            ));
        }
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let ranges = build_batch_ranges(inputs, runtime.threads)?;
        let worker_count = runtime.threads.min(ranges.len()).max(1);
        if worker_count == 1 {
            return inputs
                .iter()
                .cloned()
                .map(|raw| {
                    self.nedoformer_best_sidecar_with_flat_cache(raw, &mut runtime.caches[0])
                })
                .collect();
        }

        let next = AtomicUsize::new(0);
        let completed = thread::scope(|scope| {
            let handles = runtime.caches[..worker_count]
                .iter_mut()
                .map(|cache| {
                    let ranges = &ranges;
                    let next = &next;
                    scope.spawn(move || {
                        let mut chunks = Vec::new();
                        loop {
                            let chunk_index = next.fetch_add(1, AtomicOrdering::Relaxed);
                            let Some(range) = ranges.get(chunk_index) else {
                                break;
                            };
                            let mut rows = Vec::with_capacity(range.len());
                            for index in range.clone() {
                                rows.push(self.nedoformer_best_sidecar_with_flat_cache(
                                    inputs[index].clone(),
                                    cache,
                                )?);
                            }
                            chunks.push((chunk_index, rows));
                        }
                        Ok::<_, TokenizerError>(chunks)
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().map_err(|_| TokenizerError::WorkerPanicked)?)
                .collect::<Result<Vec<_>, _>>()
        })?;
        let mut chunks = completed.into_iter().flatten().collect::<Vec<_>>();
        chunks.sort_unstable_by_key(|entry| entry.0);
        let sidecars = chunks
            .into_iter()
            .flat_map(|(_, rows)| rows)
            .collect::<Vec<_>>();
        if sidecars.len() != inputs.len() {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "NedoFormer persistent best batch output cardinality differs",
            ));
        }
        Ok(sidecars)
    }

    /// Builds deterministic best-path-only sidecars without materializing the
    /// full segmentation lattice or conditional candidate scores.
    ///
    /// The resulting sidecar intentionally contains one selected cut class per
    /// unit. Its `Best` sampled lossless document is required to equal the `Best`
    /// sample from [`Self::nedoformer_sidecar_batch`], but it cannot be used for
    /// uniform/context-weighted segmentation sampling because alternatives were
    /// never serialized.
    ///
    /// # Errors
    ///
    /// Returns any scanning, morphology, contextual-selection, sidecar, invalid
    /// thread configuration, or worker error.
    pub fn nedoformer_best_sidecar_batch(
        &self,
        inputs: &[Vec<u8>],
        threads: usize,
    ) -> Result<Vec<Vec<u8>>, TokenizerError> {
        if threads == 0 {
            return Err(TokenizerError::InvalidConfiguration(
                "NedoFormer best-sidecar threads must be positive",
            ));
        }
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let ranges = build_batch_ranges(inputs, threads)?;
        let worker_count = threads.min(ranges.len()).max(1);
        let shared_prepared_words = self.new_nedoformer_shared_prepared_word_cache();
        if worker_count == 1 {
            let mut cache = self.new_nedoformer_flat_analysis_cache(
                analysis_cache_entries_for_parallelism(1),
                ContextCachePolicy::Full,
                shared_prepared_words.clone(),
            );
            return inputs
                .iter()
                .cloned()
                .map(|raw| self.nedoformer_best_sidecar_with_flat_cache(raw, &mut cache))
                .collect();
        }

        let next = AtomicUsize::new(0);
        let completed = thread::scope(|scope| {
            let handles = (0..worker_count)
                .map(|_| {
                    let ranges = &ranges;
                    let next = &next;
                    let shared_prepared_words = shared_prepared_words.clone();
                    scope.spawn(move || {
                        let mut cache = self.new_nedoformer_flat_analysis_cache(
                            analysis_cache_entries_for_parallelism(worker_count),
                            ContextCachePolicy::Full,
                            shared_prepared_words,
                        );
                        let mut chunks = Vec::new();
                        loop {
                            let chunk_index = next.fetch_add(1, AtomicOrdering::Relaxed);
                            let Some(range) = ranges.get(chunk_index) else {
                                break;
                            };
                            let mut rows = Vec::with_capacity(range.len());
                            for index in range.clone() {
                                rows.push(self.nedoformer_best_sidecar_with_flat_cache(
                                    inputs[index].clone(),
                                    &mut cache,
                                )?);
                            }
                            chunks.push((chunk_index, rows));
                        }
                        Ok::<_, TokenizerError>(chunks)
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().map_err(|_| TokenizerError::WorkerPanicked)?)
                .collect::<Result<Vec<_>, _>>()
        })?;
        let mut chunks = completed.into_iter().flatten().collect::<Vec<_>>();
        chunks.sort_unstable_by_key(|entry| entry.0);
        let sidecars = chunks
            .into_iter()
            .flat_map(|(_, rows)| rows)
            .collect::<Vec<_>>();
        if sidecars.len() != inputs.len() {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "NedoFormer best-sidecar batch output cardinality differs",
            ));
        }
        Ok(sidecars)
    }

    /// Builds compact `NedoFormer` sidecars in the same worker that constructs
    /// each lattice, so expensive sidecar serialization does not become a serial
    /// post-pass after parallel morphology/disambiguation.
    ///
    /// Output order is identical to input order. `threads` must be positive.
    ///
    /// # Errors
    ///
    /// Returns any lattice/sidecar error, invalid thread configuration, batch
    /// length overflow, or worker panic.
    #[allow(clippy::too_many_lines)] // Mirrors the fail-closed lattice batch worker contract.
    pub fn nedoformer_sidecar_batch(
        &self,
        inputs: &[Vec<u8>],
        threads: usize,
    ) -> Result<Vec<Vec<u8>>, TokenizerError> {
        if threads == 0 {
            return Err(TokenizerError::InvalidConfiguration(
                "NedoFormer sidecar threads must be positive",
            ));
        }
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let ranges = build_batch_ranges(inputs, threads)?;
        let worker_count = threads.min(ranges.len()).max(1);
        let shared_prepared_words = self.new_nedoformer_shared_prepared_word_cache();
        if worker_count == 1 {
            let mut cache = self.new_nedoformer_flat_analysis_cache(
                analysis_cache_entries_for_parallelism(1),
                ContextCachePolicy::Compact,
                shared_prepared_words.clone(),
            );
            return inputs
                .iter()
                .cloned()
                .map(|raw| self.nedoformer_sidecar_with_flat_cache(raw, &mut cache))
                .collect();
        }

        let next = AtomicUsize::new(0);
        let completed = thread::scope(|scope| {
            let handles = (0..worker_count)
                .map(|_| {
                    let ranges = &ranges;
                    let next = &next;
                    let shared_prepared_words = shared_prepared_words.clone();
                    scope.spawn(move || {
                        let mut cache = self.new_nedoformer_flat_analysis_cache(
                            analysis_cache_entries_for_parallelism(worker_count),
                            ContextCachePolicy::Compact,
                            shared_prepared_words,
                        );
                        let mut chunks = Vec::new();
                        loop {
                            let chunk_index = next.fetch_add(1, AtomicOrdering::Relaxed);
                            let Some(range) = ranges.get(chunk_index) else {
                                break;
                            };
                            let mut rows = Vec::with_capacity(range.len());
                            for index in range.clone() {
                                rows.push(self.nedoformer_sidecar_with_flat_cache(
                                    inputs[index].clone(),
                                    &mut cache,
                                )?);
                            }
                            chunks.push((chunk_index, rows));
                        }
                        Ok::<_, TokenizerError>(chunks)
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().map_err(|_| TokenizerError::WorkerPanicked)?)
                .collect::<Result<Vec<_>, _>>()
        })?;
        let mut chunks = completed.into_iter().flatten().collect::<Vec<_>>();
        chunks.sort_unstable_by_key(|entry| entry.0);
        let sidecars = chunks
            .into_iter()
            .flat_map(|(_, rows)| rows)
            .collect::<Vec<_>>();
        if sidecars.len() != inputs.len() {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "NedoFormer sidecar batch output cardinality differs",
            ));
        }
        Ok(sidecars)
    }

    fn nedoformer_best_sidecar_with_flat_cache(
        &self,
        raw: Vec<u8>,
        cache: &mut FlatAnalysisCache,
    ) -> Result<Vec<u8>, TokenizerError> {
        let scan = nedo_core::scan_compact(raw)?;
        let raw_slice = scan.raw();
        let code_spans = match self.config.mode {
            TokenizerMode::Auto if self.config.detect_unmarked_code => {
                auto_code_spans(raw_slice, scan.code_hints())
            }
            TokenizerMode::Auto => explicit_code_spans(raw_slice),
            TokenizerMode::Turkish | TokenizerMode::Code => Vec::new(),
        };
        let mut units = super::split_compact_units(&scan, &code_spans, self.config.mode)?;
        let mut segment = Vec::new();
        for index in 0..units.len() {
            let unit = &units[index];
            if unit.mode != TokenMode::Turkish
                || matches!(
                    unit.kind,
                    LexicalKind::LineBreak | LexicalKind::Control | LexicalKind::Opaque
                )
            {
                self.flush_best_sidecar_segment(raw_slice, &mut units, &mut segment, cache)?;
                continue;
            }
            if matches!(unit.kind, LexicalKind::Whitespace) {
                continue;
            }
            segment.push(index);
            if is_sentence_boundary(unit.kind, raw_slice, unit.span)?
                || segment.len() >= self.config.max_sentence_tokens
            {
                self.flush_best_sidecar_segment(raw_slice, &mut units, &mut segment, cache)?;
            }
        }
        self.flush_best_sidecar_segment(raw_slice, &mut units, &mut segment, cache)?;
        units = super::split_long_flat_units(raw_slice, units, self.config.max_fallback_chars)?;
        assign_inner_groups(raw_slice, &mut units)?;
        let sidecar_units = units
            .into_iter()
            .map(|unit| NedoFormerSidecarUnit {
                span: unit.span,
                kind: unit.kind,
                mode: unit.mode,
                group_id: unit.group_id,
                candidates: vec![NedoFormerSidecarCandidate {
                    cuts: unit.cuts,
                    conditional_log_score: 0.0,
                    selected: true,
                }],
            })
            .collect::<Vec<_>>();
        let raw = scan.into_raw();
        super::nedoformer_sidecar::encode_sidecar_units(&raw, &sidecar_units)
    }

    fn flush_best_sidecar_segment(
        &self,
        raw: &[u8],
        units: &mut [TokenizedUnit],
        indices: &mut Vec<usize>,
        cache: &mut FlatAnalysisCache,
    ) -> Result<(), TokenizerError> {
        if indices.is_empty() {
            return Ok(());
        }
        let tokens = indices
            .iter()
            .map(|&index| unit_str(raw, units[index].span))
            .collect::<Result<Vec<_>, _>>()?;
        #[cfg(feature = "compiled-surface-table")]
        let sources = {
            let mut values = Vec::with_capacity(tokens.len());
            for token in &tokens {
                if let Some(set) = self
                    .nedoformer_compiled_surface_analysis_table
                    .as_ref()
                    .and_then(|table| table.get(token))
                {
                    values.push(NedoFormerFlatAnalysisSource::Compiled(set));
                } else {
                    values.push(NedoFormerFlatAnalysisSource::Live(
                        cache.analyze_nedoformer(&self.morphology, token)?,
                    ));
                }
            }
            values
        };
        #[cfg(feature = "compiled-surface-table")]
        let sets = sources
            .iter()
            .map(NedoFormerFlatAnalysisSource::set)
            .collect::<Vec<_>>();
        #[cfg(not(feature = "compiled-surface-table"))]
        let owned_sets = tokens
            .iter()
            .map(|token| cache.analyze_nedoformer(&self.morphology, token))
            .collect::<Result<Vec<_>, _>>()?;
        #[cfg(not(feature = "compiled-surface-table"))]
        let sets = owned_sets.iter().map(AsRef::as_ref).collect::<Vec<_>>();

        let needs_context =
            self.config.contextual_disambiguation && sets.iter().any(|set| !set.output_invariant);
        let selected = if needs_context {
            let ambiguity = sets
                .iter()
                .map(|set| set.ambiguity.as_ref())
                .collect::<Vec<_>>();
            let scoring_codes = sets
                .iter()
                .map(|set| set.scoring_codes.as_ref())
                .collect::<Vec<_>>();
            self.disambiguator.disambiguate_indices_scored_causal(
                &ambiguity,
                &scoring_codes,
                &mut cache.disambiguation_scores,
            )?
        } else {
            vec![0; sets.len()]
        };
        if selected.len() != indices.len() {
            return Err(TokenizerError::ContextLengthMismatch);
        }
        for ((&unit_index, set), &candidate_index) in indices.iter().zip(&sets).zip(&selected) {
            if candidate_index >= set.relative_cuts.len() || candidate_index >= set.unknown.len() {
                return Err(TokenizerError::InvalidTrainingEncoding(
                    "NedoFormer best candidate index is out of range",
                ));
            }
            let span = units[unit_index].span;
            let mut cuts =
                set.relative_cuts[candidate_index]
                    .iter()
                    .map(|cut| {
                        span.start.checked_add(u64::from(*cut)).ok_or(
                            TokenizerError::LengthOverflow("NedoFormer best sidecar cut"),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            if units[unit_index].kind == LexicalKind::Number {
                cuts.extend(super::numeric_micro_cuts(raw, span)?);
                cuts.sort_unstable();
                cuts.dedup();
            }
            units[unit_index].cuts = cuts;
            units[unit_index].status = if set.unknown[candidate_index] {
                TokenStatus::Unknown
            } else {
                TokenStatus::Morphological
            };
            units[unit_index].analysis = None;
        }
        indices.clear();
        Ok(())
    }

    fn nedoformer_sidecar_with_flat_cache(
        &self,
        raw: Vec<u8>,
        cache: &mut FlatAnalysisCache,
    ) -> Result<Vec<u8>, TokenizerError> {
        let scan = nedo_core::scan(raw)?;
        let raw_slice = scan.document().decode();
        let code_spans = match self.config.mode {
            TokenizerMode::Auto if self.config.detect_unmarked_code => {
                auto_code_spans(raw_slice, scan.code_hints())
            }
            TokenizerMode::Auto => explicit_code_spans(raw_slice),
            TokenizerMode::Turkish | TokenizerMode::Code => Vec::new(),
        };
        let units = split_units(&scan, &code_spans, self.config.mode)?;
        let mut work = units
            .into_iter()
            .map(|unit| CompactSidecarWorkUnit {
                candidates: vec![CompactSidecarClass {
                    cuts: unit.cuts.clone(),
                    unknown: unit.status == TokenStatus::Unknown,
                    conditional_log_score: 0.0,
                    selected: true,
                }],
                selected_unit: unit,
            })
            .collect::<Vec<_>>();

        let mut segment = Vec::new();
        for index in 0..work.len() {
            let unit = &work[index].selected_unit;
            if unit.mode != TokenMode::Turkish
                || matches!(
                    unit.kind,
                    LexicalKind::LineBreak | LexicalKind::Control | LexicalKind::Opaque
                )
            {
                self.flush_compact_sidecar_segment(raw_slice, &mut work, &mut segment, cache)?;
                continue;
            }
            if matches!(unit.kind, LexicalKind::Whitespace) {
                continue;
            }
            segment.push(index);
            if is_sentence_boundary(unit.kind, raw_slice, unit.span)?
                || segment.len() >= self.config.max_sentence_tokens
            {
                self.flush_compact_sidecar_segment(raw_slice, &mut work, &mut segment, cache)?;
            }
        }
        self.flush_compact_sidecar_segment(raw_slice, &mut work, &mut segment, cache)?;
        work =
            split_compact_sidecar_fallback_units(raw_slice, work, self.config.max_fallback_chars)?;

        let mut selected_units = work
            .iter()
            .map(|unit| unit.selected_unit.clone())
            .collect::<Vec<_>>();
        assign_inner_groups(raw_slice, &mut selected_units)?;
        let units = work
            .into_iter()
            .zip(selected_units)
            .map(|(work, selected)| NedoFormerSidecarUnit {
                span: selected.span,
                kind: selected.kind,
                mode: selected.mode,
                group_id: selected.group_id,
                candidates: work
                    .candidates
                    .into_iter()
                    .map(|candidate| NedoFormerSidecarCandidate {
                        cuts: candidate.cuts,
                        conditional_log_score: candidate.conditional_log_score,
                        selected: candidate.selected,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let (raw, _) = scan.into_document().into_parts();
        super::nedoformer_sidecar::encode_sidecar_units(&raw, &units)
    }

    fn flush_compact_sidecar_segment(
        &self,
        raw: &[u8],
        units: &mut [CompactSidecarWorkUnit],
        indices: &mut Vec<usize>,
        cache: &mut FlatAnalysisCache,
    ) -> Result<(), TokenizerError> {
        if indices.is_empty() {
            return Ok(());
        }
        let tokens = indices
            .iter()
            .map(|&index| unit_str(raw, units[index].selected_unit.span))
            .collect::<Result<Vec<_>, _>>()?;
        #[cfg(feature = "compiled-surface-table")]
        let sources = {
            let mut values = Vec::with_capacity(tokens.len());
            for token in &tokens {
                if let Some(set) = self
                    .nedoformer_compiled_surface_analysis_table
                    .as_ref()
                    .and_then(|table| table.get(token))
                {
                    values.push(NedoFormerFlatAnalysisSource::Compiled(set));
                } else {
                    values.push(NedoFormerFlatAnalysisSource::Live(
                        cache.analyze_nedoformer(&self.morphology, token)?,
                    ));
                }
            }
            values
        };
        #[cfg(feature = "compiled-surface-table")]
        let sets = sources
            .iter()
            .map(NedoFormerFlatAnalysisSource::set)
            .collect::<Vec<_>>();
        #[cfg(not(feature = "compiled-surface-table"))]
        let owned_sets = tokens
            .iter()
            .map(|token| cache.analyze_nedoformer(&self.morphology, token))
            .collect::<Result<Vec<_>, _>>()?;
        #[cfg(not(feature = "compiled-surface-table"))]
        let sets = owned_sets.iter().map(AsRef::as_ref).collect::<Vec<_>>();
        let ambiguity = sets
            .iter()
            .map(|set| set.ambiguity.as_ref())
            .collect::<Vec<_>>();
        let scoring_codes = sets
            .iter()
            .map(|set| set.scoring_codes.as_ref())
            .collect::<Vec<_>>();
        let selected = if self.config.contextual_disambiguation {
            self.disambiguator.disambiguate_indices_scored_causal(
                &ambiguity,
                &scoring_codes,
                &mut cache.disambiguation_scores,
            )?
        } else {
            vec![0; ambiguity.len()]
        };
        if selected.len() != indices.len() {
            return Err(TokenizerError::ContextLengthMismatch);
        }
        let conditional_scores = if self.config.contextual_disambiguation {
            self.disambiguator
                .causal_candidate_scores(&ambiguity, &selected)?
        } else {
            ambiguity
                .iter()
                .map(|values| vec![0.0; values.len()])
                .collect::<Vec<_>>()
        };

        for (((&unit_index, set), &selected_index), scores) in indices
            .iter()
            .zip(&sets)
            .zip(&selected)
            .zip(&conditional_scores)
        {
            if set.relative_cuts.len() != scores.len()
                || set.unknown.len() != scores.len()
                || selected_index >= scores.len()
            {
                return Err(TokenizerError::InvalidTrainingEncoding(
                    "NedoFormer compact candidate/score cardinality differs",
                ));
            }
            let mut classes = build_compact_sidecar_classes(
                raw,
                &units[unit_index].selected_unit,
                set,
                selected_index,
                scores,
            )?;
            classes.sort_by(compare_compact_sidecar_classes);
            if classes.iter().filter(|entry| entry.selected).count() != 1 {
                return Err(TokenizerError::InvalidTrainingEncoding(
                    "NedoFormer compact sidecar requires one selected class",
                ));
            }
            let selected_class = classes.iter().find(|entry| entry.selected).ok_or(
                TokenizerError::InvalidTrainingEncoding(
                    "NedoFormer compact sidecar selected class is absent",
                ),
            )?;
            units[unit_index]
                .selected_unit
                .cuts
                .clone_from(&selected_class.cuts);
            units[unit_index].selected_unit.status = if selected_class.unknown {
                TokenStatus::Unknown
            } else {
                TokenStatus::Morphological
            };
            units[unit_index].candidates = classes;
        }
        indices.clear();
        Ok(())
    }

    fn nedoformer_lattice_with_cache(
        &self,
        raw: Vec<u8>,
        cache: &mut AnalysisCache,
    ) -> Result<NedoFormerLatticeDocument, TokenizerError> {
        let scan = nedo_core::scan(raw)?;
        let raw_slice = scan.document().decode();
        let code_spans = match self.config.mode {
            TokenizerMode::Auto if self.config.detect_unmarked_code => {
                auto_code_spans(raw_slice, scan.code_hints())
            }
            TokenizerMode::Auto => explicit_code_spans(raw_slice),
            TokenizerMode::Turkish | TokenizerMode::Code => Vec::new(),
        };
        let units = split_units(&scan, &code_spans, self.config.mode)?;
        let mut lattice_units = units
            .into_iter()
            .map(|unit| NedoFormerLatticeUnit {
                candidates: vec![NedoFormerSegmentationCandidate {
                    cuts: unit.cuts.clone(),
                    status: unit.status,
                    analysis: unit.analysis.clone(),
                    analysis_count: 1,
                    conditional_log_score: 0.0,
                    selected: true,
                }],
                selected_unit: unit,
            })
            .collect::<Vec<_>>();

        let mut segment = Vec::new();
        for index in 0..lattice_units.len() {
            let unit = &lattice_units[index].selected_unit;
            if unit.mode != TokenMode::Turkish
                || matches!(
                    unit.kind,
                    LexicalKind::LineBreak | LexicalKind::Control | LexicalKind::Opaque
                )
            {
                self.flush_nedoformer_lattice_segment(
                    raw_slice,
                    &mut lattice_units,
                    &mut segment,
                    cache,
                )?;
                continue;
            }
            if matches!(unit.kind, LexicalKind::Whitespace) {
                continue;
            }
            segment.push(index);
            if is_sentence_boundary(unit.kind, raw_slice, unit.span)?
                || segment.len() >= self.config.max_sentence_tokens
            {
                self.flush_nedoformer_lattice_segment(
                    raw_slice,
                    &mut lattice_units,
                    &mut segment,
                    cache,
                )?;
            }
        }
        self.flush_nedoformer_lattice_segment(raw_slice, &mut lattice_units, &mut segment, cache)?;

        lattice_units = split_nedoformer_fallback_units(
            raw_slice,
            lattice_units,
            self.config.max_fallback_chars,
        )?;
        let mut selected_units = lattice_units
            .iter()
            .map(|unit| unit.selected_unit.clone())
            .collect::<Vec<_>>();
        assign_inner_groups(raw_slice, &mut selected_units)?;
        for (lattice, selected) in lattice_units.iter_mut().zip(&selected_units) {
            lattice.selected_unit.group_id = selected.group_id;
        }

        let (raw, _) = scan.into_document().into_parts();
        let lattice = NedoFormerLatticeDocument {
            raw,
            units: lattice_units,
        };
        let _ = lattice.selected_document()?;
        Ok(lattice)
    }

    #[allow(clippy::too_many_lines)] // Keeps one contextual segment's scoring/class collapse together.
    fn flush_nedoformer_lattice_segment(
        &self,
        raw: &[u8],
        lattice_units: &mut [NedoFormerLatticeUnit],
        indices: &mut Vec<usize>,
        cache: &mut AnalysisCache,
    ) -> Result<(), TokenizerError> {
        if indices.is_empty() {
            return Ok(());
        }
        let tokens = indices
            .iter()
            .map(|&index| unit_str(raw, lattice_units[index].selected_unit.span))
            .collect::<Result<Vec<_>, _>>()?;
        let mut sets = Vec::with_capacity(tokens.len());
        for token in &tokens {
            sets.push(cache.analyze_nedoformer(&self.morphology, token)?);
        }
        let ambiguity = sets
            .iter()
            .map(|set| set.ambiguity.as_ref())
            .collect::<Vec<_>>();
        let selected = if self.config.contextual_disambiguation {
            self.disambiguator.disambiguate_indices_causal(&ambiguity)?
        } else {
            vec![0; ambiguity.len()]
        };
        if selected.len() != indices.len() {
            return Err(TokenizerError::ContextLengthMismatch);
        }
        let conditional_scores = if self.config.contextual_disambiguation {
            self.disambiguator
                .causal_candidate_scores(&ambiguity, &selected)?
        } else {
            ambiguity
                .iter()
                .map(|values| vec![0.0; values.len()])
                .collect::<Vec<_>>()
        };

        for (((&unit_index, set), &selected_index), scores) in indices
            .iter()
            .zip(&sets)
            .zip(&selected)
            .zip(&conditional_scores)
        {
            if set.analyses.len() != scores.len() || selected_index >= set.analyses.len() {
                return Err(TokenizerError::InvalidTrainingEncoding(
                    "NedoFormer lattice analysis/score cardinality differs",
                ));
            }
            let base = lattice_units[unit_index].selected_unit.clone();
            let mut classes = Vec::<NedoFormerSegmentationCandidate>::new();
            for (candidate_index, (analysis, &score)) in set.analyses.iter().zip(scores).enumerate()
            {
                if !score.is_finite() {
                    return Err(TokenizerError::InvalidTrainingEncoding(
                        "NedoFormer lattice contains a non-finite model score",
                    ));
                }
                let mut realized = base.clone();
                super::apply_selected_analysis(raw, &mut realized, analysis.clone())?;
                let is_selected = candidate_index == selected_index;
                if let Some(existing) = classes
                    .iter_mut()
                    .find(|entry| entry.status == realized.status && entry.cuts == realized.cuts)
                {
                    let old = existing.conditional_log_score;
                    existing.conditional_log_score = log_add_exp(old, score);
                    existing.analysis_count = existing.analysis_count.saturating_add(1);
                    if is_selected || (!existing.selected && score > old) {
                        existing.analysis.clone_from(&realized.analysis);
                    }
                    existing.selected |= is_selected;
                } else {
                    classes.push(NedoFormerSegmentationCandidate {
                        cuts: realized.cuts,
                        status: realized.status,
                        analysis: realized.analysis,
                        analysis_count: 1,
                        conditional_log_score: score,
                        selected: is_selected,
                    });
                }
            }
            classes.sort_by(compare_candidates);
            let selected_classes = classes.iter().filter(|entry| entry.selected).count();
            if selected_classes != 1 {
                return Err(TokenizerError::InvalidTrainingEncoding(
                    "NedoFormer lattice must contain exactly one selected output class",
                ));
            }
            let selected_class = classes.iter().find(|entry| entry.selected).ok_or(
                TokenizerError::InvalidTrainingEncoding(
                    "NedoFormer lattice selected output class is absent",
                ),
            )?;
            lattice_units[unit_index]
                .selected_unit
                .cuts
                .clone_from(&selected_class.cuts);
            lattice_units[unit_index].selected_unit.status = selected_class.status;
            lattice_units[unit_index]
                .selected_unit
                .analysis
                .clone_from(&selected_class.analysis);
            lattice_units[unit_index].candidates = classes;
        }
        indices.clear();
        Ok(())
    }
}

fn build_compact_sidecar_classes(
    raw: &[u8],
    unit: &TokenizedUnit,
    set: &super::FlatAnalysisSet,
    selected_index: usize,
    scores: &[f32],
) -> Result<Vec<CompactSidecarClass>, TokenizerError> {
    let span = unit.span;
    let numeric_cuts = if unit.kind == LexicalKind::Number {
        super::numeric_micro_cuts(raw, span)?
    } else {
        Vec::new()
    };
    let mut classes = Vec::new();
    for (candidate_index, ((relative_cuts, &unknown), &score)) in set
        .relative_cuts
        .iter()
        .zip(set.unknown.iter())
        .zip(scores.iter())
        .enumerate()
    {
        let mut cuts = relative_cuts
            .iter()
            .map(|cut| {
                span.start
                    .checked_add(u64::from(*cut))
                    .ok_or(TokenizerError::LengthOverflow(
                        "NedoFormer compact sidecar cut",
                    ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !numeric_cuts.is_empty() {
            cuts.extend_from_slice(&numeric_cuts);
            cuts.sort_unstable();
            cuts.dedup();
        }
        if !score.is_finite() {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "NedoFormer compact sidecar contains a non-finite score",
            ));
        }
        let is_selected = candidate_index == selected_index;
        if let Some(existing) = classes
            .iter_mut()
            .find(|entry: &&mut CompactSidecarClass| entry.unknown == unknown && entry.cuts == cuts)
        {
            existing.conditional_log_score = log_add_exp(existing.conditional_log_score, score);
            existing.selected |= is_selected;
        } else {
            classes.push(CompactSidecarClass {
                cuts,
                unknown,
                conditional_log_score: score,
                selected: is_selected,
            });
        }
    }
    Ok(classes)
}

fn split_compact_sidecar_fallback_units(
    raw: &[u8],
    units: Vec<CompactSidecarWorkUnit>,
    maximum_chars: usize,
) -> Result<Vec<CompactSidecarWorkUnit>, TokenizerError> {
    let mut output = Vec::with_capacity(units.len());
    for work in units {
        let unit = &work.selected_unit;
        let should_split = matches!(unit.status, TokenStatus::Unknown | TokenStatus::Code)
            || unit.mode == TokenMode::Opaque;
        if !should_split {
            output.push(work);
            continue;
        }
        let spans = chunk_span(
            raw,
            unit.span,
            maximum_chars,
            unit.mode == TokenMode::Opaque,
        )?;
        if spans.len() == 1 {
            output.push(work);
            continue;
        }
        for span in spans {
            output.push(CompactSidecarWorkUnit {
                selected_unit: TokenizedUnit {
                    span,
                    kind: unit.kind,
                    mode: unit.mode,
                    status: unit.status,
                    group_id: None,
                    cuts: Vec::new(),
                    analysis: None,
                },
                candidates: vec![CompactSidecarClass {
                    cuts: Vec::new(),
                    unknown: unit.status == TokenStatus::Unknown,
                    conditional_log_score: 0.0,
                    selected: true,
                }],
            });
        }
    }
    Ok(output)
}

fn compare_compact_sidecar_classes(
    left: &CompactSidecarClass,
    right: &CompactSidecarClass,
) -> Ordering {
    right
        .selected
        .cmp(&left.selected)
        .then_with(|| left.cuts.cmp(&right.cuts))
        .then_with(|| left.unknown.cmp(&right.unknown))
}

fn split_nedoformer_fallback_units(
    raw: &[u8],
    units: Vec<NedoFormerLatticeUnit>,
    maximum_chars: usize,
) -> Result<Vec<NedoFormerLatticeUnit>, TokenizerError> {
    let mut output = Vec::with_capacity(units.len());
    for lattice in units {
        let unit = &lattice.selected_unit;
        let should_split = matches!(unit.status, TokenStatus::Unknown | TokenStatus::Code)
            || unit.mode == TokenMode::Opaque;
        if !should_split {
            output.push(lattice);
            continue;
        }
        let spans = chunk_span(
            raw,
            unit.span,
            maximum_chars,
            unit.mode == TokenMode::Opaque,
        )?;
        if spans.len() == 1 {
            output.push(lattice);
            continue;
        }
        for span in spans {
            let analysis = if unit.status == TokenStatus::Unknown {
                Some(unknown_chunk_analysis(raw, span)?)
            } else {
                None
            };
            let selected_unit = TokenizedUnit {
                span,
                kind: unit.kind,
                mode: unit.mode,
                status: unit.status,
                group_id: None,
                cuts: Vec::new(),
                analysis: analysis.clone(),
            };
            output.push(NedoFormerLatticeUnit {
                candidates: vec![NedoFormerSegmentationCandidate {
                    cuts: Vec::new(),
                    status: unit.status,
                    analysis,
                    analysis_count: 1,
                    conditional_log_score: 0.0,
                    selected: true,
                }],
                selected_unit,
            });
        }
    }
    Ok(output)
}

fn compare_candidates(
    left: &NedoFormerSegmentationCandidate,
    right: &NedoFormerSegmentationCandidate,
) -> Ordering {
    right
        .selected
        .cmp(&left.selected)
        .then_with(|| left.cuts.cmp(&right.cuts))
        .then_with(|| (left.status as u8).cmp(&(right.status as u8)))
}

fn log_add_exp(left: f32, right: f32) -> f32 {
    if left.is_infinite() && left.is_sign_negative() {
        return right;
    }
    if right.is_infinite() && right.is_sign_negative() {
        return left;
    }
    let maximum = left.max(right);
    maximum + ((left - maximum).exp() + (right - maximum).exp()).ln()
}

fn choose_candidate(
    candidates: &[NedoFormerSegmentationCandidate],
    policy: NedoFormerSamplingPolicy,
    rng: &mut SplitMix64,
) -> Result<usize, TokenizerError> {
    if candidates.is_empty() {
        return Err(TokenizerError::InvalidTrainingEncoding(
            "NedoFormer lattice unit has no candidates",
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
                "NedoFormer lattice unit has no selected candidate",
            )),
        NedoFormerSamplingPolicy::Uniform => {
            let length = u64::try_from(candidates.len())
                .map_err(|_| TokenizerError::LengthOverflow("lattice candidates"))?;
            Ok(usize::try_from(rng.next_u64() % length)
                .map_err(|_| TokenizerError::LengthOverflow("sampled lattice index"))?)
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
                    "NedoFormer lattice sampling weights are invalid",
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

fn write_analysis(output: &mut Vec<u8>, analysis: &AnalysisMetadata) -> Result<(), TokenizerError> {
    write_string(output, &analysis.canonical)?;
    write_string(output, &analysis.dictionary_id)?;
    write_string(output, &analysis.lemma)?;
    write_string(output, &analysis.primary_pos)?;
    write_string(output, &analysis.secondary_pos)?;
    write_u32(
        output,
        u32::try_from(analysis.morphemes.len())
            .map_err(|_| TokenizerError::LengthOverflow("NedoFormer lattice morphemes"))?,
    );
    for morpheme in &analysis.morphemes {
        write_string(output, &morpheme.id)?;
        write_string(output, &morpheme.surface)?;
        write_u64(output, morpheme.span.start);
        write_u64(output, morpheme.span.end);
        output.push(u8::from(morpheme.derivational));
    }
    Ok(())
}

fn read_analysis(reader: &mut LatticeReader<'_>) -> Result<AnalysisMetadata, TokenizerError> {
    let canonical = reader.string()?;
    let dictionary_id = reader.string()?;
    let lemma = reader.string()?;
    let primary_pos = reader.string()?;
    let secondary_pos = reader.string()?;
    let count = reader.usize32("NedoFormer lattice morpheme count")?;
    if count > reader.remaining() / 21 {
        return Err(TokenizerError::ImpossibleCodecCount(
            "NedoFormer lattice morpheme count",
        ));
    }
    let mut morphemes = Vec::with_capacity(count);
    for _ in 0..count {
        morphemes.push(AlignedMorpheme {
            id: reader.string()?,
            surface: reader.string()?,
            span: ByteSpan {
                start: reader.u64()?,
                end: reader.u64()?,
            },
            derivational: reader.boolean("NedoFormer lattice derivational")?,
        });
    }
    Ok(AnalysisMetadata {
        canonical,
        dictionary_id,
        lemma,
        primary_pos,
        secondary_pos,
        morphemes,
    })
}

fn write_string(output: &mut Vec<u8>, value: &str) -> Result<(), TokenizerError> {
    write_u32(
        output,
        u32::try_from(value.len())
            .map_err(|_| TokenizerError::LengthOverflow("NedoFormer lattice string"))?,
    );
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn write_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn write_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

const fn lattice_lexical_kind(value: u8) -> Result<LexicalKind, TokenizerError> {
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
            "NedoFormer lattice lexical kind",
            value,
        )),
    }
}

struct LatticeReader<'a> {
    input: &'a [u8],
    position: usize,
}

impl<'a> LatticeReader<'a> {
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
            .ok_or(TokenizerError::LengthOverflow("NedoFormer lattice reader"))?;
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

    fn boolean(&mut self, field: &'static str) -> Result<bool, TokenizerError> {
        let value = self.u8()?;
        match value {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(TokenizerError::InvalidCodecBoolean { field, value }),
        }
    }

    fn u32(&mut self) -> Result<u32, TokenizerError> {
        Ok(u32::from_le_bytes(
            self.bytes(4)?
                .try_into()
                .map_err(|_| TokenizerError::TruncatedCodec)?,
        ))
    }

    fn u64(&mut self) -> Result<u64, TokenizerError> {
        Ok(u64::from_le_bytes(
            self.bytes(8)?
                .try_into()
                .map_err(|_| TokenizerError::TruncatedCodec)?,
        ))
    }

    fn usize32(&mut self, field: &'static str) -> Result<usize, TokenizerError> {
        usize::try_from(self.u32()?).map_err(|_| TokenizerError::LengthOverflow(field))
    }

    fn usize64(&mut self, field: &'static str) -> Result<usize, TokenizerError> {
        usize::try_from(self.u64()?).map_err(|_| TokenizerError::LengthOverflow(field))
    }

    fn string(&mut self) -> Result<String, TokenizerError> {
        let length = self.usize32("NedoFormer lattice string length")?;
        let bytes = self.bytes(length)?;
        Ok(std::str::from_utf8(bytes)
            .map_err(|_| TokenizerError::InvalidUtf8Unit)?
            .to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::NedoFormerSamplingPolicy;
    use crate::{ContextCachePolicy, Tokenizer, TokenizerConfig};

    #[test]
    fn real_ambiguity_exposes_multiple_distinct_cut_classes() -> Result<(), crate::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let mut found = None;
        for raw in ["koyun", "yazar", "dolar", "kazma", "asma", "oyun"] {
            let lattice = tokenizer.nedoformer_lattice(raw.as_bytes().to_vec())?;
            if let Some(unit) = lattice
                .units()
                .iter()
                .find(|unit| unit.candidates.len() > 1)
            {
                found = Some((raw, unit.candidates.len()));
                break;
            }
        }
        let (surface, classes) = found.ok_or(crate::TokenizerError::InvalidTrainingEncoding(
            "NedoFormer ambiguity probe found no multi-cut surface",
        ))?;
        assert!(!surface.is_empty());
        assert!(classes > 1);
        Ok(())
    }

    #[test]
    fn lattice_codec_preserves_sampling_and_rejects_corruption() -> Result<(), crate::TokenizerError>
    {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let raw = "koyun evi cocuklarimizdan 23.07.2026".as_bytes().to_vec();
        let lattice = tokenizer.nedoformer_lattice(raw)?;
        let before = lattice.sample(
            NedoFormerSamplingPolicy::ContextWeighted { temperature: 0.8 },
            123,
        )?;
        let bytes = lattice.to_bytes()?;
        let loaded = super::NedoFormerLatticeDocument::from_bytes(&bytes)?;
        assert_eq!(loaded.to_bytes()?, bytes);
        let after = loaded.sample(
            NedoFormerSamplingPolicy::ContextWeighted { temperature: 0.8 },
            123,
        )?;
        assert_eq!(before, after);

        let mut corrupted = bytes;
        let last = corrupted
            .last_mut()
            .ok_or(crate::TokenizerError::TruncatedCodec)?;
        *last ^= 1;
        assert!(super::NedoFormerLatticeDocument::from_bytes(&corrupted).is_err());
        Ok(())
    }

    #[test]
    fn lattice_batch_matches_independent_documents_and_preserves_order(
    ) -> Result<(), crate::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let inputs = vec![
            "koyun çocuklarımızdan geliyor mu?".as_bytes().to_vec(),
            "Ankara'da 23.07.2026".as_bytes().to_vec(),
            b"```rust\nfn parseHttpRequest2XX() {}\n```".to_vec(),
            "cocuklarimizdan geliyor mu?".as_bytes().to_vec(),
            b"raw\x00\xffbytes".to_vec(),
        ];
        let independent = inputs
            .iter()
            .cloned()
            .map(|raw| tokenizer.nedoformer_lattice(raw))
            .collect::<Result<Vec<_>, _>>()?;
        for threads in [1_usize, 2, 4] {
            let batched = tokenizer.nedoformer_lattice_batch(&inputs, threads)?;
            assert_eq!(batched, independent);
        }
        assert!(tokenizer.nedoformer_lattice_batch(&inputs, 0).is_err());
        Ok(())
    }

    #[test]
    fn sidecar_batch_matches_independent_serializers_across_threads(
    ) -> Result<(), crate::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let inputs = vec![
            "koyun çocuklarımızdan geliyor mu?".as_bytes().to_vec(),
            "Ankara'da 23.07.2026".as_bytes().to_vec(),
            b"```rust\nfn parseHttpRequest2XX() {}\n```".to_vec(),
            "cocuklarimizdan geliyor mu?".as_bytes().to_vec(),
            b"raw\x00\xffbytes".to_vec(),
        ];
        let expected = inputs
            .iter()
            .cloned()
            .map(|raw| tokenizer.nedoformer_lattice(raw)?.to_sidecar_bytes())
            .collect::<Result<Vec<_>, _>>()?;
        for threads in [1_usize, 2, 4] {
            assert_eq!(
                tokenizer.nedoformer_sidecar_batch(&inputs, threads)?,
                expected
            );
        }
        assert!(tokenizer.nedoformer_sidecar_batch(&inputs, 0).is_err());
        Ok(())
    }

    #[test]
    fn persistent_best_runtime_matches_stateless_across_calls() -> Result<(), crate::TokenizerError>
    {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let inputs = vec![
            b"koyun koyun geliyor mu?".to_vec(),
            "cocuklarimizdan Ankara'da".as_bytes().to_vec(),
            b"parseHttpRequest2XX foo_bar".to_vec(),
        ];
        let expected = tokenizer.nedoformer_best_sidecar_batch(&inputs, 2)?;
        let mut runtime = tokenizer.nedoformer_best_runtime(2, ContextCachePolicy::Balanced)?;
        assert_eq!(
            tokenizer.nedoformer_best_sidecar_batch_with_runtime(&inputs, &mut runtime)?,
            expected
        );
        assert_eq!(
            tokenizer.nedoformer_best_sidecar_batch_with_runtime(&inputs, &mut runtime)?,
            expected
        );
        Ok(())
    }

    #[test]
    fn best_sidecar_matches_full_lattice_best_sample_across_threads(
    ) -> Result<(), crate::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let inputs = vec![
            b"koyun koyun evi geliyor mu?".to_vec(),
            "Ankara'da 23.07.2026 çocuklarımızdan".as_bytes().to_vec(),
            b"```python\nparseHttpRequest2XX(foo_bar)\n```".to_vec(),
            vec![b'q'; 130],
        ];
        let full = tokenizer.nedoformer_sidecar_batch(&inputs, 1)?;
        for threads in [1_usize, 2, 4] {
            let best = tokenizer.nedoformer_best_sidecar_batch(&inputs, threads)?;
            assert_eq!(best.len(), full.len());
            for ((raw, full_bytes), best_bytes) in inputs.iter().zip(&full).zip(&best) {
                let full_sidecar =
                    crate::NedoFormerLatticeSidecar::from_bytes(raw.clone(), full_bytes)?;
                let best_sidecar =
                    crate::NedoFormerLatticeSidecar::from_bytes(raw.clone(), best_bytes)?;
                assert_eq!(
                    full_sidecar.sample_lossless(NedoFormerSamplingPolicy::Best, 0)?,
                    best_sidecar.sample_lossless(NedoFormerSamplingPolicy::Best, 0)?
                );
            }
        }
        Ok(())
    }

    #[test]
    fn selected_lattice_path_matches_canonical_tokenization() -> Result<(), crate::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let raw = "Koyun evlerde mi? 23.07.2026 parseHttpRequest2XX"
            .as_bytes()
            .to_vec();
        let expected = tokenizer.tokenize(raw.clone())?;
        let lattice = tokenizer.nedoformer_lattice(raw)?;
        assert_eq!(lattice.selected_document()?, expected);
        assert_eq!(lattice.raw(), expected.raw());
        Ok(())
    }

    #[test]
    fn shadow_deasciify_recovers_original_byte_boundaries() -> Result<(), crate::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let raw = b"cocuklarimizdan".to_vec();
        let lattice = tokenizer.nedoformer_lattice(raw.clone())?;
        let selected = lattice.selected_document()?;
        assert_eq!(selected.decode(), raw.as_slice());
        let lexical = selected
            .units()
            .iter()
            .find(|unit| unit.analysis.is_some())
            .ok_or(crate::TokenizerError::InvalidTrainingEncoding(
                "shadow deasciify test has no lexical analysis",
            ))?;
        assert!(lexical.cuts.len() >= 2);
        let analysis =
            lexical
                .analysis
                .as_ref()
                .ok_or(crate::TokenizerError::InvalidTrainingEncoding(
                    "shadow analysis disappeared",
                ))?;
        assert!(
            analysis
                .dictionary_id
                .starts_with("NEDO_ShadowDeasciify_Fallback")
                || analysis
                    .dictionary_id
                    .starts_with("NEDO_ShadowLower_Fallback")
                || !analysis.dictionary_id.starts_with("UNK_")
        );
        Ok(())
    }

    #[test]
    fn lattice_sampling_is_seed_deterministic_and_lossless() -> Result<(), crate::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let raw = "koyun koyun evi".as_bytes().to_vec();
        let lattice = tokenizer.nedoformer_lattice(raw.clone())?;
        let first = lattice.sample(
            NedoFormerSamplingPolicy::ContextWeighted { temperature: 1.0 },
            7,
        )?;
        let second = lattice.sample(
            NedoFormerSamplingPolicy::ContextWeighted { temperature: 1.0 },
            7,
        )?;
        assert_eq!(first, second);
        assert_eq!(first.decode(), raw.as_slice());
        let uniform = lattice.sample(NedoFormerSamplingPolicy::Uniform, 11)?;
        assert_eq!(uniform.decode(), raw.as_slice());
        Ok(())
    }
}
