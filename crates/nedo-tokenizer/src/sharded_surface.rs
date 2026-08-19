//! Stable one-thread-per-shard surface encoding.

use std::{ops::Range, thread};

use crate::{
    analysis_cache_entries_for_parallelism, concatenate_surface_chunks, FlatSurfaceEncoder,
    SurfaceEncoderOptions, SurfaceRuntimeCache, SurfaceVocabulary, Tokenizer, TokenizerError,
    TrainingBatch, TrainingCacheStats,
};

/// One independently encoded contiguous document shard.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SurfaceShardBatch {
    /// Stable zero-based shard position in the encoder pool.
    pub shard_index: usize,
    /// Original input document interval represented by this shard.
    pub document_range: Range<usize>,
    /// Local batch whose document offsets start at zero.
    pub batch: TrainingBatch,
}

/// Owned persistent pool of exact single-thread surface runtimes.
///
/// This type does not borrow the tokenizer or vocabulary, so language bindings
/// can retain exact shard-local caches safely across repeated calls.
pub struct ShardedSurfaceRuntimeCache {
    runtimes: Vec<SurfaceRuntimeCache>,
}

impl ShardedSurfaceRuntimeCache {
    /// Number of retained independent shard runtimes.
    #[must_use]
    pub const fn shard_count(&self) -> usize {
        self.runtimes.len()
    }

    /// Returns aggregate cache telemetry across retained shards.
    #[must_use]
    pub fn cache_stats(&self) -> TrainingCacheStats {
        let mut stats = TrainingCacheStats::default();
        for runtime in &self.runtimes {
            stats.merge(runtime.cache_stats());
        }
        stats
    }

    /// Clears every retained shard runtime.
    pub fn clear(&mut self) {
        for runtime in &mut self.runtimes {
            runtime.clear();
        }
    }
}

/// Persistent pool of independent single-thread surface encoders.
///
/// Every shard owns its morphology, segment, surface, and whole-document caches.
/// Callers may consume [`Self::encode_shards`] directly to avoid concatenating
/// large output vectors, or use [`Self::encode_batch`] for a single ordered batch.
pub struct ShardedSurfaceEncoder<'tokenizer, 'assets> {
    encoders: Vec<FlatSurfaceEncoder<'tokenizer, 'assets>>,
}

impl<'tokenizer, 'assets> ShardedSurfaceEncoder<'tokenizer, 'assets> {
    /// Encodes one input batch into deterministic contiguous shard outputs.
    ///
    /// Each active shard runs exactly one encoder thread. Returned entries are
    /// ordered by `shard_index`, and their document ranges exactly cover the
    /// input once without overlap.
    ///
    /// # Errors
    ///
    /// Returns an error for input/newline cardinality mismatch, worker panic,
    /// range overflow, or any underlying tokenizer/encoding failure.
    pub fn encode_shards(
        &mut self,
        inputs: &[Vec<u8>],
        newline_flags: &[bool],
    ) -> Result<Vec<SurfaceShardBatch>, TokenizerError> {
        if inputs.len() != newline_flags.len() {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "sharded surface input and newline counts differ",
            ));
        }
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let ranges = build_shard_ranges(inputs, self.encoders.len())?;
        if ranges.len() == 1 {
            let range = ranges[0].clone();
            let encoder = self
                .encoders
                .first_mut()
                .ok_or(TokenizerError::InvalidConfiguration(
                    "sharded surface encoder has no shard",
                ))?;
            let batch =
                encoder.encode_batch(&inputs[range.clone()], &newline_flags[range.clone()])?;
            return Ok(vec![SurfaceShardBatch {
                shard_index: 0,
                document_range: range,
                batch,
            }]);
        }
        thread::scope(|scope| {
            let handles = self.encoders[..ranges.len()]
                .iter_mut()
                .zip(ranges)
                .enumerate()
                .map(|(shard_index, (encoder, document_range))| {
                    scope.spawn(move || {
                        let batch = encoder.encode_batch(
                            &inputs[document_range.clone()],
                            &newline_flags[document_range.clone()],
                        )?;
                        Ok::<_, TokenizerError>(SurfaceShardBatch {
                            shard_index,
                            document_range,
                            batch,
                        })
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().map_err(|_| TokenizerError::WorkerPanicked)?)
                .collect::<Result<Vec<_>, _>>()
        })
    }

    /// Encodes and merges all shards into one document-ordered batch.
    ///
    /// # Errors
    ///
    /// Returns any error from [`Self::encode_shards`] or [`merge_surface_shards`].
    pub fn encode_batch(
        &mut self,
        inputs: &[Vec<u8>],
        newline_flags: &[bool],
    ) -> Result<TrainingBatch, TokenizerError> {
        merge_surface_shards(self.encode_shards(inputs, newline_flags)?)
    }

    /// Number of persistent single-thread encoder shards.
    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.encoders.len()
    }

    /// Returns aggregate cache telemetry across all shards.
    #[must_use]
    pub fn cache_stats(&self) -> TrainingCacheStats {
        let mut stats = TrainingCacheStats::default();
        for encoder in &self.encoders {
            stats.merge(encoder.cache_stats());
        }
        stats
    }

    /// Clears every shard-local cache and counter.
    pub fn clear_caches(&mut self) {
        for encoder in &mut self.encoders {
            encoder.clear_caches();
        }
    }
}

impl<'assets> Tokenizer<'assets> {
    /// Creates owned persistent single-thread shard runtimes.
    ///
    /// # Errors
    ///
    /// Returns an error when `shards` is zero or a runtime cannot be built.
    pub fn sharded_surface_runtime_cache(
        &self,
        shards: usize,
        options: SurfaceEncoderOptions,
    ) -> Result<ShardedSurfaceRuntimeCache, TokenizerError> {
        if shards == 0 {
            return Err(TokenizerError::InvalidConfiguration(
                "surface shard count must be positive",
            ));
        }
        let analysis_cache_entries = analysis_cache_entries_for_parallelism(shards);
        let runtimes = (0..shards)
            .map(|_| {
                self.surface_runtime_cache_with_analysis_entries(1, options, analysis_cache_entries)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ShardedSurfaceRuntimeCache { runtimes })
    }

    /// Encodes one batch through retained exact shard runtimes.
    ///
    /// # Errors
    ///
    /// Returns errors for cardinality mismatch, incompatible retained runtimes,
    /// worker panic, or any underlying encoding failure.
    pub fn encode_surface_batch_with_sharded_runtime(
        &self,
        inputs: &[Vec<u8>],
        newline_flags: &[bool],
        vocabulary: &SurfaceVocabulary,
        runtime: &mut ShardedSurfaceRuntimeCache,
    ) -> Result<TrainingBatch, TokenizerError> {
        if inputs.len() != newline_flags.len() {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "sharded surface input and newline counts differ",
            ));
        }
        if inputs.is_empty() {
            return Ok(TrainingBatch {
                ids: Vec::new(),
                lengths: Vec::new(),
                document_offsets: vec![0],
            });
        }
        let ranges = build_shard_ranges(inputs, runtime.runtimes.len())?;
        if ranges.len() == 1 {
            let range = ranges[0].clone();
            let shard_runtime =
                runtime
                    .runtimes
                    .first_mut()
                    .ok_or(TokenizerError::InvalidConfiguration(
                        "sharded surface runtime has no shard",
                    ))?;
            let batch = self.encode_surface_batch_with_runtime(
                &inputs[range.clone()],
                &newline_flags[range.clone()],
                vocabulary,
                shard_runtime,
            )?;
            return merge_surface_shards(vec![SurfaceShardBatch {
                shard_index: 0,
                document_range: range,
                batch,
            }]);
        }
        let shards = thread::scope(|scope| {
            let handles = runtime.runtimes[..ranges.len()]
                .iter_mut()
                .zip(ranges)
                .enumerate()
                .map(|(shard_index, (shard_runtime, document_range))| {
                    scope.spawn(move || {
                        let batch = self.encode_surface_batch_with_runtime(
                            &inputs[document_range.clone()],
                            &newline_flags[document_range.clone()],
                            vocabulary,
                            shard_runtime,
                        )?;
                        Ok::<_, TokenizerError>(SurfaceShardBatch {
                            shard_index,
                            document_range,
                            batch,
                        })
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().map_err(|_| TokenizerError::WorkerPanicked)?)
                .collect::<Result<Vec<_>, _>>()
        })?;
        merge_surface_shards(shards)
    }

    /// Creates persistent independent one-thread surface encoder shards.
    ///
    /// This is the production scaling API for large corpora. It avoids the
    /// dynamic multi-thread scheduler inside one encoder and keeps each shard's
    /// caches private and reusable across calls.
    ///
    /// # Errors
    ///
    /// Returns an error when `shards` is zero or a shard encoder cannot be built.
    pub fn sharded_surface_encoder<'tokenizer>(
        &'tokenizer self,
        vocabulary: &'tokenizer SurfaceVocabulary,
        shards: usize,
        use_morphology: bool,
    ) -> Result<ShardedSurfaceEncoder<'tokenizer, 'assets>, TokenizerError> {
        self.sharded_surface_encoder_with_options(
            vocabulary,
            shards,
            SurfaceEncoderOptions::cached(use_morphology),
        )
    }

    /// Creates persistent independent shards with an explicit cache policy.
    ///
    /// # Errors
    ///
    /// Returns an error when `shards` is zero or a shard encoder cannot be built.
    pub fn sharded_surface_encoder_with_options<'tokenizer>(
        &'tokenizer self,
        vocabulary: &'tokenizer SurfaceVocabulary,
        shards: usize,
        options: SurfaceEncoderOptions,
    ) -> Result<ShardedSurfaceEncoder<'tokenizer, 'assets>, TokenizerError> {
        if shards == 0 {
            return Err(TokenizerError::InvalidConfiguration(
                "surface shard count must be positive",
            ));
        }
        let analysis_cache_entries = analysis_cache_entries_for_parallelism(shards);
        let encoders = (0..shards)
            .map(|_| {
                self.surface_encoder_with_analysis_cache_entries(
                    vocabulary,
                    1,
                    options,
                    analysis_cache_entries,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ShardedSurfaceEncoder { encoders })
    }
}

/// Merges independently encoded contiguous shards into one ordered batch.
///
/// The function rejects duplicate, overlapping, missing, reversed, or
/// cardinality-inconsistent document ranges before moving any token vectors.
///
/// # Errors
///
/// Returns an error when shard metadata is not one exact contiguous cover or
/// when merged token/document offsets overflow.
pub fn merge_surface_shards(
    mut shards: Vec<SurfaceShardBatch>,
) -> Result<TrainingBatch, TokenizerError> {
    if shards.is_empty() {
        return Ok(TrainingBatch {
            ids: Vec::new(),
            lengths: Vec::new(),
            document_offsets: vec![0],
        });
    }
    shards.sort_unstable_by_key(|shard| shard.document_range.start);
    let mut expected_start = 0_usize;
    let mut seen_indices = vec![false; shards.len()];
    for shard in &shards {
        if shard.shard_index >= seen_indices.len() || seen_indices[shard.shard_index] {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "surface shard index is duplicate or out of range",
            ));
        }
        seen_indices[shard.shard_index] = true;
        if shard.document_range.start != expected_start
            || shard.document_range.end <= shard.document_range.start
        {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "surface shard ranges are not one contiguous cover",
            ));
        }
        let local_documents = shard.batch.document_offsets.len().saturating_sub(1);
        if local_documents != shard.document_range.len() {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "surface shard document count differs from its range",
            ));
        }
        expected_start = shard.document_range.end;
    }
    concatenate_surface_chunks(
        shards
            .into_iter()
            .map(|shard| (shard.document_range.start, shard.batch))
            .collect(),
    )
}

fn build_shard_ranges(
    inputs: &[Vec<u8>],
    requested_shards: usize,
) -> Result<Vec<Range<usize>>, TokenizerError> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    if requested_shards == 0 {
        return Err(TokenizerError::InvalidConfiguration(
            "surface shard count must be positive",
        ));
    }
    let shard_count = requested_shards.min(inputs.len());
    let total_weight = inputs.iter().try_fold(0_usize, |total, input| {
        total
            .checked_add(input.len().max(1))
            .ok_or(TokenizerError::LengthOverflow("surface shard bytes"))
    })?;
    let mut ranges = Vec::with_capacity(shard_count);
    let mut start = 0_usize;
    let mut consumed_weight = 0_usize;
    for shard_index in 0..shard_count {
        let remaining_shards = shard_count - shard_index;
        if remaining_shards == 1 {
            ranges.push(start..inputs.len());
            break;
        }
        let remaining_weight = total_weight.saturating_sub(consumed_weight);
        let target_weight = remaining_weight.div_ceil(remaining_shards);
        let max_end = inputs.len() - (remaining_shards - 1);
        let mut end = start;
        let mut shard_weight = 0_usize;
        while end < max_end && (end == start || shard_weight < target_weight) {
            shard_weight = shard_weight
                .checked_add(inputs[end].len().max(1))
                .ok_or(TokenizerError::LengthOverflow("surface shard range bytes"))?;
            end += 1;
        }
        ranges.push(start..end);
        start = end;
        consumed_weight =
            consumed_weight
                .checked_add(shard_weight)
                .ok_or(TokenizerError::LengthOverflow(
                    "surface shard consumed bytes",
                ))?;
    }
    Ok(ranges)
}

#[cfg(test)]
mod tests {
    use super::{build_shard_ranges, merge_surface_shards, SurfaceShardBatch};
    use crate::{
        SurfaceEncoderOptions, SurfaceVocabulary, Tokenizer, TokenizerConfig, TokenizerError,
        TrainingBatch,
    };

    #[test]
    fn byte_weighted_ranges_cover_every_document_once() -> Result<(), TokenizerError> {
        let inputs = vec![
            vec![],
            vec![0; 1],
            vec![0; 20],
            vec![0; 3],
            vec![0; 40],
            vec![0; 2],
        ];
        let ranges = build_shard_ranges(&inputs, 4)?;
        assert_eq!(ranges.first().map(|range| range.start), Some(0));
        assert_eq!(ranges.last().map(|range| range.end), Some(inputs.len()));
        assert!(ranges.windows(2).all(|pair| pair[0].end == pair[1].start));
        assert!(ranges.iter().all(|range| !range.is_empty()));
        Ok(())
    }

    #[test]
    fn sharded_surface_output_matches_single_encoder() -> Result<(), TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let vocabulary = SurfaceVocabulary::from_ranked(Vec::new())?;
        let inputs = vec![
            "Ankara'da evlerimizden mi?".as_bytes().to_vec(),
            "napıyon kanka bişey dicem".as_bytes().to_vec(),
            b"fn parseHttpRequest_header42() { return 0; }".to_vec(),
            "14:30:05'te görüşürüz 👩‍💻".as_bytes().to_vec(),
            vec![0, 0xff, b'a', 0x80, b'Z'],
            "Çekoslovakyalılaştıramadıklarımızdan".as_bytes().to_vec(),
            b"request_id=abc-123 status=200".to_vec(),
            Vec::new(),
        ];
        let newline_flags = vec![true, false, true, false, true, false, true, false];
        let mut single = tokenizer.surface_encoder(&vocabulary, 1, true)?;
        let expected = single.encode_batch(&inputs, &newline_flags)?;
        let mut sharded = tokenizer.sharded_surface_encoder(&vocabulary, 4, true)?;
        assert_eq!(sharded.shard_count(), 4);
        let shards = sharded.encode_shards(&inputs, &newline_flags)?;
        assert_eq!(shards.len(), 4);
        assert_eq!(merge_surface_shards(shards)?, expected);
        assert_eq!(sharded.encode_batch(&inputs, &newline_flags)?, expected);
        assert!(sharded.cache_stats().entries > 0);
        sharded.clear_caches();
        assert_eq!(sharded.cache_stats(), crate::TrainingCacheStats::default());
        Ok(())
    }

    #[test]
    fn one_pass_policy_matches_cached_output_without_document_retention(
    ) -> Result<(), TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let vocabulary = SurfaceVocabulary::from_ranked(Vec::new())?;
        let inputs = vec![
            "Ankara'da evlerimizden geliyoruz.".as_bytes().to_vec(),
            "geliyom kanka kodu da yazdım".as_bytes().to_vec(),
            b"fn parse_request_v2() { return 42; }".to_vec(),
        ];
        let newline_flags = vec![false, true, false];
        let mut cached = tokenizer.sharded_surface_encoder(&vocabulary, 2, true)?;
        let expected = cached.encode_batch(&inputs, &newline_flags)?;
        let mut one_pass = tokenizer.sharded_surface_encoder_with_options(
            &vocabulary,
            2,
            SurfaceEncoderOptions::one_pass(true),
        )?;
        assert_eq!(one_pass.encode_batch(&inputs, &newline_flags)?, expected);
        assert_eq!(one_pass.encode_batch(&inputs, &newline_flags)?, expected);
        let mut compact = tokenizer.sharded_surface_encoder_with_options(
            &vocabulary,
            2,
            SurfaceEncoderOptions::one_pass_compact(true),
        )?;
        assert_eq!(compact.encode_batch(&inputs, &newline_flags)?, expected);
        assert!(one_pass
            .encoders
            .iter()
            .all(|encoder| encoder.runtime.caches.iter().all(|cache| {
                cache.document_program_capacity == 0
                    && cache.document_program_entries == 0
                    && cache.document_program_bytes == 0
            })));
        Ok(())
    }

    #[test]
    fn persistent_sharded_runtime_matches_encoder_and_reuses_cache() -> Result<(), TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let vocabulary = SurfaceVocabulary::from_ranked(Vec::new())?;
        let inputs = vec![
            "Evlerimizden geldik ve yarın yine evlerimizden çıkacağız."
                .as_bytes()
                .to_vec(),
            "Ankara'da evlerimizden mi?".as_bytes().to_vec(),
            "Evlerimizden geldik ve yarın yine evlerimizden çıkacağız."
                .as_bytes()
                .to_vec(),
            b"fn parse_request_v2() { return 42; }".to_vec(),
        ];
        let newline_flags = vec![false; inputs.len()];
        let mut reference = tokenizer.sharded_surface_encoder_with_options(
            &vocabulary,
            2,
            SurfaceEncoderOptions::one_pass_compact(true),
        )?;
        let expected = reference.encode_batch(&inputs, &newline_flags)?;
        let mut runtime = tokenizer
            .sharded_surface_runtime_cache(2, SurfaceEncoderOptions::one_pass_compact(true))?;
        let first = tokenizer.encode_surface_batch_with_sharded_runtime(
            &inputs,
            &newline_flags,
            &vocabulary,
            &mut runtime,
        )?;
        let first_stats = runtime.cache_stats();
        let second = tokenizer.encode_surface_batch_with_sharded_runtime(
            &inputs,
            &newline_flags,
            &vocabulary,
            &mut runtime,
        )?;
        let second_stats = runtime.cache_stats();
        assert_eq!(first, expected);
        assert_eq!(second, expected);
        assert!(second_stats.hits > first_stats.hits);
        assert_eq!(second_stats.misses, first_stats.misses);
        runtime.clear();
        assert_eq!(runtime.cache_stats(), crate::TrainingCacheStats::default());
        Ok(())
    }

    #[test]
    fn merge_rejects_noncontiguous_ranges() {
        let empty = || TrainingBatch {
            ids: Vec::new(),
            lengths: Vec::new(),
            document_offsets: vec![0, 0],
        };
        let result = merge_surface_shards(vec![
            SurfaceShardBatch {
                shard_index: 0,
                document_range: 0..1,
                batch: empty(),
            },
            SurfaceShardBatch {
                shard_index: 1,
                document_range: 2..3,
                batch: empty(),
            },
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn zero_shards_are_rejected() -> Result<(), TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let vocabulary = SurfaceVocabulary::from_ranked(Vec::new())?;
        assert!(tokenizer
            .sharded_surface_encoder(&vocabulary, 0, true)
            .is_err());
        Ok(())
    }
}
