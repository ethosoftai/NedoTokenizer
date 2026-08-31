//! Python bindings for the native `NedoTokenizer` engine.

#![forbid(unsafe_code)]

use core::fmt::Write as _;
use std::collections::HashMap;
use std::sync::Mutex;

use nedo_tokenizer::{
    decode_tokenized, encode_tokenized, CharacterVocabulary, CompiledSurfaceAnalysisTable,
    NedoFormerInputEncoding, NedoFormerLatticeDocument, NedoFormerLatticeSidecar,
    NedoFormerSamplingPolicy, NedoFormerVocabulary, ShardedSurfaceRuntimeCache,
    SurfaceEncoderOptions, SurfaceRuntimeCache, SurfaceVocabulary, TokenMode, TokenStatus,
    Tokenizer as NativeTokenizer, TokenizerConfig, TokenizerMode, MODEL_SHA256, MORPHOLOGY_SHA256,
    NEDOFORMER_INPUT_ENCODING_VERSION, NEDOFORMER_LATTICE_SCHEMA_VERSION,
    NEDOFORMER_SIDECAR_SCHEMA_VERSION, NEDOFORMER_TOKENIZER_CONTRACT_VERSION, SURFACE_BOS_ID,
    SURFACE_BYTE_BASE_ID, SURFACE_ENTRY_BASE_ID, SURFACE_EOS_ID, SURFACE_PAD_ID,
    TOKENIZER_SCHEMA_VERSION,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyList};
use serde_json::{json, Value};

/// Python-facing tokenizer whose work crosses the language boundary by document batch.
#[pyclass(module = "nedotokenizer._native", frozen)]
struct Tokenizer {
    inner: NativeTokenizer<'static>,
}

#[pymethods]
impl Tokenizer {
    /// Creates a self-contained native tokenizer.
    #[new]
    #[pyo3(signature = (mode = "auto", max_sentence_tokens = 512, max_fallback_chars = 48, contextual_disambiguation = true, detect_unmarked_code = true))]
    fn new(
        mode: &str,
        max_sentence_tokens: usize,
        max_fallback_chars: usize,
        contextual_disambiguation: bool,
        detect_unmarked_code: bool,
    ) -> PyResult<Self> {
        let inner = native_tokenizer(
            mode,
            max_sentence_tokens,
            max_fallback_chars,
            contextual_disambiguation,
            detect_unmarked_code,
        )?;
        Ok(Self { inner })
    }

    /// Tokenizes a sequence of Python `bytes` objects and returns encoded document blobs.
    #[pyo3(signature = (documents, threads = 1))]
    fn tokenize_batch<'py>(
        &self,
        py: Python<'py>,
        documents: &Bound<'py, PyAny>,
        threads: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let inputs = extract_byte_documents(documents)?;
        let encoded = py
            .detach(|| {
                self.inner
                    .tokenize_batch(&inputs, threads)
                    .and_then(|tokenized| {
                        tokenized
                            .iter()
                            .map(encode_tokenized)
                            .collect::<Result<Vec<_>, _>>()
                    })
            })
            .map_err(runtime_error)?;
        bytes_list(py, &encoded)
    }

    /// Decodes a sequence of encoded tokenized-document blobs back to exact bytes.
    #[staticmethod]
    fn decode_batch<'py>(
        py: Python<'py>,
        documents: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let encoded = extract_byte_documents(documents)?;
        let decoded = py
            .detach(|| {
                encoded
                    .iter()
                    .map(|value| decode_tokenized(value).map(|document| document.decode().to_vec()))
                    .collect::<Result<Vec<_>, _>>()
            })
            .map_err(runtime_error)?;
        bytes_list(py, &decoded)
    }

    /// Tokenizes and verifies byte-exact round-trip for every input document.
    #[pyo3(signature = (documents, threads = 1))]
    fn roundtrip_batch(
        &self,
        py: Python<'_>,
        documents: &Bound<'_, PyAny>,
        threads: usize,
    ) -> PyResult<bool> {
        let inputs = extract_byte_documents(documents)?;
        py.detach(|| {
            let tokenized = self
                .inner
                .tokenize_batch(&inputs, threads)
                .map_err(runtime_error)?;
            Ok(tokenized
                .iter()
                .zip(&inputs)
                .all(|(document, original)| document.decode() == original))
        })
    }
}

/// NedoFormer-specific byte-exact lattice and unified generation-vocabulary interface.
#[pyclass(module = "nedotokenizer._native", frozen)]
struct NedoFormerTokenizer {
    inner: NativeTokenizer<'static>,
    characters: Option<CharacterVocabulary>,
    generation: Option<NedoFormerVocabulary>,
}

#[pymethods]
impl NedoFormerTokenizer {
    /// Creates a `NedoFormer` tokenizer, optionally loading stable input/output vocabularies.
    #[new]
    #[allow(clippy::too_many_arguments)] // Python constructor mirrors explicit tokenizer config plus optional exact assets.
    #[pyo3(signature = (mode = "auto", max_sentence_tokens = 512, max_fallback_chars = 48, contextual_disambiguation = true, detect_unmarked_code = true, character_vocabulary = None, generation_vocabulary = None, compiled_analysis_table = None))]
    fn new(
        mode: &str,
        max_sentence_tokens: usize,
        max_fallback_chars: usize,
        contextual_disambiguation: bool,
        detect_unmarked_code: bool,
        character_vocabulary: Option<&Bound<'_, PyBytes>>,
        generation_vocabulary: Option<&Bound<'_, PyBytes>>,
        compiled_analysis_table: Option<&Bound<'_, PyBytes>>,
    ) -> PyResult<Self> {
        let inner = native_tokenizer(
            mode,
            max_sentence_tokens,
            max_fallback_chars,
            contextual_disambiguation,
            detect_unmarked_code,
        )?;
        let inner = if let Some(value) = compiled_analysis_table {
            let table = CompiledSurfaceAnalysisTable::from_bytes(value.as_bytes())
                .map_err(runtime_error)?;
            inner
                .with_verified_nedoformer_compiled_surface_analysis_table(table)
                .map_err(runtime_error)?
        } else {
            inner
        };
        let characters = character_vocabulary
            .map(|value| CharacterVocabulary::from_bytes(value.as_bytes()).map_err(runtime_error))
            .transpose()?;
        let generation = generation_vocabulary
            .map(|value| NedoFormerVocabulary::from_bytes(value.as_bytes()).map_err(runtime_error))
            .transpose()?;
        Ok(Self {
            inner,
            characters,
            generation,
        })
    }

    /// Builds one checksum-protected multi-candidate `NedoFormer` lattice blob.
    fn lattice<'py>(
        &self,
        py: Python<'py>,
        document: &Bound<'_, PyBytes>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let raw = document.as_bytes().to_vec();
        let bytes = py
            .detach(|| {
                self.inner
                    .nedoformer_lattice(raw)
                    .and_then(|lattice| lattice.to_bytes())
            })
            .map_err(runtime_error)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Returns JSON metadata for all full-lattice cut classes and model scores.
    #[staticmethod]
    fn lattice_metadata_json(lattice: &Bound<'_, PyBytes>) -> PyResult<String> {
        let lattice =
            NedoFormerLatticeDocument::from_bytes(lattice.as_bytes()).map_err(runtime_error)?;
        let units = lattice
            .units()
            .iter()
            .map(|unit| {
                json!({
                    "start": unit.selected_unit.span.start,
                    "end": unit.selected_unit.span.end,
                    "mode": token_mode_label(unit.selected_unit.mode),
                    "group_id": unit.selected_unit.group_id,
                    "candidates": unit.candidates.iter().map(|candidate| json!({
                        "cuts": candidate.cuts,
                        "status": token_status_label(candidate.status),
                        "analysis_count": candidate.analysis_count,
                        "conditional_log_score": candidate.conditional_log_score,
                        "selected": candidate.selected,
                    })).collect::<Vec<_>>(),
                })
            })
            .collect::<Vec<_>>();
        serde_json::to_string(&json!({
            "schema": NEDOFORMER_LATTICE_SCHEMA_VERSION,
            "raw_length": lattice.raw().len(),
            "units": units,
        }))
        .map_err(runtime_error)
    }

    /// Builds self-contained rich lattices for a document batch with worker-local cache reuse.
    #[pyo3(signature = (documents, threads = 1))]
    fn lattice_batch<'py>(
        &self,
        py: Python<'py>,
        documents: &Bound<'py, PyAny>,
        threads: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let inputs = extract_byte_documents(documents)?;
        let encoded = py
            .detach(|| {
                self.inner
                    .nedoformer_lattice_batch(&inputs, threads)?
                    .into_iter()
                    .map(|lattice| lattice.to_bytes())
                    .collect::<Result<Vec<_>, _>>()
            })
            .map_err(runtime_error)?;
        bytes_list(py, &encoded)
    }

    /// Builds the compact metadata-only large-corpus sidecar for one document.
    fn lattice_sidecar<'py>(
        &self,
        py: Python<'py>,
        document: &Bound<'_, PyBytes>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let raw = document.as_bytes().to_vec();
        let bytes = py
            .detach(|| {
                self.inner
                    .nedoformer_lattice(raw)
                    .and_then(|lattice| lattice.to_sidecar_bytes())
            })
            .map_err(runtime_error)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Builds compact sidecars for a batch using the native byte-weighted worker path.
    #[pyo3(signature = (documents, threads = 1))]
    fn lattice_sidecar_batch<'py>(
        &self,
        py: Python<'py>,
        documents: &Bound<'py, PyAny>,
        threads: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let inputs = extract_byte_documents(documents)?;
        let encoded = py
            .detach(|| self.inner.nedoformer_sidecar_batch(&inputs, threads))
            .map_err(runtime_error)?;
        bytes_list(py, &encoded)
    }

    /// Emits inner-Mamba character IDs plus recurrent reset and pooling metadata.
    #[pyo3(signature = (document, policy = "best", seed = 0, temperature = 1.0))]
    fn input_encoding<'py>(
        &self,
        py: Python<'py>,
        document: &Bound<'_, PyBytes>,
        policy: &str,
        seed: u64,
        temperature: f32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let characters = self.characters.as_ref().ok_or_else(|| {
            PyValueError::new_err("character_vocabulary must be loaded for input_encoding")
        })?;
        let raw = document.as_bytes().to_vec();
        let policy = parse_sampling_policy(policy, temperature)?;
        let encoding = py
            .detach(|| {
                self.inner
                    .nedoformer_lattice(raw)?
                    .sample_input_encoding(characters, policy, seed)
            })
            .map_err(runtime_error)?;
        input_encoding_dict(py, &encoding)
    }

    /// Samples a metadata-only sidecar and emits the exact same inner-Mamba contract.
    #[pyo3(signature = (document, sidecar, policy = "best", seed = 0, temperature = 1.0))]
    fn input_encoding_from_sidecar<'py>(
        &self,
        py: Python<'py>,
        document: &Bound<'_, PyBytes>,
        sidecar: &Bound<'_, PyBytes>,
        policy: &str,
        seed: u64,
        temperature: f32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let characters = self.characters.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "character_vocabulary must be loaded for input_encoding_from_sidecar",
            )
        })?;
        let raw = document.as_bytes().to_vec();
        let sidecar = sidecar.as_bytes().to_vec();
        let policy = parse_sampling_policy(policy, temperature)?;
        let encoding = py
            .detach(|| {
                NedoFormerLatticeSidecar::from_bytes(raw, &sidecar)?
                    .sample_input_encoding(characters, policy, seed)
            })
            .map_err(runtime_error)?;
        input_encoding_dict(py, &encoding)
    }

    /// Samples a stored lattice and returns the existing stable rich-document codec blob.
    #[pyo3(signature = (lattice, policy = "best", seed = 0, temperature = 1.0))]
    #[staticmethod]
    fn sample_lattice<'py>(
        py: Python<'py>,
        lattice: &Bound<'_, PyBytes>,
        policy: &str,
        seed: u64,
        temperature: f32,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = lattice.as_bytes().to_vec();
        let policy = parse_sampling_policy(policy, temperature)?;
        let encoded = py
            .detach(|| {
                let lattice = NedoFormerLatticeDocument::from_bytes(&bytes)?;
                let document = lattice.sample(policy, seed)?;
                encode_tokenized(&document)
            })
            .map_err(runtime_error)?;
        Ok(PyBytes::new(py, &encoded))
    }

    /// Trains deterministic input-character and unified generation vocabularies.
    ///
    /// Returns `(character_vocab_bytes, generation_vocab_bytes, contract_sha256_hex)`.
    #[pyo3(signature = (documents, max_chars = 500, max_roots = 16000, max_code_pieces = 4096))]
    fn train_assets<'py>(
        &self,
        py: Python<'py>,
        documents: &Bound<'_, PyAny>,
        max_chars: usize,
        max_roots: usize,
        max_code_pieces: usize,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, String)> {
        let inputs = extract_byte_documents(documents)?;
        let (characters, generation, fingerprint) = py
            .detach(|| {
                let mut selected = Vec::with_capacity(inputs.len());
                for raw in inputs {
                    selected.push(self.inner.nedoformer_lattice(raw)?.selected_document()?);
                }
                let characters = CharacterVocabulary::train(&selected, max_chars);
                let generation =
                    NedoFormerVocabulary::train(&selected, max_roots, max_chars, max_code_pieces)?;
                let fingerprint = self
                    .inner
                    .nedoformer_contract_fingerprint(&characters, &generation)?
                    .hex();
                Ok::<_, nedo_tokenizer::TokenizerError>((
                    characters.to_bytes()?,
                    generation.to_bytes()?,
                    fingerprint,
                ))
            })
            .map_err(runtime_error)?;
        Ok((
            PyBytes::new(py, &characters),
            PyBytes::new(py, &generation),
            fingerprint,
        ))
    }

    /// Encodes raw bytes into the unified `NedoFormer` generation-ID target.
    fn generation_ids(&self, py: Python<'_>, document: &Bound<'_, PyBytes>) -> PyResult<Vec<u16>> {
        let generation = self.generation.as_ref().ok_or_else(|| {
            PyValueError::new_err("generation_vocabulary must be loaded for generation_ids")
        })?;
        let raw = document.as_bytes().to_vec();
        py.detach(|| {
            let selected = self.inner.nedoformer_lattice(raw)?.selected_document()?;
            Ok::<_, nedo_tokenizer::TokenizerError>(generation.encode_document(&selected)?.ids)
        })
        .map_err(runtime_error)
    }

    /// Encodes a sampled rich-lattice path into the unified generation target.
    #[pyo3(signature = (lattice, policy = "best", seed = 0, temperature = 1.0))]
    fn generation_ids_from_lattice(
        &self,
        py: Python<'_>,
        lattice: &Bound<'_, PyBytes>,
        policy: &str,
        seed: u64,
        temperature: f32,
    ) -> PyResult<Vec<u16>> {
        let generation = self.generation.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "generation_vocabulary must be loaded for generation_ids_from_lattice",
            )
        })?;
        let bytes = lattice.as_bytes().to_vec();
        let policy = parse_sampling_policy(policy, temperature)?;
        py.detach(|| {
            let document = NedoFormerLatticeDocument::from_bytes(&bytes)?.sample(policy, seed)?;
            Ok::<_, nedo_tokenizer::TokenizerError>(generation.encode_document(&document)?.ids)
        })
        .map_err(runtime_error)
    }

    /// Decodes unified `NedoFormer` generation IDs to exact bytes.
    #[allow(clippy::needless_pass_by_value)] // PyO3 owns the extracted Python list here.
    fn generation_decode<'py>(
        &self,
        py: Python<'py>,
        ids: Vec<u16>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let generation = self.generation.as_ref().ok_or_else(|| {
            PyValueError::new_err("generation_vocabulary must be loaded for generation_decode")
        })?;
        let raw = generation.decode(&ids).map_err(runtime_error)?;
        Ok(PyBytes::new(py, &raw))
    }

    /// Returns the complete tokenizer-side fingerprint for loaded vocabularies.
    ///
    /// Decoder-side FSM/allomorph/pronunciation assets are intentionally outside this
    /// package and must extend this digest in the final model checkpoint contract.
    fn contract_fingerprint(&self) -> PyResult<String> {
        let characters = self.characters.as_ref().ok_or_else(|| {
            PyValueError::new_err("character_vocabulary must be loaded for contract_fingerprint")
        })?;
        let generation = self.generation.as_ref().ok_or_else(|| {
            PyValueError::new_err("generation_vocabulary must be loaded for contract_fingerprint")
        })?;
        self.inner
            .nedoformer_contract_fingerprint(characters, generation)
            .map(|fingerprint| fingerprint.hex())
            .map_err(runtime_error)
    }
}

/// Surface-piece tokenizer used by interactive inspectors and demos.
#[pyclass(module = "nedotokenizer._native", frozen)]
struct SurfaceTokenizer {
    inner: NativeTokenizer<'static>,
    vocabulary: SurfaceVocabulary,
    runtimes: Mutex<HashMap<usize, SurfaceRuntimeCache>>,
    sharded_runtimes: Mutex<HashMap<usize, ShardedSurfaceRuntimeCache>>,
}

#[pymethods]
impl SurfaceTokenizer {
    /// Creates a native tokenizer paired with one checksum-protected surface vocabulary.
    #[new]
    #[pyo3(signature = (vocabulary, mode = "auto", max_sentence_tokens = 512, max_fallback_chars = 48, contextual_disambiguation = true, detect_unmarked_code = true, analysis_table = None))]
    fn new(
        vocabulary: &Bound<'_, PyBytes>,
        mode: &str,
        max_sentence_tokens: usize,
        max_fallback_chars: usize,
        contextual_disambiguation: bool,
        detect_unmarked_code: bool,
        analysis_table: Option<&Bound<'_, PyBytes>>,
    ) -> PyResult<Self> {
        let mut inner = native_tokenizer(
            mode,
            max_sentence_tokens,
            max_fallback_chars,
            contextual_disambiguation,
            detect_unmarked_code,
        )?;
        if let Some(table_bytes) = analysis_table {
            let table = CompiledSurfaceAnalysisTable::from_bytes(table_bytes.as_bytes())
                .map_err(runtime_error)?;
            inner = inner
                .with_verified_compiled_surface_analysis_table(table)
                .map_err(runtime_error)?;
        }
        let vocabulary =
            SurfaceVocabulary::from_bytes(vocabulary.as_bytes()).map_err(runtime_error)?;
        Ok(Self {
            inner,
            vocabulary,
            runtimes: Mutex::new(HashMap::new()),
            sharded_runtimes: Mutex::new(HashMap::new()),
        })
    }

    /// Returns a stable JSON description of final surface IDs, byte spans and rich units.
    fn inspect_json(&self, py: Python<'_>, document: &Bound<'_, PyBytes>) -> PyResult<String> {
        let raw = document.as_bytes().to_vec();
        let payload = py
            .detach(|| inspect_surface(&self.inner, &self.vocabulary, &raw))
            .map_err(runtime_error)?;
        serde_json::to_string(&payload).map_err(runtime_error)
    }

    /// Encodes one document directly to native/content surface IDs without JSON, codec serialization, BOS, or EOS.
    fn encode_ids(&self, py: Python<'_>, document: &Bound<'_, PyBytes>) -> PyResult<Vec<u16>> {
        let raw = document.as_bytes().to_vec();
        py.detach(|| {
            let inputs = [raw];
            let newline_flags = [false];
            let mut runtimes = self
                .runtimes
                .lock()
                .map_err(|_| runtime_error("surface runtime lock is poisoned"))?;
            if let std::collections::hash_map::Entry::Vacant(entry) = runtimes.entry(1) {
                entry.insert(
                    self.inner
                        .surface_runtime_cache(1, SurfaceEncoderOptions::one_pass(true))
                        .map_err(runtime_error)?,
                );
            }
            let runtime = runtimes
                .get_mut(&1)
                .ok_or_else(|| runtime_error("surface runtime was not created"))?;
            let batch = self
                .inner
                .encode_surface_batch_with_runtime(
                    &inputs,
                    &newline_flags,
                    &self.vocabulary,
                    runtime,
                )
                .map_err(runtime_error)?;
            drop(runtimes);
            content_ids_for_document(&batch.ids, &batch.lengths)
        })
    }

    /// Encodes a batch directly to native/content surface IDs using the optimized flat batch encoder.
    fn encode_ids_batch(
        &self,
        py: Python<'_>,
        documents: &Bound<'_, PyAny>,
        threads: usize,
    ) -> PyResult<Vec<Vec<u16>>> {
        if threads == 0 {
            return Err(PyValueError::new_err("threads must be positive"));
        }
        let inputs = extract_byte_documents(documents)?;
        py.detach(|| {
            let newline_flags = vec![false; inputs.len()];
            let mut runtimes = self
                .sharded_runtimes
                .lock()
                .map_err(|_| runtime_error("sharded surface runtime lock is poisoned"))?;
            if let std::collections::hash_map::Entry::Vacant(entry) = runtimes.entry(threads) {
                entry.insert(
                    self.inner
                        .sharded_surface_runtime_cache(
                            threads,
                            SurfaceEncoderOptions::one_pass_compact(true),
                        )
                        .map_err(runtime_error)?,
                );
            }
            let runtime = runtimes
                .get_mut(&threads)
                .ok_or_else(|| runtime_error("sharded surface runtime was not created"))?;
            let batch = self
                .inner
                .encode_surface_batch_with_sharded_runtime(
                    &inputs,
                    &newline_flags,
                    &self.vocabulary,
                    runtime,
                )
                .map_err(runtime_error)?;
            drop(runtimes);
            if batch.document_offsets.len() != inputs.len().saturating_add(1)
                || batch.ids.len() != batch.lengths.len()
            {
                return Err(runtime_error("surface batch metadata is inconsistent"));
            }
            let mut rows = Vec::with_capacity(inputs.len());
            for offsets in batch.document_offsets.windows(2) {
                let start = usize::try_from(offsets[0])
                    .map_err(|_| runtime_error("surface batch start offset overflow"))?;
                let end = usize::try_from(offsets[1])
                    .map_err(|_| runtime_error("surface batch end offset overflow"))?;
                if end <= start.saturating_add(1)
                    || end > batch.ids.len()
                    || batch.lengths[start] != 0
                    || batch.lengths[end - 1] != 0
                {
                    return Err(runtime_error(
                        "surface batch document boundaries are invalid",
                    ));
                }
                rows.push(batch.ids[start + 1..end - 1].to_vec());
            }
            Ok(rows)
        })
    }

    /// Clears persistent worker caches retained by prior encode calls.
    fn clear_runtime_caches(&self) -> PyResult<()> {
        let mut runtimes = self
            .runtimes
            .lock()
            .map_err(|_| runtime_error("surface runtime lock is poisoned"))?;
        for runtime in runtimes.values_mut() {
            runtime.clear();
        }
        drop(runtimes);
        let mut sharded = self
            .sharded_runtimes
            .lock()
            .map_err(|_| runtime_error("sharded surface runtime lock is poisoned"))?;
        for runtime in sharded.values_mut() {
            runtime.clear();
        }
        drop(sharded);
        Ok(())
    }

    /// Returns aggregate cache counters for one retained thread-count runtime.
    fn runtime_cache_stats<'py>(
        &self,
        py: Python<'py>,
        threads: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        let sharded_stats = {
            let sharded = self
                .sharded_runtimes
                .lock()
                .map_err(|_| runtime_error("sharded surface runtime lock is poisoned"))?;
            sharded.get(&threads).map(|runtime| {
                let stats = runtime.cache_stats();
                (
                    runtime.shard_count(),
                    stats.hits,
                    stats.misses,
                    stats.saturated_misses,
                    stats.entries,
                    stats.approximate_bytes,
                )
            })
        };
        let stats = if let Some(stats) = sharded_stats {
            stats
        } else {
            let runtimes = self
                .runtimes
                .lock()
                .map_err(|_| runtime_error("surface runtime lock is poisoned"))?;
            runtimes
                .get(&threads)
                .map_or((threads, 0, 0, 0, 0, 0), |runtime| {
                    let stats = runtime.cache_stats();
                    (
                        runtime.threads(),
                        stats.hits,
                        stats.misses,
                        stats.saturated_misses,
                        stats.entries,
                        stats.approximate_bytes,
                    )
                })
        };
        result.set_item("threads", stats.0)?;
        result.set_item("hits", stats.1)?;
        result.set_item("misses", stats.2)?;
        result.set_item("saturated_misses", stats.3)?;
        result.set_item("entries", stats.4)?;
        result.set_item("approximate_bytes", stats.5)?;
        Ok(result)
    }

    /// Decodes final surface token IDs to exact bytes.
    fn decode_ids<'py>(&self, py: Python<'py>, ids: Vec<u16>) -> PyResult<Bound<'py, PyBytes>> {
        let ids = ids.into_boxed_slice();
        let decoded = py
            .detach(move || self.vocabulary.decode_ids(&ids))
            .map_err(runtime_error)?;
        Ok(PyBytes::new(py, &decoded))
    }

    /// Total embedding vocabulary size, including specials and byte fallback.
    fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }
}

fn content_ids_for_document(ids: &[u16], lengths: &[u8]) -> PyResult<Vec<u16>> {
    if ids.len() < 2
        || ids.len() != lengths.len()
        || lengths.first() != Some(&0)
        || lengths.last() != Some(&0)
    {
        return Err(runtime_error("surface document boundaries are invalid"));
    }
    Ok(ids[1..ids.len() - 1].to_vec())
}

#[pyfunction]
fn asset_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("schema_version", TOKENIZER_SCHEMA_VERSION)?;
    result.set_item("morphology_sha256", MORPHOLOGY_SHA256)?;
    result.set_item("model_sha256", MODEL_SHA256)?;
    result.set_item("runtime", "rust-native")?;
    result.set_item("python_hot_path", false)?;
    result.set_item("compiled_surface_table_supported", true)?;
    result.set_item("nedoformer_supported", true)?;
    result.set_item(
        "nedoformer_contract_version",
        NEDOFORMER_TOKENIZER_CONTRACT_VERSION,
    )?;
    result.set_item(
        "nedoformer_lattice_schema_version",
        NEDOFORMER_LATTICE_SCHEMA_VERSION,
    )?;
    result.set_item(
        "nedoformer_input_encoding_version",
        NEDOFORMER_INPUT_ENCODING_VERSION,
    )?;
    result.set_item("nedoformer_sidecar_supported", true)?;
    result.set_item(
        "nedoformer_sidecar_schema_version",
        NEDOFORMER_SIDECAR_SCHEMA_VERSION,
    )?;
    Ok(result)
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Tokenizer>()?;
    module.add_class::<NedoFormerTokenizer>()?;
    module.add_class::<SurfaceTokenizer>()?;
    module.add_function(wrap_pyfunction!(asset_info, module)?)?;
    Ok(())
}

fn native_tokenizer(
    mode: &str,
    max_sentence_tokens: usize,
    max_fallback_chars: usize,
    contextual_disambiguation: bool,
    detect_unmarked_code: bool,
) -> PyResult<NativeTokenizer<'static>> {
    let mode = parse_mode(mode)?;
    NativeTokenizer::embedded(TokenizerConfig {
        mode,
        max_sentence_tokens,
        max_fallback_chars,
        contextual_disambiguation,
        detect_unmarked_code,
    })
    .map_err(runtime_error)
}

fn inspect_surface(
    tokenizer: &NativeTokenizer<'_>,
    vocabulary: &SurfaceVocabulary,
    raw: &[u8],
) -> Result<Value, nedo_tokenizer::TokenizerError> {
    let document = tokenizer.tokenize(raw.to_vec())?;
    let encoded = vocabulary.encode_document(&document, false)?;
    let decoded = vocabulary.decode_ids(&encoded.ids)?;
    let units = document
        .units()
        .iter()
        .enumerate()
        .map(|(index, unit)| unit_json(index, unit, raw))
        .collect::<Result<Vec<_>, _>>()?;
    let tokens = surface_tokens_json(&document, &encoded.ids, &encoded.lengths, raw)?;
    let summary = surface_summary(
        &document,
        &encoded.ids,
        &encoded.lengths,
        raw,
        vocabulary,
        decoded == raw,
    );
    Ok(json!({"schema": 1, "tokens": tokens, "units": units, "summary": summary}))
}

fn surface_tokens_json(
    document: &nedo_tokenizer::TokenizedDocument,
    ids: &[u16],
    lengths: &[u8],
    raw: &[u8],
) -> Result<Vec<Value>, nedo_tokenizer::TokenizerError> {
    let mut byte_cursor = 0_usize;
    let mut unit_cursor = 0_usize;
    let mut tokens = Vec::with_capacity(ids.len());
    for (index, (&raw_id, &length)) in ids.iter().zip(lengths).enumerate() {
        let id = u32::from(raw_id);
        let start = byte_cursor;
        let end = start.checked_add(usize::from(length)).ok_or(
            nedo_tokenizer::TokenizerError::LengthOverflow("surface inspector token end"),
        )?;
        let token_bytes =
            raw.get(start..end)
                .ok_or(nedo_tokenizer::TokenizerError::InvalidTrainingEncoding(
                    "surface inspector token span exceeds source bytes",
                ))?;
        byte_cursor = end;
        unit_cursor = advance_unit_cursor(document, unit_cursor, start)?;
        let unit_index = (length > 0)
            .then_some(unit_cursor)
            .filter(|value| document.units().get(*value).is_some());
        let unit = unit_index.and_then(|value| document.units().get(value));
        let text = core::str::from_utf8(token_bytes).ok().map(str::to_owned);
        tokens.push(json!({
            "index": index,
            "id": id,
            "kind": surface_id_kind(id),
            "length": length,
            "start": start,
            "end": end,
            "text": text,
            "display": display_bytes(token_bytes),
            "hex": hex_bytes(token_bytes),
            "unit_index": unit_index,
            "unit_kind": unit.map(|value| format!("{:?}", value.kind)),
            "mode": unit.map(|value| format!("{:?}", value.mode)),
            "status": unit.map(|value| format!("{:?}", value.status)),
        }));
    }
    if byte_cursor != raw.len() {
        return Err(nedo_tokenizer::TokenizerError::InvalidTrainingEncoding(
            "surface inspector byte accounting differs from source",
        ));
    }
    Ok(tokens)
}

fn advance_unit_cursor(
    document: &nedo_tokenizer::TokenizedDocument,
    mut cursor: usize,
    byte_start: usize,
) -> Result<usize, nedo_tokenizer::TokenizerError> {
    let byte_start = u64::try_from(byte_start).map_err(|_| {
        nedo_tokenizer::TokenizerError::LengthOverflow("surface inspector unit lookup")
    })?;
    while let Some(unit) = document.units().get(cursor) {
        if unit.span.end > byte_start {
            break;
        }
        cursor += 1;
    }
    Ok(cursor)
}

fn surface_summary(
    document: &nedo_tokenizer::TokenizedDocument,
    ids: &[u16],
    lengths: &[u8],
    raw: &[u8],
    vocabulary: &SurfaceVocabulary,
    roundtrip: bool,
) -> Value {
    let lexical_words = document
        .units()
        .iter()
        .filter(|unit| format!("{:?}", unit.kind) == "Word")
        .count();
    let status_count = |status| {
        document
            .units()
            .iter()
            .filter(|unit| unit.status == status)
            .count()
    };
    let content_tokens = lengths.iter().filter(|length| **length > 0).count();
    let byte_fallback_tokens = ids
        .iter()
        .zip(lengths)
        .filter(|(id, length)| {
            **length > 0
                && (SURFACE_BYTE_BASE_ID..SURFACE_ENTRY_BASE_ID).contains(&u32::from(**id))
        })
        .count();
    let byte_fallback_bytes = ids
        .iter()
        .zip(lengths)
        .filter(|(id, _)| {
            (SURFACE_BYTE_BASE_ID..SURFACE_ENTRY_BASE_ID).contains(&u32::from(**id))
        })
        .map(|(_, length)| usize::from(*length))
        .sum::<usize>();
    let learned_tokens = ids
        .iter()
        .zip(lengths)
        .filter(|(id, length)| **length > 0 && u32::from(**id) >= SURFACE_ENTRY_BASE_ID)
        .count();
    let characters = core::str::from_utf8(raw).map_or(0, |text| text.chars().count());
    json!({
        "bytes": raw.len(),
        "characters": characters,
        "words": lexical_words,
        "content_tokens": content_tokens,
        "learned_tokens": learned_tokens,
        "byte_fallback_tokens": byte_fallback_tokens,
        "byte_fallback_bytes": byte_fallback_bytes,
        "special_tokens": lengths.len().saturating_sub(content_tokens),
        "all_tokens": lengths.len(),
        "units": document.units().len(),
        "morphological_units": status_count(TokenStatus::Morphological),
        "code_units": status_count(TokenStatus::Code),
        "unknown_units": status_count(TokenStatus::Unknown),
        "vocabulary_size": vocabulary.len(),
        "roundtrip": roundtrip,
    })
}

fn unit_json(
    index: usize,
    unit: &nedo_tokenizer::TokenizedUnit,
    raw: &[u8],
) -> Result<Value, nedo_tokenizer::TokenizerError> {
    let start = usize::try_from(unit.span.start)
        .map_err(|_| nedo_tokenizer::TokenizerError::LengthOverflow("inspector unit start"))?;
    let end = usize::try_from(unit.span.end)
        .map_err(|_| nedo_tokenizer::TokenizerError::LengthOverflow("inspector unit end"))?;
    let surface = raw
        .get(start..end)
        .ok_or(nedo_tokenizer::TokenizerError::UnitOutsideDocument)?;
    let analysis = unit.analysis.as_ref().map(|value| {
        let morphemes = value
            .morphemes
            .iter()
            .map(|morpheme| {
                json!({
                    "id": morpheme.id,
                    "surface": morpheme.surface,
                    "start": morpheme.span.start,
                    "end": morpheme.span.end,
                    "derivational": morpheme.derivational,
                })
            })
            .collect::<Vec<_>>();
        json!({
            "canonical": value.canonical,
            "dictionary_id": value.dictionary_id,
            "lemma": value.lemma,
            "primary_pos": value.primary_pos,
            "secondary_pos": value.secondary_pos,
            "morphemes": morphemes,
        })
    });
    Ok(json!({
        "index": index,
        "start": start,
        "end": end,
        "surface": core::str::from_utf8(surface).ok(),
        "display": display_bytes(surface),
        "hex": hex_bytes(surface),
        "kind": format!("{:?}", unit.kind),
        "mode": format!("{:?}", unit.mode),
        "status": format!("{:?}", unit.status),
        "group_id": unit.group_id,
        "cuts": unit.cuts,
        "analysis": analysis,
    }))
}

const fn surface_id_kind(id: u32) -> &'static str {
    match id {
        SURFACE_PAD_ID => "pad",
        SURFACE_BOS_ID => "bos",
        SURFACE_EOS_ID => "eos",
        SURFACE_BYTE_BASE_ID..SURFACE_ENTRY_BASE_ID => "byte",
        _ => "learned",
    }
}

fn display_bytes(value: &[u8]) -> String {
    core::str::from_utf8(value).map_or_else(
        |_| {
            value.iter().fold(String::new(), |mut output, byte| {
                let _ = write!(output, "\\x{byte:02X}");
                output
            })
        },
        |text| {
            text.chars().fold(String::new(), |mut output, character| {
                match character {
                    ' ' => output.push('␠'),
                    '\t' => output.push('⇥'),
                    '\r' => output.push('␍'),
                    '\n' => output.push('↵'),
                    value if value.is_control() => {
                        let _ = write!(output, "\\u{{{:X}}}", u32::from(value));
                    }
                    value => output.push(value),
                }
                output
            })
        },
    )
}

fn hex_bytes(value: &[u8]) -> String {
    value
        .iter()
        .enumerate()
        .fold(String::new(), |mut output, (index, byte)| {
            if index > 0 {
                output.push(' ');
            }
            let _ = write!(output, "{byte:02X}");
            output
        })
}

const fn token_mode_label(mode: TokenMode) -> &'static str {
    match mode {
        TokenMode::Turkish => "turkish",
        TokenMode::Code => "code",
        TokenMode::Opaque => "opaque",
    }
}

const fn token_status_label(status: TokenStatus) -> &'static str {
    match status {
        TokenStatus::Structural => "structural",
        TokenStatus::Morphological => "morphological",
        TokenStatus::Unknown => "unknown",
        TokenStatus::Code => "code",
        TokenStatus::Opaque => "opaque",
    }
}

fn input_encoding_dict<'py>(
    py: Python<'py>,
    encoding: &NedoFormerInputEncoding,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("ids", &encoding.ids)?;
    result.set_item("segment_offsets", &encoding.segment_offsets)?;
    result.set_item("pooled_segments", &encoding.pooled_segments)?;
    result.set_item(
        "pool_spans",
        encoding
            .pool_spans
            .iter()
            .map(|span| (span.start, span.end))
            .collect::<Vec<_>>(),
    )?;
    result.set_item(
        "pool_modes",
        encoding
            .pool_modes
            .iter()
            .map(|mode| token_mode_label(*mode))
            .collect::<Vec<_>>(),
    )?;
    result.set_item("pool_group_ids", &encoding.pool_group_ids)?;
    Ok(result)
}

fn parse_sampling_policy(value: &str, temperature: f32) -> PyResult<NedoFormerSamplingPolicy> {
    match value {
        "best" => Ok(NedoFormerSamplingPolicy::Best),
        "uniform" => Ok(NedoFormerSamplingPolicy::Uniform),
        "context" | "context_weighted" => {
            if !temperature.is_finite() || temperature <= 0.0 {
                return Err(PyValueError::new_err(
                    "temperature must be positive and finite",
                ));
            }
            Ok(NedoFormerSamplingPolicy::ContextWeighted { temperature })
        }
        _ => Err(PyValueError::new_err(
            "policy must be one of: best, uniform, context_weighted",
        )),
    }
}

fn parse_mode(value: &str) -> PyResult<TokenizerMode> {
    match value {
        "auto" => Ok(TokenizerMode::Auto),
        "turkish" => Ok(TokenizerMode::Turkish),
        "code" => Ok(TokenizerMode::Code),
        _ => Err(PyValueError::new_err(
            "mode must be one of: auto, turkish, code",
        )),
    }
}

fn extract_byte_documents(value: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<u8>>> {
    let sequence = value
        .cast::<PyList>()
        .map_err(|_| PyTypeError::new_err("documents must be a list of bytes objects"))?;
    sequence
        .iter()
        .enumerate()
        .map(|(index, item)| {
            item.cast::<PyBytes>()
                .map(|bytes| bytes.as_bytes().to_vec())
                .map_err(|_| PyTypeError::new_err(format!("documents[{index}] is not bytes")))
        })
        .collect()
}

fn bytes_list<'py>(py: Python<'py>, values: &[Vec<u8>]) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, values.iter().map(|value| PyBytes::new(py, value)))
}

fn runtime_error(error: impl core::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}
