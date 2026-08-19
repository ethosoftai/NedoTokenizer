//! Allocation-conscious native morphology runtime over the validated binary bundle.

use std::collections::HashSet;
use std::ops::Range;
use std::sync::Arc;

use super::{
    checked_add, checked_mul, fixed_record, invalid, read_byte, read_program_u16, read_program_u32,
    read_u16, read_u32, require_index, BinaryBundleView, BinaryError, BinarySummary, Section,
    StringTable, DICTIONARY_RECORD_SIZE, EDGE_RECORD_SIZE, MORPHEME_RECORD_SIZE, NONE_U16,
    NONE_U32, PRIMARY_POS_SHORT, SECONDARY_POS_SHORT, STATE_RECORD_SIZE, STEM_RECORD_SIZE,
    TEMPLATE_RECORD_SIZE,
};

const LAST_LETTER_VOWEL: u32 = 1 << 0;
const LAST_LETTER_CONSONANT: u32 = 1 << 1;
const LAST_VOWEL_FRONTAL: u32 = 1 << 2;
const LAST_VOWEL_BACK: u32 = 1 << 3;
const LAST_VOWEL_ROUNDED: u32 = 1 << 4;
const LAST_VOWEL_UNROUNDED: u32 = 1 << 5;
const LAST_LETTER_VOICELESS: u32 = 1 << 6;
const LAST_LETTER_VOICED: u32 = 1 << 7;
const LAST_LETTER_VOICELESS_STOP: u32 = 1 << 8;
const FIRST_LETTER_VOWEL: u32 = 1 << 9;
const FIRST_LETTER_CONSONANT: u32 = 1 << 10;
const HAS_NO_VOWEL: u32 = 1 << 11;
const EXPECTS_VOWEL: u32 = 1 << 12;
const EXPECTS_CONSONANT: u32 = 1 << 13;
const CANNOT_TERMINATE: u32 = 1 << 17;
const ROOT_ATTRIBUTE_DUMMY: u32 = 1 << 20;

/// Hard limits protecting native graph search from malformed or unexpectedly explosive input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AnalysisLimits {
    /// Maximum number of paths alive in one breadth-first search layer.
    pub max_active_paths: usize,
    /// Maximum number of immutable transition nodes stored in the path arena.
    pub max_path_nodes: usize,
    /// Maximum accepted analyses returned for one input.
    pub max_results: usize,
}

impl Default for AnalysisLimits {
    fn default() -> Self {
        Self {
            max_active_paths: 250_000,
            max_path_nodes: 2_000_000,
            max_results: 100_000,
        }
    }
}

/// One morpheme and its realized surface in a native analysis.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeMorpheme {
    /// Stable Zemberek morpheme ID.
    pub id: String,
    /// Human-readable morpheme name.
    pub name: String,
    /// Surface consumed by this morpheme. Epsilon transitions use an empty string.
    pub surface: String,
    /// Whether this morpheme starts a derivational group.
    pub derivational: bool,
    /// Whether this morpheme belongs to the informal model.
    pub informal: bool,
    /// Optional primary POS short form.
    pub pos: Option<String>,
    /// Optional mapped formal morpheme ID.
    pub mapped_id: Option<String>,
}

/// One accepted native morphological analysis.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeAnalysis {
    /// Canonical parity key used by the pinned Java exporter.
    pub canonical: String,
    /// Stable dictionary item ID.
    pub dictionary_id: String,
    /// Root dictionary lemma.
    pub lemma: String,
    /// Root dictionary primary POS short form.
    pub primary_pos: String,
    /// Root dictionary secondary POS short form.
    pub secondary_pos: String,
    /// Surface form analyzed after apostrophe removal, matching Zemberek `SingleAnalysis`.
    pub surface_form: String,
    /// Stem surface selected for this path.
    pub stem: String,
    /// Concatenated non-stem surface.
    pub ending: String,
    /// Ordered root and suffix morphemes.
    pub morphemes: Vec<NativeMorpheme>,
}

/// One borrowed stem transition from the native binary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeStem<'a> {
    /// Stem surface.
    pub surface: &'a str,
    /// Dictionary table index.
    pub dictionary_index: u32,
    /// Target state table index.
    pub state_index: u32,
    /// Initial phonetic attribute bitset.
    pub phonetic_bits: u32,
}

/// Exact-surface stem iterator. Records borrow directly from the native binary.
pub struct StemMatches<'m, 'a> {
    morphology: &'m NativeMorphology<'a>,
    next: usize,
    end: usize,
    failed: bool,
}

impl<'a> Iterator for StemMatches<'_, 'a> {
    type Item = Result<NativeStem<'a>, BinaryError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed || self.next >= self.end {
            return None;
        }
        let index = self.next;
        self.next += 1;
        let result = self.morphology.stem_at(index).map(StemData::public_view);
        if result.is_err() {
            self.failed = true;
        }
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end.saturating_sub(self.next);
        (remaining, Some(remaining))
    }
}

/// Validated zero-copy runtime over one native morphology binary.
#[derive(Clone)]
pub struct NativeMorphology<'a> {
    view: BinaryBundleView<'a>,
    strings: StringTable<'a>,
    stems: Arc<[StemData<'a>]>,
    dictionaries: Arc<[DictionaryData<'a>]>,
    morphemes: Arc<[MorphemeData<'a>]>,
    templates: Arc<[TemplateData]>,
    states: Arc<[StateData<'a>]>,
    edges: Arc<[EdgeData]>,
    stem_prefix_index: Arc<StemPrefixIndex>,
}

/// Reusable dictionary-to-stem index for high-throughput native word generation.
pub struct NativeGenerator<'m, 'a> {
    morphology: &'m NativeMorphology<'a>,
    offsets: Vec<usize>,
    stem_indices: Vec<usize>,
}

impl<'m, 'a> NativeGenerator<'m, 'a> {
    fn new(morphology: &'m NativeMorphology<'a>) -> Result<Self, BinaryError> {
        let dictionary_count = morphology.view.header.counts[2] as usize;
        let stem_count = morphology.view.header.counts[3] as usize;
        let mut counts = vec![0_usize; dictionary_count + 1];
        for index in 0..stem_count {
            let dictionary = morphology.stem_at(index)?.dictionary_index as usize;
            let slot = counts
                .get_mut(dictionary + 1)
                .ok_or_else(|| invalid("generation index dictionary is out of bounds"))?;
            *slot = slot
                .checked_add(1)
                .ok_or_else(|| invalid("generation index count overflow"))?;
        }
        for index in 1..counts.len() {
            counts[index] = counts[index]
                .checked_add(counts[index - 1])
                .ok_or_else(|| invalid("generation index prefix overflow"))?;
        }
        let mut positions = counts[..dictionary_count].to_vec();
        let mut stem_indices = vec![0_usize; stem_count];
        for index in 0..stem_count {
            let dictionary = morphology.stem_at(index)?.dictionary_index as usize;
            let position = positions
                .get_mut(dictionary)
                .ok_or_else(|| invalid("generation index position is out of bounds"))?;
            stem_indices[*position] = index;
            *position = position
                .checked_add(1)
                .ok_or_else(|| invalid("generation index position overflow"))?;
        }
        Ok(Self {
            morphology,
            offsets: counts,
            stem_indices,
        })
    }

    /// Generates all graph-valid forms using the reusable dictionary-to-stem index.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown IDs, corrupt data, or exceeded graph-search limits.
    pub fn generate(
        &self,
        dictionary_id: &str,
        morpheme_ids: &[&str],
    ) -> Result<Vec<NativeAnalysis>, BinaryError> {
        self.generate_with_limits(dictionary_id, morpheme_ids, AnalysisLimits::default())
    }

    /// Generates forms using the reusable index and explicit limits.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown IDs, corrupt data, or exceeded graph-search limits.
    pub fn generate_with_limits(
        &self,
        dictionary_id: &str,
        morpheme_ids: &[&str],
        limits: AnalysisLimits,
    ) -> Result<Vec<NativeAnalysis>, BinaryError> {
        let dictionary = self
            .morphology
            .dictionary_index_by_id(dictionary_id)?
            .ok_or_else(|| invalid(format!("unknown generation dictionary ID {dictionary_id}")))?;
        let dictionary_index = dictionary as usize;
        let start = *self
            .offsets
            .get(dictionary_index)
            .ok_or_else(|| invalid("generation index start is out of bounds"))?;
        let end = *self
            .offsets
            .get(dictionary_index + 1)
            .ok_or_else(|| invalid("generation index end is out of bounds"))?;
        self.morphology.generate_with_candidate_stems(
            dictionary,
            morpheme_ids,
            limits,
            &self.stem_indices[start..end],
        )
    }
}

impl<'a> NativeMorphology<'a> {
    /// Validates a native binary and creates a borrowed runtime view.
    ///
    /// # Errors
    ///
    /// Returns an error for any binary schema, checksum, reference, or bytecode failure.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, BinaryError> {
        let view = BinaryBundleView::parse(bytes)?;
        let strings = StringTable::parse(view)?;
        let stems = decode_stem_table(view, &strings)?;
        let dictionaries = decode_dictionary_table(view, &strings)?;
        let morphemes = decode_morpheme_table(view, &strings)?;
        let templates = decode_template_table(view)?;
        let states = decode_state_table(view, &strings)?;
        let edges = decode_edge_table(view, &strings)?;
        let mut morphology = Self {
            view,
            strings,
            stems: stems.into(),
            dictionaries: dictionaries.into(),
            morphemes: morphemes.into(),
            templates: templates.into(),
            states: states.into(),
            edges: edges.into(),
            stem_prefix_index: Arc::new(StemPrefixIndex::empty()),
        };
        morphology.stem_prefix_index = Arc::new(StemPrefixIndex::build(&morphology)?);
        Ok(morphology)
    }

    /// Returns the already validated binary summary.
    #[must_use]
    pub const fn summary(&self) -> BinarySummary {
        self.view.summary()
    }

    /// Builds a reusable dictionary-to-stem index for high-throughput generation.
    ///
    /// # Errors
    ///
    /// Returns an error if validated stem records cannot be indexed safely.
    pub fn generator(&self) -> Result<NativeGenerator<'_, 'a>, BinaryError> {
        NativeGenerator::new(self)
    }

    /// Returns exact stem transitions for one surface without copying binary records.
    ///
    /// # Errors
    ///
    /// Returns an error only if an internal validated record cannot be read.
    pub fn stem_matches<'m>(&'m self, surface: &str) -> Result<StemMatches<'m, 'a>, BinaryError> {
        let range = match self.strings.find(surface)? {
            Some(surface_id) => self.stem_range(surface_id)?,
            None => 0..0,
        };
        Ok(StemMatches {
            morphology: self,
            next: range.start,
            end: range.end,
            failed: false,
        })
    }

    /// Normalizes and analyzes one original token, including Zemberek-compatible
    /// runtime numeral and apostrophized unknown-proper handling.
    ///
    /// # Errors
    ///
    /// Returns an error for corrupt internal data or if a graph-search limit is exceeded.
    pub fn analyze_token(&self, input: &str) -> Result<Vec<NativeAnalysis>, BinaryError> {
        self.analyze_token_with_limits(input, AnalysisLimits::default())
    }

    /// Normalizes and analyzes one original token with explicit safety limits.
    ///
    /// # Errors
    ///
    /// Returns an error for corrupt internal data or if a graph-search limit is exceeded.
    pub fn analyze_token_with_limits(
        &self,
        input: &str,
        limits: AnalysisLimits,
    ) -> Result<Vec<NativeAnalysis>, BinaryError> {
        validate_limits(limits)?;
        let normalized = normalize_for_analysis(input);
        let analyses = self.analyze_with_limits(&normalized, limits)?;
        if !analyses.is_empty() {
            return Ok(analyses);
        }
        let runtime_original = normalize_apostrophes(input);
        if is_url_token(&runtime_original) {
            let url = self.analyze_runtime_url(&runtime_original, limits)?;
            if !url.is_empty() {
                return Ok(url);
            }
        }
        if is_roman_numeral_token(&runtime_original) {
            let roman = self.analyze_runtime_roman_numeral(&runtime_original, limits)?;
            if !roman.is_empty() {
                return Ok(roman);
            }
        }
        if is_dotted_abbreviation_token(&runtime_original) {
            let abbreviation = self.analyze_runtime_abbreviation(&runtime_original, limits)?;
            if !abbreviation.is_empty() {
                return Ok(abbreviation);
            }
        }
        let runtime_input = turkish_lower(&runtime_original);
        if runtime_input.chars().any(|value| value.is_ascii_digit()) {
            let numeral = self.analyze_runtime_numeral(&runtime_input, limits)?;
            if !numeral.is_empty() {
                return Ok(numeral);
            }
        }
        if apostrophe_range(&runtime_input).is_some() {
            return self.analyze_runtime_proper(&runtime_input, limits);
        }
        Ok(Vec::new())
    }

    /// Analyzes one already-normalized input using default safety limits.
    ///
    /// # Errors
    ///
    /// Returns an error for corrupt internal data or if a safety limit is exceeded.
    pub fn analyze(&self, input: &str) -> Result<Vec<NativeAnalysis>, BinaryError> {
        self.analyze_with_limits(input, AnalysisLimits::default())
    }

    /// Analyzes one already-normalized input with explicit graph-search limits.
    ///
    /// # Errors
    ///
    /// Returns an error for corrupt internal data or if a supplied safety limit is exceeded.
    pub fn analyze_with_limits(
        &self,
        input: &str,
        limits: AnalysisLimits,
    ) -> Result<Vec<NativeAnalysis>, BinaryError> {
        validate_limits(limits)?;
        if let Some((apostrophe_start, apostrophe_end)) = apostrophe_range(input) {
            if apostrophe_start == 0 || apostrophe_end == input.len() {
                return Ok(Vec::new());
            }
            let stem = &input[..apostrophe_start];
            let mut without_apostrophe = String::with_capacity(input.len());
            without_apostrophe.push_str(stem);
            without_apostrophe.push_str(&input[apostrophe_end..]);
            let mut analyses = self.analyze_plain(&without_apostrophe, limits)?;
            analyses.retain(|analysis| {
                analysis.primary_pos == "Noun"
                    && (analysis.stem == stem
                        || analysis
                            .morphemes
                            .iter()
                            .any(|morpheme| morpheme.id == "P3sg"))
            });
            return Ok(analyses);
        }
        self.analyze_plain(input, limits)
    }

    fn analyze_plain(
        &self,
        input: &str,
        limits: AnalysisLimits,
    ) -> Result<Vec<NativeAnalysis>, BinaryError> {
        let (mut arena, initial) = self.initial_paths(input, limits)?;
        let accepted = self.search(input, &mut arena, initial, limits)?;
        let mut analyses = Vec::with_capacity(accepted.len());
        for path in accepted {
            analyses.push(self.materialize_analysis(input, &arena, path)?);
        }
        Ok(analyses)
    }

    /// Generates all graph-valid forms for one dictionary item and ordered morpheme ID list.
    /// Epsilon morphemes may be traversed without being explicitly requested, matching Zemberek.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown dictionary/morpheme IDs, corrupt data, or exceeded limits.
    pub fn generate(
        &self,
        dictionary_id: &str,
        morpheme_ids: &[&str],
    ) -> Result<Vec<NativeAnalysis>, BinaryError> {
        self.generate_with_limits(dictionary_id, morpheme_ids, AnalysisLimits::default())
    }

    /// Generates forms with explicit graph-search limits.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown dictionary/morpheme IDs, corrupt data, or exceeded limits.
    pub fn generate_with_limits(
        &self,
        dictionary_id: &str,
        morpheme_ids: &[&str],
        limits: AnalysisLimits,
    ) -> Result<Vec<NativeAnalysis>, BinaryError> {
        self.generator()?
            .generate_with_limits(dictionary_id, morpheme_ids, limits)
    }

    fn generate_with_candidate_stems(
        &self,
        dictionary: u32,
        morpheme_ids: &[&str],
        limits: AnalysisLimits,
        stem_indices: &[usize],
    ) -> Result<Vec<NativeAnalysis>, BinaryError> {
        validate_limits(limits)?;
        let requested: Vec<u32> = morpheme_ids
            .iter()
            .map(|morpheme| {
                self.morpheme_index_by_id(morpheme)?
                    .ok_or_else(|| invalid(format!("unknown generation morpheme ID {morpheme}")))
            })
            .collect::<Result<_, _>>()?;
        let (mut arena, initial) =
            self.initial_generation_paths(dictionary, &requested, limits, stem_indices)?;
        let accepted = self.search_generation(&mut arena, initial, &requested, limits)?;
        let mut output = Vec::with_capacity(accepted.len());
        for path in accepted {
            let surface = path_surface(&arena, path.search.node);
            output.push(self.materialize_analysis(&surface, &arena, path.search)?);
        }
        deduplicate_analyses_preserving_order(&mut output);
        Ok(output)
    }

    fn initial_generation_paths(
        &self,
        dictionary: u32,
        requested: &[u32],
        limits: AnalysisLimits,
        stem_indices: &[usize],
    ) -> Result<(Vec<PathNode>, Vec<GenerationPath>), BinaryError> {
        let mut arena = Vec::new();
        let mut paths = Vec::new();
        for &index in stem_indices {
            let stem = self.stem_at(index)?;
            if stem.dictionary_index != dictionary {
                return Err(invalid("generation index points to a different dictionary"));
            }
            check_path_capacity(
                paths.len(),
                limits.max_active_paths,
                "generation initial paths",
            )?;
            check_path_capacity(arena.len(), limits.max_path_nodes, "generation path arena")?;
            let state = self.state(stem.state_index)?;
            let node = arena.len();
            arena.push(PathNode {
                parent: None,
                state: stem.state_index,
                morpheme: state.morpheme,
                surface: stem.surface.to_owned(),
                derivative: state.derivative,
                depth: 1,
            });
            let consumed = usize::from(
                requested
                    .first()
                    .is_some_and(|value| *value == state.morpheme),
            );
            paths.push(GenerationPath {
                search: SearchPath {
                    node,
                    dictionary,
                    stem_surface: self
                        .strings
                        .find(stem.surface)?
                        .ok_or_else(|| invalid("generation stem is absent from string table"))?,
                    tail_offset: 0,
                    phonetic_bits: stem.phonetic_bits,
                    contains_derivation: false,
                    contains_suffix_surface: false,
                },
                consumed,
            });
        }
        Ok((arena, paths))
    }

    #[allow(clippy::iter_with_drain)] // drain preserves the reusable Vec allocation across search rounds.
    fn search_generation(
        &self,
        arena: &mut Vec<PathNode>,
        mut current: Vec<GenerationPath>,
        requested: &[u32],
        limits: AnalysisLimits,
    ) -> Result<Vec<GenerationPath>, BinaryError> {
        let mut accepted = Vec::new();
        let mut next = Vec::new();
        while !current.is_empty() {
            check_path_capacity(
                current.len(),
                limits.max_active_paths,
                "generation active paths",
            )?;
            next.clear();
            for path in current.drain(..) {
                let state = self.state(arena[path.search.node].state)?;
                if path.consumed == requested.len()
                    && state.terminal
                    && path.search.phonetic_bits & CANNOT_TERMINATE == 0
                {
                    check_path_capacity(accepted.len(), limits.max_results, "generation results")?;
                    accepted.push(path);
                    continue;
                }
                for edge_index in state.edge_range {
                    if let Some(new_path) =
                        self.try_generation_edge(arena, path, edge_index, requested, limits)?
                    {
                        check_path_capacity(
                            next.len(),
                            limits.max_active_paths,
                            "generation next paths",
                        )?;
                        next.push(new_path);
                    }
                }
            }
            std::mem::swap(&mut current, &mut next);
        }
        Ok(accepted)
    }

    fn try_generation_edge(
        &self,
        arena: &mut Vec<PathNode>,
        path: GenerationPath,
        edge_index: usize,
        requested: &[u32],
        limits: AnalysisLimits,
    ) -> Result<Option<GenerationPath>, BinaryError> {
        let edge = self.edge(edge_index)?;
        let matches_requested = requested
            .get(path.consumed)
            .is_some_and(|morpheme| *morpheme == edge.morpheme);
        if edge.template_count != 0 && !matches_requested {
            return Ok(None);
        }
        let program = self.condition_program(edge)?;
        if !program.is_empty()
            && !ConditionVm::new_generation(self, path.search, arena, program).evaluate()?
        {
            return Ok(None);
        }
        let surface = if edge.template_count == 0 {
            String::new()
        } else {
            self.generate_surface(edge, path.search.phonetic_bits)?
        };
        let phonetic_bits = if surface.is_empty() {
            path.search.phonetic_bits
        } else {
            self.generated_phonetic_bits(path.search.phonetic_bits, &surface, edge)?
        };
        check_path_capacity(arena.len(), limits.max_path_nodes, "generation path arena")?;
        let target = self.state(edge.to_state)?;
        let depth = arena[path.search.node].depth + 1;
        let node = arena.len();
        arena.push(PathNode {
            parent: Some(path.search.node),
            state: edge.to_state,
            morpheme: edge.morpheme,
            surface,
            derivative: target.derivative,
            depth,
        });
        Ok(Some(GenerationPath {
            search: SearchPath {
                node,
                dictionary: path.search.dictionary,
                stem_surface: path.search.stem_surface,
                tail_offset: 0,
                phonetic_bits,
                contains_derivation: path.search.contains_derivation || target.derivative,
                contains_suffix_surface: path.search.contains_suffix_surface
                    || !arena[node].surface.is_empty(),
            },
            consumed: path.consumed + usize::from(matches_requested),
        }))
    }

    fn generated_phonetic_bits(
        &self,
        predecessor: u32,
        surface: &str,
        edge: EdgeData,
    ) -> Result<u32, BinaryError> {
        let mut bits = morphemic_attributes(surface, predecessor);
        bits &= !CANNOT_TERMINATE;
        if let Some(opcode) = self.last_template_opcode(edge)? {
            if opcode == 4 {
                bits |= EXPECTS_CONSONANT;
            } else if opcode == 5 {
                bits |= EXPECTS_VOWEL | CANNOT_TERMINATE;
            }
        }
        Ok(bits)
    }

    fn dictionary_index_by_id(&self, id: &str) -> Result<Option<u32>, BinaryError> {
        let Some(string_id) = self.strings.find(id)? else {
            return Ok(None);
        };
        self.fixed_table_index_by_string(
            Section::Dictionary,
            DICTIONARY_RECORD_SIZE,
            self.view.header.counts[2] as usize,
            string_id,
        )
    }

    fn morpheme_index_by_id(&self, id: &str) -> Result<Option<u32>, BinaryError> {
        let Some(string_id) = self.strings.find(id)? else {
            return Ok(None);
        };
        self.fixed_table_index_by_string(
            Section::Morphemes,
            MORPHEME_RECORD_SIZE,
            self.view.header.counts[1] as usize,
            string_id,
        )
    }

    fn fixed_table_index_by_string(
        &self,
        section: Section,
        record_size: usize,
        count: usize,
        string_id: u32,
    ) -> Result<Option<u32>, BinaryError> {
        let bytes = self.view.section(section)?;
        let mut low = 0_usize;
        let mut high = count;
        while low < high {
            let middle = low + (high - low) / 2;
            let current = read_u32(fixed_record(bytes, middle, record_size)?, 0)?;
            match current.cmp(&string_id) {
                std::cmp::Ordering::Less => low = middle + 1,
                std::cmp::Ordering::Equal => {
                    return Ok(Some(
                        u32::try_from(middle)
                            .map_err(|_| invalid("fixed-table lookup index exceeds u32"))?,
                    ));
                }
                std::cmp::Ordering::Greater => high = middle,
            }
        }
        Ok(None)
    }

    fn analyze_runtime_url(
        &self,
        input: &str,
        limits: AnalysisLimits,
    ) -> Result<Vec<NativeAnalysis>, BinaryError> {
        let normalized = normalize_circumflex(&turkish_lower(input));
        let (lemma, root, ending) = if let Some((start, end)) = apostrophe_range(input) {
            let original_stem = &input[..start];
            let normalized_stem = normalize_circumflex(&turkish_lower(original_stem));
            (
                original_stem.to_owned(),
                normalized_stem
                    .chars()
                    .filter(|value| *value != '.')
                    .collect(),
                normalize_circumflex(&turkish_lower(&input[end..])),
            )
        } else {
            let root = normalized.clone();
            (normalized, root, String::new())
        };
        let pronunciation: String = root
            .chars()
            .filter(|value| is_turkish_letter(*value))
            .collect();
        if pronunciation.is_empty() || !pronunciation.chars().any(is_vowel) {
            return Ok(Vec::new());
        }
        let dictionary_id = format!("{lemma}_Noun_Url");
        self.analyze_runtime_noun_like(
            &dictionary_id,
            &lemma,
            "Url",
            &root,
            &pronunciation,
            &ending,
            "nounProper_S",
            limits,
        )
    }

    fn analyze_runtime_roman_numeral(
        &self,
        input: &str,
        limits: AnalysisLimits,
    ) -> Result<Vec<NativeAnalysis>, BinaryError> {
        let (stem, ending) = split_at_apostrophe(input);
        let numeral = stem.strip_suffix('.').unwrap_or(stem);
        let Some(decimal) = roman_to_decimal(numeral) else {
            return Ok(Vec::new());
        };
        let decimal_string = decimal.to_string();
        let mut lemma = numeral_ending_lemma(&decimal_string);
        if stem.ends_with('.') {
            lemma = ordinal_lemma(lemma).unwrap_or(lemma);
        }
        if lemma.is_empty() {
            return Ok(Vec::new());
        }
        let normalized_ending = normalize_circumflex(&turkish_lower(ending));
        let parse_stem = if !normalized_ending.is_empty()
            && lemma == "dört"
            && normalized_ending.chars().next().is_some_and(is_vowel)
        {
            "dörd"
        } else {
            lemma
        };
        let mut to_parse = String::with_capacity(parse_stem.len() + normalized_ending.len());
        to_parse.push_str(parse_stem);
        to_parse.push_str(&normalized_ending);
        let source = self.analyze_plain(&to_parse, limits)?;
        let dictionary_id = format!("{stem}_Num_RomanNumeral");
        let mut output = Vec::new();
        for analysis in source {
            if analysis.primary_pos == "Num" {
                output.push(rewrite_analysis_root(
                    analysis,
                    &dictionary_id,
                    stem,
                    "Num",
                    "RomanNumeral",
                    stem,
                )?);
            }
        }
        deduplicate_analyses_preserving_order(&mut output);
        Ok(output)
    }

    fn analyze_runtime_abbreviation(
        &self,
        input: &str,
        limits: AnalysisLimits,
    ) -> Result<Vec<NativeAnalysis>, BinaryError> {
        let Some((start, end)) = apostrophe_range(input) else {
            return Ok(Vec::new());
        };
        if start == 0 || end == input.len() {
            return Ok(Vec::new());
        }
        let root: String = normalize_circumflex(&turkish_lower(&input[..start]))
            .chars()
            .filter(|value| *value != '.')
            .collect();
        let pronunciation = if root.chars().any(is_vowel) {
            root.clone()
        } else {
            turkish_letter_pronunciations(&root)
        };
        if pronunciation.is_empty() || !pronunciation.chars().any(is_vowel) {
            return Ok(Vec::new());
        }
        let lemma = turkish_capitalize(input);
        let dictionary_id = format!("{lemma}_Noun_Abbrv");
        let ending = normalize_circumflex(&turkish_lower(&input[end..]));
        self.analyze_runtime_noun_like(
            &dictionary_id,
            &lemma,
            "Abbrv",
            &root,
            &pronunciation,
            &ending,
            "nounAbbrv_S",
            limits,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn analyze_runtime_noun_like(
        &self,
        dictionary_id: &str,
        lemma: &str,
        secondary_pos: &str,
        root: &str,
        pronunciation: &str,
        ending: &str,
        state_id: &str,
        limits: AnalysisLimits,
    ) -> Result<Vec<NativeAnalysis>, BinaryError> {
        let phonetic_bits = morphemic_attributes(pronunciation, 0);
        let Some(candidate) = self.noun_state_candidate(phonetic_bits, state_id)? else {
            return Ok(Vec::new());
        };
        let candidate_dictionary = self.dictionary(candidate.dictionary_index)?;
        let mut to_parse = String::with_capacity(candidate.surface.len() + ending.len());
        to_parse.push_str(candidate.surface);
        to_parse.push_str(ending);
        let source = self.analyze_plain(&to_parse, limits)?;
        let mut output = Vec::new();
        for analysis in source {
            if analysis.dictionary_id == candidate_dictionary.id
                && analysis.stem == candidate.surface
            {
                output.push(rewrite_analysis_root(
                    analysis,
                    dictionary_id,
                    lemma,
                    "Noun",
                    secondary_pos,
                    root,
                )?);
            }
        }
        deduplicate_analyses_preserving_order(&mut output);
        Ok(output)
    }

    fn analyze_runtime_numeral(
        &self,
        input: &str,
        limits: AnalysisLimits,
    ) -> Result<Vec<NativeAnalysis>, BinaryError> {
        let (stem, ending) = split_numeral(input);
        let kinds = RuntimeNumeralKind::classify(stem);
        if kinds.is_empty() {
            return Ok(Vec::new());
        }
        let mut lemma = numeral_ending_lemma(stem.trim_end_matches('.'));
        if stem.ends_with('.') {
            lemma = ordinal_lemma(lemma).unwrap_or(lemma);
        }
        if lemma.is_empty() {
            return Ok(Vec::new());
        }
        let parse_stem =
            if !ending.is_empty() && lemma == "dört" && ending.chars().next().is_some_and(is_vowel)
            {
                "dörd"
            } else {
                lemma
            };
        let mut to_parse = String::with_capacity(parse_stem.len() + ending.len());
        to_parse.push_str(parse_stem);
        to_parse.push_str(ending);
        let source = self.analyze_plain(&to_parse, limits)?;
        let mut output = Vec::new();
        for kind in kinds {
            let dictionary_id = format!("{stem}_Num_{}", kind.secondary_short());
            for analysis in &source {
                if analysis.primary_pos != "Num" {
                    continue;
                }
                output.push(rewrite_analysis_root(
                    analysis.clone(),
                    &dictionary_id,
                    stem,
                    "Num",
                    kind.secondary_short(),
                    stem,
                )?);
            }
        }
        deduplicate_analyses_preserving_order(&mut output);
        Ok(output)
    }

    fn analyze_runtime_proper(
        &self,
        input: &str,
        limits: AnalysisLimits,
    ) -> Result<Vec<NativeAnalysis>, BinaryError> {
        let Some((apostrophe_start, apostrophe_end)) = apostrophe_range(input) else {
            return Ok(Vec::new());
        };
        if apostrophe_start == 0 || apostrophe_end == input.len() {
            return Ok(Vec::new());
        }
        let stem = normalize_runtime_component(&input[..apostrophe_start]).replace('.', "");
        let ending = normalize_runtime_component(&input[apostrophe_end..]);
        if stem.is_empty() || !stem.chars().any(is_vowel) {
            return Ok(Vec::new());
        }
        let phonetic_bits = morphemic_attributes(&stem, 0);
        let Some(candidate) = self.noun_state_candidate(phonetic_bits, "nounProper_S")? else {
            return Ok(Vec::new());
        };
        let candidate_dictionary = self.dictionary(candidate.dictionary_index)?;
        let mut to_parse = String::with_capacity(candidate.surface.len() + ending.len());
        to_parse.push_str(candidate.surface);
        to_parse.push_str(&ending);
        let source = self.analyze_plain(&to_parse, limits)?;
        let mut actual_to_parse = String::with_capacity(stem.len() + ending.len());
        actual_to_parse.push_str(&stem);
        actual_to_parse.push_str(&ending);
        let static_source = self.analyze_plain(&actual_to_parse, limits)?;
        let normalized_word = normalize_apostrophes(input);
        let runtime_lemma = turkish_capitalize(&normalized_word);
        let dictionary_id = format!("{runtime_lemma}_Noun_Prop");
        let mut output = Vec::new();
        for analysis in source {
            if analysis.dictionary_id == candidate_dictionary.id
                && analysis.stem == candidate.surface
            {
                output.push(rewrite_analysis_root(
                    analysis,
                    &dictionary_id,
                    &runtime_lemma,
                    "Noun",
                    "Prop",
                    &stem,
                )?);
            }
        }
        output.extend(
            static_source
                .into_iter()
                .filter(|analysis| analysis.stem == stem),
        );
        deduplicate_analyses_preserving_order(&mut output);
        Ok(output)
    }

    fn noun_state_candidate(
        &self,
        phonetic_bits: u32,
        state_id: &str,
    ) -> Result<Option<StemData<'a>>, BinaryError> {
        let count = self.view.header.counts[3] as usize;
        for index in 0..count {
            let stem = self.stem_at(index)?;
            if stem.phonetic_bits != phonetic_bits {
                continue;
            }
            let state = self.state(stem.state_index)?;
            if state.zemberek_id != state_id {
                continue;
            }
            let dictionary = self.dictionary(stem.dictionary_index)?;
            if PRIMARY_POS_SHORT[usize::from(dictionary.primary_pos)] == "Noun"
                && dictionary.attributes & ROOT_ATTRIBUTE_DUMMY == 0
            {
                return Ok(Some(stem));
            }
        }
        Ok(None)
    }

    fn initial_paths(
        &self,
        input: &str,
        limits: AnalysisLimits,
    ) -> Result<(Vec<PathNode>, Vec<SearchPath>), BinaryError> {
        let mut arena = Vec::new();
        let mut paths = Vec::new();
        if input.is_empty() {
            return Ok((arena, paths));
        }
        let mut trie_node = 0_usize;
        for (offset, byte) in input.bytes().enumerate() {
            let Some(next) = self.stem_prefix_index.child(trie_node, byte) else {
                break;
            };
            trie_node = next;
            if let Some(range) = self.stem_prefix_index.stem_range(trie_node) {
                self.append_initial_range(
                    input,
                    offset + 1,
                    range,
                    &mut arena,
                    &mut paths,
                    limits,
                )?;
            }
        }
        Ok((arena, paths))
    }

    fn append_initial_range(
        &self,
        input: &str,
        end: usize,
        range: Range<usize>,
        arena: &mut Vec<PathNode>,
        paths: &mut Vec<SearchPath>,
        limits: AnalysisLimits,
    ) -> Result<(), BinaryError> {
        let surface = &input[..end];
        let surface_id = self.stem_surface_id(range.start)?;
        for index in range {
            check_path_capacity(paths.len(), limits.max_active_paths, "initial active paths")?;
            check_path_capacity(arena.len(), limits.max_path_nodes, "initial path arena")?;
            let stem = self.stem_at(index)?;
            let state = self.state(stem.state_index)?;
            let node = arena.len();
            arena.push(PathNode {
                parent: None,
                state: stem.state_index,
                morpheme: state.morpheme,
                surface: surface.to_owned(),
                derivative: state.derivative,
                depth: 1,
            });
            paths.push(SearchPath {
                node,
                dictionary: stem.dictionary_index,
                stem_surface: surface_id,
                tail_offset: end,
                phonetic_bits: stem.phonetic_bits,
                contains_derivation: false,
                contains_suffix_surface: false,
            });
        }
        Ok(())
    }

    #[allow(clippy::iter_with_drain)] // drain preserves the reusable Vec allocation across search rounds.
    fn search(
        &self,
        input: &str,
        arena: &mut Vec<PathNode>,
        mut current: Vec<SearchPath>,
        limits: AnalysisLimits,
    ) -> Result<Vec<SearchPath>, BinaryError> {
        let mut accepted = Vec::new();
        let mut next = Vec::new();
        while !current.is_empty() {
            check_path_capacity(current.len(), limits.max_active_paths, "active paths")?;
            next.clear();
            for path in current.drain(..) {
                let state = self.state(arena[path.node].state)?;
                if Self::accepts_finished(input, path, &state) {
                    check_path_capacity(accepted.len(), limits.max_results, "analysis results")?;
                    accepted.push(path);
                    continue;
                }
                self.advance(input, arena, path, state, &mut next, limits)?;
            }
            std::mem::swap(&mut current, &mut next);
        }
        Ok(accepted)
    }

    const fn accepts_finished(input: &str, path: SearchPath, state: &StateData<'_>) -> bool {
        path.tail_offset == input.len()
            && state.terminal
            && path.phonetic_bits & CANNOT_TERMINATE == 0
    }

    fn advance(
        &self,
        input: &str,
        arena: &mut Vec<PathNode>,
        path: SearchPath,
        state: StateData<'a>,
        next: &mut Vec<SearchPath>,
        limits: AnalysisLimits,
    ) -> Result<(), BinaryError> {
        for edge_index in state.edge_range {
            if let Some(new_path) = self.try_edge(input, arena, path, edge_index, limits)? {
                check_path_capacity(next.len(), limits.max_active_paths, "next active paths")?;
                next.push(new_path);
            }
        }
        Ok(())
    }

    fn try_edge(
        &self,
        input: &str,
        arena: &mut Vec<PathNode>,
        path: SearchPath,
        edge_index: usize,
        limits: AnalysisLimits,
    ) -> Result<Option<SearchPath>, BinaryError> {
        let edge = self.edge(edge_index)?;
        let tail = &input[path.tail_offset..];
        if tail.is_empty() && edge.template_count != 0 {
            return Ok(None);
        }
        let surface = self.generate_surface(edge, path.phonetic_bits)?;
        if !tail.starts_with(&surface) {
            return Ok(None);
        }
        let program = self.condition_program(edge)?;
        if !program.is_empty()
            && !ConditionVm::new_analysis(self, input, path, arena, program).evaluate()?
        {
            return Ok(None);
        }
        check_path_capacity(arena.len(), limits.max_path_nodes, "path arena")?;
        let target = self.state(edge.to_state)?;
        let phonetic_bits = self.next_phonetic_bits(path, tail, &surface, edge)?;
        let node = arena.len();
        let depth = arena[path.node].depth + 1;
        let surface_len = surface.len();
        let has_surface = !surface.is_empty();
        arena.push(PathNode {
            parent: Some(path.node),
            state: edge.to_state,
            morpheme: edge.morpheme,
            surface,
            derivative: target.derivative,
            depth,
        });
        Ok(Some(SearchPath {
            node,
            dictionary: path.dictionary,
            stem_surface: path.stem_surface,
            tail_offset: path.tail_offset + surface_len,
            phonetic_bits,
            contains_derivation: path.contains_derivation || target.derivative,
            contains_suffix_surface: path.contains_suffix_surface || has_surface,
        }))
    }

    fn next_phonetic_bits(
        &self,
        path: SearchPath,
        tail: &str,
        surface: &str,
        edge: EdgeData,
    ) -> Result<u32, BinaryError> {
        if surface.is_empty() {
            return Ok(path.phonetic_bits);
        }
        let mut bits = if tail == surface {
            path.phonetic_bits
        } else {
            morphemic_attributes(surface, path.phonetic_bits)
        };
        bits &= !CANNOT_TERMINATE;
        if let Some(opcode) = self.last_template_opcode(edge)? {
            if opcode == 4 {
                bits |= EXPECTS_CONSONANT;
            } else if opcode == 5 {
                bits |= EXPECTS_VOWEL | CANNOT_TERMINATE;
            }
        }
        Ok(bits)
    }

    fn generate_surface(&self, edge: EdgeData, predecessor: u32) -> Result<String, BinaryError> {
        let mut output = String::with_capacity(edge.template_count.saturating_mul(2));
        for relative in 0..edge.template_count {
            let index = edge.template_start + relative;
            let token = self
                .templates
                .get(index)
                .copied()
                .ok_or_else(|| invalid("runtime template index is out of bounds"))?;
            let attributes = if matches!(token.opcode, 1 | 2 | 3 | 6) {
                morphemic_attributes(&output, predecessor)
            } else {
                0
            };
            realize_token(
                &mut output,
                token.opcode,
                token.append,
                token.letter,
                relative,
                predecessor,
                attributes,
            )?;
        }
        Ok(output)
    }

    fn last_template_opcode(&self, edge: EdgeData) -> Result<Option<u8>, BinaryError> {
        if edge.template_count == 0 {
            return Ok(None);
        }
        let index = edge.template_start + edge.template_count - 1;
        self.templates
            .get(index)
            .map(|token| Some(token.opcode))
            .ok_or_else(|| invalid("runtime template index is out of bounds"))
    }

    fn condition_program(&self, edge: EdgeData) -> Result<&'a [u8], BinaryError> {
        let section = self.view.section(Section::Conditions)?;
        let end = checked_add(
            edge.condition_start,
            edge.condition_length,
            "runtime condition range",
        )?;
        section
            .get(edge.condition_start..end)
            .ok_or_else(|| invalid("runtime condition range is out of bounds"))
    }

    fn materialize_analysis(
        &self,
        input: &str,
        arena: &[PathNode],
        path: SearchPath,
    ) -> Result<NativeAnalysis, BinaryError> {
        let source_dictionary = self.dictionary(path.dictionary)?;
        let dictionary = if source_dictionary.attributes & ROOT_ATTRIBUTE_DUMMY != 0 {
            let reference = source_dictionary
                .reference
                .ok_or_else(|| invalid("dummy dictionary item has no reference"))?;
            self.dictionary(reference)?
        } else {
            source_dictionary
        };
        let mut nodes = history_indices(arena, path.node);
        nodes.reverse();
        let mut morphemes = Vec::with_capacity(nodes.len());
        let mut canonical =
            String::with_capacity(input.len() + nodes.len() * 8 + dictionary.id.len());
        canonical.push_str(dictionary.id);
        canonical.push('\u{1}');
        let root_node = nodes
            .first()
            .copied()
            .ok_or_else(|| invalid("accepted analysis has no root node"))?;
        let stem = arena[root_node].surface.clone();
        let mut ending = String::new();
        for node_index in nodes {
            let node = &arena[node_index];
            let morpheme = self.morpheme(node.morpheme)?;
            if morpheme.id == "Nom" || morpheme.id == "Pnon" {
                continue;
            }
            canonical.push_str(morpheme.id);
            canonical.push('=');
            canonical.push_str(&node.surface);
            canonical.push('\u{2}');
            if node_index != root_node {
                ending.push_str(&node.surface);
            }
            morphemes.push(NativeMorpheme {
                id: morpheme.id.to_owned(),
                name: morpheme.name.to_owned(),
                surface: node.surface.clone(),
                derivational: morpheme.derivational,
                informal: morpheme.informal,
                pos: morpheme.pos.map(str::to_owned),
                mapped_id: morpheme.mapped_id.map(str::to_owned),
            });
        }
        Ok(NativeAnalysis {
            canonical,
            dictionary_id: dictionary.id.to_owned(),
            lemma: dictionary.lemma.to_owned(),
            primary_pos: PRIMARY_POS_SHORT[usize::from(dictionary.primary_pos)].to_owned(),
            secondary_pos: SECONDARY_POS_SHORT[usize::from(dictionary.secondary_pos)].to_owned(),
            surface_form: input.to_owned(),
            stem,
            ending,
            morphemes,
        })
    }

    fn stem_range(&self, surface_id: u32) -> Result<Range<usize>, BinaryError> {
        let count = self.view.header.counts[3] as usize;
        let mut low = 0_usize;
        let mut high = count;
        while low < high {
            let middle = low + (high - low) / 2;
            if self.stem_surface_id(middle)? < surface_id {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        let start = low;
        high = count;
        while low < high {
            let middle = low + (high - low) / 2;
            if self.stem_surface_id(middle)? <= surface_id {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        Ok(start..low)
    }

    fn stem_surface_id(&self, index: usize) -> Result<u32, BinaryError> {
        self.stems
            .get(index)
            .map(|stem| stem.surface_id)
            .ok_or_else(|| invalid("runtime stem index is out of bounds"))
    }

    fn stem_at(&self, index: usize) -> Result<StemData<'a>, BinaryError> {
        self.stems
            .get(index)
            .copied()
            .ok_or_else(|| invalid("runtime stem index is out of bounds"))
    }

    fn dictionary(&self, index: u32) -> Result<DictionaryData<'a>, BinaryError> {
        let index = require_index(index, self.dictionaries.len(), "runtime dictionary")?;
        self.dictionaries
            .get(index)
            .copied()
            .ok_or_else(|| invalid("runtime dictionary index is out of bounds"))
    }

    fn morpheme(&self, index: u32) -> Result<MorphemeData<'a>, BinaryError> {
        let index = require_index(index, self.morphemes.len(), "runtime morpheme")?;
        self.morphemes
            .get(index)
            .copied()
            .ok_or_else(|| invalid("runtime morpheme index is out of bounds"))
    }

    fn state(&self, index: u32) -> Result<StateData<'a>, BinaryError> {
        let index = require_index(index, self.states.len(), "runtime state")?;
        self.states
            .get(index)
            .cloned()
            .ok_or_else(|| invalid("runtime state index is out of bounds"))
    }

    fn edge(&self, index: usize) -> Result<EdgeData, BinaryError> {
        self.edges
            .get(index)
            .copied()
            .ok_or_else(|| invalid("runtime edge index is out of bounds"))
    }
}

fn decode_stem_table<'a>(
    view: BinaryBundleView<'a>,
    strings: &StringTable<'a>,
) -> Result<Vec<StemData<'a>>, BinaryError> {
    let section = view.section(Section::Stems)?;
    let count = view.header.counts[3] as usize;
    let mut stems = Vec::with_capacity(count);
    for index in 0..count {
        let record = fixed_record(section, index, STEM_RECORD_SIZE)?;
        let surface_id = read_u32(record, 0)?;
        stems.push(StemData {
            surface_id,
            surface: strings.get(surface_id)?,
            dictionary_index: read_u32(record, 4)?,
            state_index: read_u32(record, 8)?,
            phonetic_bits: read_u32(record, 12)?,
        });
    }
    Ok(stems)
}

fn decode_dictionary_table<'a>(
    view: BinaryBundleView<'a>,
    strings: &StringTable<'a>,
) -> Result<Vec<DictionaryData<'a>>, BinaryError> {
    let section = view.section(Section::Dictionary)?;
    let count = view.header.counts[2] as usize;
    let mut dictionaries = Vec::with_capacity(count);
    for index in 0..count {
        let record = fixed_record(section, index, DICTIONARY_RECORD_SIZE)?;
        let reference = read_u32(record, 24)?;
        dictionaries.push(DictionaryData {
            id: strings.get(read_u32(record, 0)?)?,
            lemma: strings.get(read_u32(record, 4)?)?,
            primary_pos: read_u16(record, 16)?,
            secondary_pos: read_u16(record, 18)?,
            attributes: read_u32(record, 20)?,
            reference: (reference != NONE_U32).then_some(reference),
        });
    }
    Ok(dictionaries)
}

fn decode_morpheme_table<'a>(
    view: BinaryBundleView<'a>,
    strings: &StringTable<'a>,
) -> Result<Vec<MorphemeData<'a>>, BinaryError> {
    let section = view.section(Section::Morphemes)?;
    let count = view.header.counts[1] as usize;
    let mut ids = Vec::with_capacity(count);
    for index in 0..count {
        let record = fixed_record(section, index, MORPHEME_RECORD_SIZE)?;
        ids.push(strings.get(read_u32(record, 0)?)?);
    }
    let mut morphemes = Vec::with_capacity(count);
    for index in 0..count {
        let record = fixed_record(section, index, MORPHEME_RECORD_SIZE)?;
        let flags = read_u16(record, 10)?;
        let pos = read_u16(record, 8)?;
        let mapped = read_u32(record, 12)?;
        let mapped_id = if mapped == NONE_U32 {
            None
        } else {
            Some(
                *ids.get(require_index(mapped, count, "mapped morpheme")?)
                    .ok_or_else(|| invalid("mapped morpheme is out of bounds"))?,
            )
        };
        morphemes.push(MorphemeData {
            id: ids[index],
            name: strings.get(read_u32(record, 4)?)?,
            derivational: flags & 1 != 0,
            informal: flags & 2 != 0,
            pos: if pos == NONE_U16 {
                None
            } else {
                Some(PRIMARY_POS_SHORT[usize::from(pos)])
            },
            mapped_id,
        });
    }
    Ok(morphemes)
}

fn decode_template_table(view: BinaryBundleView<'_>) -> Result<Vec<TemplateData>, BinaryError> {
    let section = view.section(Section::Templates)?;
    if section.len() % TEMPLATE_RECORD_SIZE != 0 {
        return Err(invalid("template section is not record aligned"));
    }
    let count = section.len() / TEMPLATE_RECORD_SIZE;
    let mut templates = Vec::with_capacity(count);
    for index in 0..count {
        let record = fixed_record(section, index, TEMPLATE_RECORD_SIZE)?;
        let scalar = read_u32(record, 4)?;
        templates.push(TemplateData {
            opcode: record[0],
            append: record[1],
            letter: if scalar == 0 {
                None
            } else {
                char::from_u32(scalar)
            },
        });
    }
    Ok(templates)
}

fn decode_state_table<'a>(
    view: BinaryBundleView<'a>,
    strings: &StringTable<'a>,
) -> Result<Vec<StateData<'a>>, BinaryError> {
    let section = view.section(Section::States)?;
    let count = view.header.counts[4] as usize;
    let edge_count = view.header.counts[5] as usize;
    let mut states = Vec::with_capacity(count);
    for index in 0..count {
        let record = fixed_record(section, index, STATE_RECORD_SIZE)?;
        let flags = read_u32(record, 12)?;
        let start = read_u32(record, 16)? as usize;
        let length = read_u32(record, 20)? as usize;
        let end = start
            .checked_add(length)
            .ok_or_else(|| invalid("state edge range overflow"))?;
        if end > edge_count {
            return Err(invalid("state edge range exceeds edge table"));
        }
        strings.get(read_u32(record, 0)?)?;
        let zemberek_id = strings.get(read_u32(record, 4)?)?;
        states.push(StateData {
            zemberek_id,
            morpheme: read_u32(record, 8)?,
            terminal: flags & 1 != 0,
            derivative: flags & 2 != 0,
            edge_range: start..end,
        });
    }
    Ok(states)
}

fn decode_edge_table(
    view: BinaryBundleView<'_>,
    strings: &StringTable<'_>,
) -> Result<Vec<EdgeData>, BinaryError> {
    let section = view.section(Section::Edges)?;
    let count = view.header.counts[5] as usize;
    let mut edges = Vec::with_capacity(count);
    for index in 0..count {
        let record = fixed_record(section, index, EDGE_RECORD_SIZE)?;
        read_u32(record, 0)?;
        strings.get(read_u32(record, 12)?)?;
        read_u16(record, 30)?;
        read_u32(record, 32)?;
        edges.push(EdgeData {
            to_state: read_u32(record, 4)?,
            morpheme: read_u32(record, 8)?,
            template_start: read_u32(record, 16)? as usize,
            condition_start: read_u32(record, 20)? as usize,
            condition_length: read_u32(record, 24)? as usize,
            template_count: read_u16(record, 28)? as usize,
        });
    }
    Ok(edges)
}

impl std::fmt::Debug for NativeMorphology<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeMorphology")
            .field("summary", &self.summary())
            .finish()
    }
}

#[derive(Clone, Copy)]
struct StemPrefixNode {
    edge_start: u32,
    edge_count: u16,
    stem_start: u32,
    stem_end: u32,
}

#[derive(Clone, Copy)]
struct StemPrefixEdge {
    byte: u8,
    child: u32,
}

struct StemPrefixIndex {
    nodes: Vec<StemPrefixNode>,
    edges: Vec<StemPrefixEdge>,
}

#[derive(Clone, Copy)]
struct BuildStemNode {
    first_edge: Option<usize>,
    stem_start: u32,
    stem_end: u32,
}

#[derive(Clone, Copy)]
struct BuildStemEdge {
    byte: u8,
    child: usize,
    next: Option<usize>,
}

impl StemPrefixIndex {
    fn empty() -> Self {
        Self {
            nodes: vec![StemPrefixNode {
                edge_start: 0,
                edge_count: 0,
                stem_start: NONE_U32,
                stem_end: NONE_U32,
            }],
            edges: Vec::new(),
        }
    }

    fn build(morphology: &NativeMorphology<'_>) -> Result<Self, BinaryError> {
        let stem_count = morphology.view.header.counts[3] as usize;
        let mut nodes = vec![BuildStemNode {
            first_edge: None,
            stem_start: NONE_U32,
            stem_end: NONE_U32,
        }];
        let mut edges = Vec::<BuildStemEdge>::new();
        let mut start = 0_usize;
        while start < stem_count {
            let surface_id = morphology.stem_surface_id(start)?;
            let surface = morphology.strings.get(surface_id)?;
            let mut end = start + 1;
            while end < stem_count && morphology.stem_surface_id(end)? == surface_id {
                end += 1;
            }
            let mut node = 0_usize;
            for byte in surface.bytes() {
                let mut cursor = nodes[node].first_edge;
                let mut child = None;
                while let Some(edge_index) = cursor {
                    let edge = edges[edge_index];
                    if edge.byte == byte {
                        child = Some(edge.child);
                        break;
                    }
                    cursor = edge.next;
                }
                node = if let Some(child) = child {
                    child
                } else {
                    let child = nodes.len();
                    nodes.push(BuildStemNode {
                        first_edge: None,
                        stem_start: NONE_U32,
                        stem_end: NONE_U32,
                    });
                    let edge_index = edges.len();
                    edges.push(BuildStemEdge {
                        byte,
                        child,
                        next: nodes[node].first_edge,
                    });
                    nodes[node].first_edge = Some(edge_index);
                    child
                };
            }
            if nodes[node].stem_start != NONE_U32 {
                return Err(invalid("stem prefix index contains a duplicate terminal"));
            }
            nodes[node].stem_start =
                u32::try_from(start).map_err(|_| invalid("stem prefix index start exceeds u32"))?;
            nodes[node].stem_end =
                u32::try_from(end).map_err(|_| invalid("stem prefix index end exceeds u32"))?;
            start = end;
        }

        let mut compact_nodes = Vec::with_capacity(nodes.len());
        let mut compact_edges = Vec::with_capacity(edges.len());
        let mut outgoing = Vec::<(u8, usize)>::new();
        for node in nodes {
            outgoing.clear();
            let mut cursor = node.first_edge;
            while let Some(edge_index) = cursor {
                let edge = edges[edge_index];
                outgoing.push((edge.byte, edge.child));
                cursor = edge.next;
            }
            outgoing.sort_unstable_by_key(|entry| entry.0);
            let edge_start = u32::try_from(compact_edges.len())
                .map_err(|_| invalid("stem prefix edge start exceeds u32"))?;
            let edge_count = u16::try_from(outgoing.len())
                .map_err(|_| invalid("stem prefix node has too many edges"))?;
            for &(byte, child) in &outgoing {
                compact_edges.push(StemPrefixEdge {
                    byte,
                    child: u32::try_from(child)
                        .map_err(|_| invalid("stem prefix child exceeds u32"))?,
                });
            }
            compact_nodes.push(StemPrefixNode {
                edge_start,
                edge_count,
                stem_start: node.stem_start,
                stem_end: node.stem_end,
            });
        }
        Ok(Self {
            nodes: compact_nodes,
            edges: compact_edges,
        })
    }

    fn child(&self, node: usize, byte: u8) -> Option<usize> {
        let node = *self.nodes.get(node)?;
        let start = usize::try_from(node.edge_start).ok()?;
        let end = start.checked_add(usize::from(node.edge_count))?;
        let edges = self.edges.get(start..end)?;
        let index = edges.binary_search_by_key(&byte, |edge| edge.byte).ok()?;
        usize::try_from(edges[index].child).ok()
    }

    fn stem_range(&self, node: usize) -> Option<Range<usize>> {
        let node = *self.nodes.get(node)?;
        if node.stem_start == NONE_U32 {
            return None;
        }
        Some(usize::try_from(node.stem_start).ok()?..usize::try_from(node.stem_end).ok()?)
    }
}

#[derive(Clone, Copy)]
struct StemData<'a> {
    surface_id: u32,
    surface: &'a str,
    dictionary_index: u32,
    state_index: u32,
    phonetic_bits: u32,
}

impl<'a> StemData<'a> {
    const fn public_view(self) -> NativeStem<'a> {
        NativeStem {
            surface: self.surface,
            dictionary_index: self.dictionary_index,
            state_index: self.state_index,
            phonetic_bits: self.phonetic_bits,
        }
    }
}

#[derive(Clone, Copy)]
struct DictionaryData<'a> {
    id: &'a str,
    lemma: &'a str,
    primary_pos: u16,
    secondary_pos: u16,
    attributes: u32,
    reference: Option<u32>,
}

#[derive(Clone, Copy)]
struct MorphemeData<'a> {
    id: &'a str,
    name: &'a str,
    derivational: bool,
    informal: bool,
    pos: Option<&'static str>,
    mapped_id: Option<&'a str>,
}

#[derive(Clone, Copy)]
struct TemplateData {
    opcode: u8,
    append: u8,
    letter: Option<char>,
}

#[derive(Clone)]
struct StateData<'a> {
    zemberek_id: &'a str,
    morpheme: u32,
    terminal: bool,
    derivative: bool,
    edge_range: Range<usize>,
}

#[derive(Clone, Copy)]
struct EdgeData {
    to_state: u32,
    morpheme: u32,
    template_start: usize,
    condition_start: usize,
    condition_length: usize,
    template_count: usize,
}

#[derive(Clone)]
struct PathNode {
    parent: Option<usize>,
    state: u32,
    morpheme: u32,
    surface: String,
    derivative: bool,
    depth: usize,
}

#[derive(Clone, Copy)]
struct SearchPath {
    node: usize,
    dictionary: u32,
    stem_surface: u32,
    tail_offset: usize,
    phonetic_bits: u32,
    contains_derivation: bool,
    contains_suffix_surface: bool,
}

#[derive(Clone, Copy)]
struct GenerationPath {
    search: SearchPath,
    consumed: usize,
}

const INLINE_CONDITION_STACK: usize = 32;

struct BoolStack {
    inline: [bool; INLINE_CONDITION_STACK],
    len: usize,
    overflow: Option<Vec<bool>>,
}

impl BoolStack {
    const fn new() -> Self {
        Self {
            inline: [false; INLINE_CONDITION_STACK],
            len: 0,
            overflow: None,
        }
    }

    fn push(&mut self, value: bool) {
        if let Some(values) = &mut self.overflow {
            values.push(value);
            return;
        }
        if self.len < INLINE_CONDITION_STACK {
            self.inline[self.len] = value;
            self.len += 1;
            return;
        }
        let mut values = Vec::with_capacity(INLINE_CONDITION_STACK * 2);
        values.extend_from_slice(&self.inline);
        values.push(value);
        self.overflow = Some(values);
    }

    fn negate_last(&mut self) -> Option<()> {
        if let Some(values) = &mut self.overflow {
            let value = values.last_mut()?;
            *value = !*value;
            return Some(());
        }
        let index = self.len.checked_sub(1)?;
        self.inline[index] = !self.inline[index];
        Some(())
    }

    fn reduce_last(&mut self, count: usize, conjunction: bool) -> Option<()> {
        if let Some(values) = &mut self.overflow {
            if count > values.len() {
                return None;
            }
            let start = values.len() - count;
            let value = if conjunction {
                values[start..].iter().all(|item| *item)
            } else {
                values[start..].iter().any(|item| *item)
            };
            values.truncate(start);
            values.push(value);
            if values.len() <= INLINE_CONDITION_STACK {
                self.len = values.len();
                self.inline[..self.len].copy_from_slice(values);
                self.overflow = None;
            }
            return Some(());
        }
        if count > self.len {
            return None;
        }
        let start = self.len - count;
        let value = if conjunction {
            self.inline[start..self.len].iter().all(|item| *item)
        } else {
            self.inline[start..self.len].iter().any(|item| *item)
        };
        self.len = start;
        self.push(value);
        Some(())
    }

    fn single(&self) -> Option<bool> {
        if let Some(values) = &self.overflow {
            return (values.len() == 1).then(|| values[0]);
        }
        (self.len == 1).then(|| self.inline[0])
    }
}

struct ConditionVm<'m, 'a, 'path> {
    morphology: &'m NativeMorphology<'a>,
    path: SearchPath,
    arena: &'path [PathNode],
    program: &'a [u8],
    has_tail: bool,
    position: usize,
    stack: BoolStack,
}

impl<'m, 'a, 'path> ConditionVm<'m, 'a, 'path> {
    fn new_analysis(
        morphology: &'m NativeMorphology<'a>,
        input: &str,
        path: SearchPath,
        arena: &'path [PathNode],
        program: &'a [u8],
    ) -> Self {
        Self::new(
            morphology,
            path,
            arena,
            program,
            path.tail_offset < input.len(),
        )
    }

    fn new_generation(
        morphology: &'m NativeMorphology<'a>,
        path: SearchPath,
        arena: &'path [PathNode],
        program: &'a [u8],
    ) -> Self {
        Self::new(morphology, path, arena, program, true)
    }

    fn new(
        morphology: &'m NativeMorphology<'a>,
        path: SearchPath,
        arena: &'path [PathNode],
        program: &'a [u8],
        has_tail: bool,
    ) -> Self {
        Self {
            morphology,
            path,
            arena,
            program,
            has_tail,
            position: 0,
            stack: BoolStack::new(),
        }
    }

    fn evaluate(mut self) -> Result<bool, BinaryError> {
        while self.position < self.program.len() {
            let opcode = read_byte(self.program, &mut self.position)?;
            if (0x01..=0x03).contains(&opcode) {
                self.evaluate_structural(opcode)?;
            } else {
                let value = self.evaluate_leaf(opcode)?;
                self.stack.push(value);
            }
        }
        self.stack
            .single()
            .ok_or_else(|| invalid("condition VM ended with invalid stack depth"))
    }

    fn evaluate_structural(&mut self, opcode: u8) -> Result<(), BinaryError> {
        if opcode == 0x03 {
            self.stack
                .negate_last()
                .ok_or_else(|| invalid("condition NOT stack underflow"))?;
            return Ok(());
        }
        let count = read_program_u16(self.program, &mut self.position)? as usize;
        if count < 2 {
            return Err(invalid("condition AND/OR stack underflow"));
        }
        let conjunction = opcode == 0x01;
        self.stack
            .reduce_last(count, conjunction)
            .ok_or_else(|| invalid("condition AND/OR stack underflow"))
    }

    fn evaluate_leaf(&mut self, opcode: u8) -> Result<bool, BinaryError> {
        match opcode {
            0x10..=0x1a => self.evaluate_feature(opcode),
            0x1b..=0x1e | 0x21..=0x22 | 0x2a..=0x2e => self.evaluate_morpheme_or_surface(opcode),
            0x1f..=0x20 | 0x23..=0x29 | 0x2f => self.evaluate_state_or_group(opcode),
            _ => Err(invalid(format!(
                "condition VM unknown opcode {opcode:#04x}"
            ))),
        }
    }

    fn evaluate_feature(&mut self, opcode: u8) -> Result<bool, BinaryError> {
        let dictionary = self.morphology.dictionary(self.path.dictionary)?;
        match opcode {
            0x10 => {
                let attribute = read_byte(self.program, &mut self.position)?;
                Ok(dictionary.attributes & (1_u32 << attribute) != 0)
            }
            0x11 => {
                let attributes = read_program_u32(self.program, &mut self.position)?;
                Ok(dictionary.attributes & attributes != 0)
            }
            0x12 => {
                let attribute = read_byte(self.program, &mut self.position)?;
                Ok(self.path.phonetic_bits & (1_u32 << attribute) != 0)
            }
            0x13 => Ok(self.path.dictionary == read_program_u32(self.program, &mut self.position)?),
            0x14 => {
                Ok(dictionary.primary_pos
                    == u16::from(read_byte(self.program, &mut self.position)?))
            }
            0x15 => {
                Ok(dictionary.secondary_pos
                    == u16::from(read_byte(self.program, &mut self.position)?))
            }
            0x16 => self.dictionary_set_contains(),
            0x17 => self.dictionary_set_contains().map(|value| !value),
            0x18 => Ok(self.path.contains_suffix_surface),
            0x19 => Ok(self.has_tail),
            0x1a => Ok(!self.has_tail),
            _ => Err(invalid("condition VM routed feature opcode incorrectly")),
        }
    }

    fn dictionary_set_contains(&mut self) -> Result<bool, BinaryError> {
        let list = self.read_index_list()?;
        Ok(list.contains(self.path.dictionary) && self.path.dictionary != NONE_U32)
    }

    fn evaluate_morpheme_or_surface(&mut self, opcode: u8) -> Result<bool, BinaryError> {
        match opcode {
            0x1b => {
                let list = self.read_index_list()?;
                Ok(self.has_tail_sequence(list)?)
            }
            0x1c => {
                let list = self.read_index_list()?;
                Ok(self.contains_morpheme_sequence(list)?)
            }
            0x1d => {
                Ok(self.current_node().morpheme
                    == read_program_u32(self.program, &mut self.position)?)
            }
            0x1e => Ok(self.previous_node().map(|node| node.morpheme)
                == Some(read_program_u32(self.program, &mut self.position)?)),
            0x21 => {
                Ok(self.path.stem_surface == read_program_u32(self.program, &mut self.position)?)
            }
            0x22 => {
                let list = self.read_index_list()?;
                Ok(list.contains(self.path.stem_surface))
            }
            0x2a => {
                let list = self.read_index_list()?;
                Ok(self.previous_group_contains_morpheme(list))
            }
            0x2b => Ok(self.no_surface_after_derivation()),
            0x2c => {
                let list = self.read_index_list()?;
                Ok(self.history_contains_morpheme(list))
            }
            0x2d => {
                let list = self.read_index_list()?;
                Ok(self
                    .previous_node()
                    .is_some_and(|node| list.contains(node.morpheme)))
            }
            0x2e => {
                let list = self.read_index_list()?;
                Ok(list.contains(self.current_node().morpheme))
            }
            _ => Err(invalid(
                "condition VM routed morpheme or surface opcode incorrectly",
            )),
        }
    }

    fn evaluate_state_or_group(&mut self, opcode: u8) -> Result<bool, BinaryError> {
        match opcode {
            0x23 => Ok(
                self.current_node().state == read_program_u32(self.program, &mut self.position)?
            ),
            0x24 => Ok(
                self.current_node().state != read_program_u32(self.program, &mut self.position)?
            ),
            0x25 => Ok(self.last_derivation_state()
                == Some(read_program_u32(self.program, &mut self.position)?)),
            0x26 => Ok(self.path.contains_derivation),
            0x27 => {
                let list = self.read_index_list()?;
                Ok(self
                    .last_derivation_state()
                    .is_some_and(|state| list.contains(state)))
            }
            0x28 => {
                let list = self.read_index_list()?;
                Ok(self.current_group_contains_state(list))
            }
            0x29 => {
                let list = self.read_index_list()?;
                Ok(self.previous_group_contains_state(list))
            }
            0x2f => {
                let list = self.read_index_list()?;
                Ok(self
                    .previous_node()
                    .is_some_and(|node| list.contains(node.state)))
            }
            0x1f => Ok(self.previous_node().map(|node| node.state)
                == Some(read_program_u32(self.program, &mut self.position)?)),
            0x20 => Ok(self.previous_node().map(|node| node.state)
                != Some(read_program_u32(self.program, &mut self.position)?)),
            _ => Err(invalid("condition VM routed state opcode incorrectly")),
        }
    }

    fn read_index_list(&mut self) -> Result<IndexList<'a>, BinaryError> {
        let count = read_program_u16(self.program, &mut self.position)? as usize;
        let byte_count = checked_mul(count, 4, "condition VM list bytes")?;
        let end = checked_add(self.position, byte_count, "condition VM list end")?;
        let bytes = self
            .program
            .get(self.position..end)
            .ok_or_else(|| invalid("condition VM list is out of bounds"))?;
        self.position = end;
        Ok(IndexList { bytes, count })
    }

    fn current_node(&self) -> &PathNode {
        &self.arena[self.path.node]
    }

    fn previous_node(&self) -> Option<&PathNode> {
        self.current_node().parent.map(|index| &self.arena[index])
    }

    fn has_tail_sequence(&self, list: IndexList<'_>) -> Result<bool, BinaryError> {
        if self.current_node().depth < list.count {
            return Ok(false);
        }
        let mut node = Some(self.path.node);
        for expected in (0..list.count).rev() {
            let Some(index) = node else {
                return Ok(false);
            };
            if self.arena[index].morpheme != list.get(expected)? {
                return Ok(false);
            }
            node = self.arena[index].parent;
        }
        Ok(true)
    }

    fn contains_morpheme_sequence(&self, list: IndexList<'_>) -> Result<bool, BinaryError> {
        if list.count == 0 || self.current_node().depth < list.count {
            return Ok(false);
        }
        let mut history = history_indices(self.arena, self.path.node);
        history.reverse();
        for window in history.windows(list.count) {
            let mut matches = true;
            for (offset, node) in window.iter().enumerate() {
                if self.arena[*node].morpheme != list.get(offset)? {
                    matches = false;
                    break;
                }
            }
            if matches {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn last_derivation_state(&self) -> Option<u32> {
        let mut index = self.path.node;
        while let Some(parent) = self.arena[index].parent {
            if self.arena[index].derivative {
                return Some(self.arena[index].state);
            }
            index = parent;
        }
        None
    }

    fn current_group_contains_state(&self, list: IndexList<'_>) -> bool {
        let mut index = self.path.node;
        while let Some(parent) = self.arena[index].parent {
            let node = &self.arena[index];
            if list.contains(node.state) {
                return true;
            }
            if node.derivative {
                return false;
            }
            index = parent;
        }
        false
    }

    fn previous_group_start(&self) -> Option<usize> {
        let mut index = self.path.node;
        while let Some(parent) = self.arena[index].parent {
            if self.arena[index].derivative {
                return Some(parent);
            }
            index = parent;
        }
        None
    }

    fn previous_group_contains_state(&self, list: IndexList<'_>) -> bool {
        let Some(mut index) = self.previous_group_start() else {
            return false;
        };
        while let Some(parent) = self.arena[index].parent {
            let node = &self.arena[index];
            if list.contains(node.state) {
                return true;
            }
            if node.derivative {
                return false;
            }
            index = parent;
        }
        false
    }

    fn previous_group_contains_morpheme(&self, list: IndexList<'_>) -> bool {
        let Some(mut index) = self.previous_group_start() else {
            return false;
        };
        while let Some(parent) = self.arena[index].parent {
            let node = &self.arena[index];
            if list.contains(node.morpheme) {
                return true;
            }
            if node.derivative {
                return false;
            }
            index = parent;
        }
        false
    }

    fn no_surface_after_derivation(&self) -> bool {
        let mut index = self.path.node;
        while let Some(parent) = self.arena[index].parent {
            let node = &self.arena[index];
            if node.derivative {
                return true;
            }
            if !node.surface.is_empty() {
                return false;
            }
            index = parent;
        }
        true
    }

    fn history_contains_morpheme(&self, list: IndexList<'_>) -> bool {
        let mut index = Some(self.path.node);
        while let Some(node_index) = index {
            let node = &self.arena[node_index];
            if list.contains(node.morpheme) {
                return true;
            }
            index = node.parent;
        }
        false
    }
}

#[derive(Clone, Copy)]
struct IndexList<'a> {
    bytes: &'a [u8],
    count: usize,
}

impl IndexList<'_> {
    fn get(self, index: usize) -> Result<u32, BinaryError> {
        if index >= self.count {
            return Err(invalid("condition VM list index is out of bounds"));
        }
        read_u32(self.bytes, index * 4)
    }

    fn contains(self, needle: u32) -> bool {
        (0..self.count)
            .any(|index| read_u32(self.bytes, index * 4).is_ok_and(|value| value == needle))
    }
}

#[derive(Clone, Copy)]
enum RuntimeNumeralKind {
    Cardinal,
    Ordinal,
    Range,
    Ratio,
    Real,
    Distribution,
    Percentage,
    Clock,
    Date,
}

impl RuntimeNumeralKind {
    fn classify(stem: &str) -> Vec<Self> {
        let mut output = Vec::new();
        if is_signed_digits(stem) {
            output.push(Self::Cardinal);
        }
        if stem.strip_suffix('.').is_some_and(is_signed_digits) {
            output.push(Self::Ordinal);
        }
        if is_numeric_pair(stem, '-') {
            output.push(Self::Range);
        }
        if is_numeric_pair(stem, '/') {
            output.push(Self::Ratio);
        }
        if is_real(stem) {
            output.push(Self::Real);
        }
        if is_distribution(stem) {
            output.push(Self::Distribution);
        }
        if is_percentage(stem) {
            output.push(Self::Percentage);
        }
        if is_clock(stem) {
            output.push(Self::Clock);
        }
        if is_date(stem) {
            output.push(Self::Date);
        }
        output
    }

    const fn secondary_short(self) -> &'static str {
        match self {
            Self::Cardinal => "Card",
            Self::Ordinal => "Ord",
            Self::Range => "Range",
            Self::Ratio => "Ratio",
            Self::Real => "Real",
            Self::Distribution => "Dist",
            Self::Percentage => "Percent",
            Self::Clock => "Clock",
            Self::Date => "Date",
        }
    }
}

fn rewrite_analysis_root(
    mut analysis: NativeAnalysis,
    dictionary_id: &str,
    lemma: &str,
    primary_pos: &str,
    secondary_pos: &str,
    stem: &str,
) -> Result<NativeAnalysis, BinaryError> {
    let root = analysis
        .morphemes
        .first_mut()
        .ok_or_else(|| invalid("runtime copied analysis has no root morpheme"))?;
    root.surface.clear();
    root.surface.push_str(stem);
    analysis.dictionary_id.clear();
    analysis.dictionary_id.push_str(dictionary_id);
    analysis.lemma.clear();
    analysis.lemma.push_str(lemma);
    analysis.primary_pos.clear();
    analysis.primary_pos.push_str(primary_pos);
    analysis.secondary_pos.clear();
    analysis.secondary_pos.push_str(secondary_pos);
    analysis.stem.clear();
    analysis.stem.push_str(stem);
    analysis.surface_form.clear();
    analysis.surface_form.push_str(stem);
    analysis.surface_form.push_str(&analysis.ending);
    analysis.canonical = canonical_key(dictionary_id, &analysis.morphemes);
    Ok(analysis)
}

fn canonical_key(dictionary_id: &str, morphemes: &[NativeMorpheme]) -> String {
    let surface_bytes: usize = morphemes
        .iter()
        .map(|morpheme| morpheme.surface.len())
        .sum();
    let mut output =
        String::with_capacity(dictionary_id.len() + surface_bytes + morphemes.len() * 8);
    output.push_str(dictionary_id);
    output.push('\u{1}');
    for morpheme in morphemes {
        output.push_str(&morpheme.id);
        output.push('=');
        output.push_str(&morpheme.surface);
        output.push('\u{2}');
    }
    output
}

fn split_at_apostrophe(input: &str) -> (&str, &str) {
    apostrophe_range(input).map_or((input, ""), |(start, end)| (&input[..start], &input[end..]))
}

fn is_url_token(input: &str) -> bool {
    let (stem, _) = split_at_apostrophe(input);
    let lower = stem.to_ascii_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") || lower.starts_with("www.") {
        return lower.len() > 4 && lower.contains('.');
    }
    let domain = lower.split('/').next().unwrap_or(&lower);
    [".com", ".org", ".edu", ".gov", ".net", ".info"]
        .iter()
        .any(|suffix| domain.contains(suffix))
}

fn is_roman_numeral_token(input: &str) -> bool {
    let (stem, _) = split_at_apostrophe(input);
    let numeral = stem.strip_suffix('.').unwrap_or(stem);
    !numeral.is_empty()
        && numeral
            .chars()
            .all(|value| matches!(value, 'I' | 'L' | 'V' | 'C' | 'D' | 'M' | 'X'))
        && roman_to_decimal(numeral).is_some()
}

fn roman_to_decimal(input: &str) -> Option<u32> {
    let upper = input.to_ascii_uppercase();
    let mut total = 0_u32;
    let mut previous = 0_u32;
    for value in upper.chars().rev() {
        let current = match value {
            'I' => 1,
            'V' => 5,
            'X' => 10,
            'L' => 50,
            'C' => 100,
            'D' => 500,
            'M' => 1000,
            _ => return None,
        };
        if current < previous {
            total = total.checked_sub(current)?;
        } else {
            total = total.checked_add(current)?;
            previous = current;
        }
    }
    if total == 0 || total > 3999 || decimal_to_roman(total) != upper {
        return None;
    }
    Some(total)
}

fn decimal_to_roman(mut value: u32) -> String {
    const VALUES: [(u32, &str); 13] = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ];
    let mut output = String::new();
    for (amount, symbol) in VALUES {
        while value >= amount {
            output.push_str(symbol);
            value -= amount;
        }
    }
    output
}

fn is_dotted_abbreviation_token(input: &str) -> bool {
    let (stem, ending) = split_at_apostrophe(input);
    if ending.is_empty() || !stem.contains('.') {
        return false;
    }
    let mut saw_group = false;
    let mut characters = stem.chars();
    while let Some(letter) = characters.next() {
        if !is_turkish_uppercase_letter(letter) || characters.next() != Some('.') {
            return false;
        }
        saw_group = true;
    }
    saw_group
}

const fn is_turkish_uppercase_letter(value: char) -> bool {
    matches!(
        value,
        'A'..='Z' | 'Ç' | 'Ğ' | 'İ' | 'Ö' | 'Ş' | 'Ü' | 'Â' | 'Î' | 'Û'
    )
}

fn turkish_letter_pronunciations(input: &str) -> String {
    let mut output = String::new();
    let count = input.chars().count();
    for (index, value) in input.chars().enumerate() {
        let pronunciation = match value {
            'a' => "a",
            'b' => "be",
            'c' => "ce",
            'ç' => "çe",
            'd' => "de",
            'e' => "e",
            'f' => "fe",
            'g' => "ge",
            'ğ' => "yumuşakge",
            'h' => "he",
            'ı' => "ı",
            'i' => "i",
            'j' => "je",
            'k' if index + 1 == count => "ka",
            'k' => "ke",
            'l' => "le",
            'm' => "me",
            'n' => "ne",
            'o' => "o",
            'ö' => "ö",
            'p' => "pe",
            'r' => "re",
            's' => "se",
            'ş' => "şe",
            't' => "te",
            'u' => "u",
            'ü' => "ü",
            'v' => "ve",
            'y' => "ye",
            'z' => "ze",
            'w' => "dabılyu",
            'q' => "kü",
            'x' => "iks",
            _ => "",
        };
        output.push_str(pronunciation);
    }
    output
}

fn split_numeral(input: &str) -> (&str, &str) {
    if let Some((start, end)) = apostrophe_range(input) {
        return (&input[..start], &input[end..]);
    }
    let mut cut = input.len();
    for (index, value) in input.char_indices().rev() {
        if value == '.' || value.is_ascii_digit() {
            break;
        }
        cut = index;
    }
    (&input[..cut], &input[cut..])
}

fn numeral_ending_lemma(input: &str) -> &'static str {
    const ONES: [&str; 10] = [
        "sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz",
    ];
    const TENS: [&str; 10] = [
        "", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan",
    ];
    let mut zeros = 0_usize;
    let mut saw_digit = false;
    for value in input.chars().rev() {
        let Some(digit) = value.to_digit(10) else {
            if zeros >= 2 {
                return "sıfır";
            }
            break;
        };
        saw_digit = true;
        if digit == 0 {
            zeros += 1;
            continue;
        }
        let index = digit as usize;
        return match zeros {
            0 => ONES[index],
            1 => TENS[index],
            2 => "yüz",
            3..=5 => "bin",
            6..=8 => "milyon",
            9..=11 => "milyar",
            _ => "",
        };
    }
    if saw_digit {
        match zeros {
            0 | 1 => "sıfır",
            2 => "yüz",
            3..=5 => "bin",
            6..=8 => "milyon",
            9..=11 => "milyar",
            _ => "",
        }
    } else {
        ""
    }
}

fn ordinal_lemma(value: &str) -> Option<&'static str> {
    match value {
        "sıfır" => Some("sıfırıncı"),
        "bir" => Some("birinci"),
        "iki" => Some("ikinci"),
        "üç" => Some("üçüncü"),
        "dört" => Some("dördüncü"),
        "beş" => Some("beşinci"),
        "altı" => Some("altıncı"),
        "yedi" => Some("yedinci"),
        "sekiz" => Some("sekizinci"),
        "dokuz" => Some("dokuzuncu"),
        "on" => Some("onuncu"),
        "yirmi" => Some("yirminci"),
        "otuz" => Some("otuzuncu"),
        "kırk" => Some("kırkıncı"),
        "elli" => Some("ellinci"),
        "altmış" => Some("altmışıncı"),
        "yetmiş" => Some("yetmişinci"),
        "seksen" => Some("sekseninci"),
        "doksan" => Some("doksanıncı"),
        "yüz" => Some("yüzüncü"),
        "bin" => Some("bininci"),
        "milyon" => Some("milyonuncu"),
        "milyar" => Some("milyarıncı"),
        _ => None,
    }
}

fn normalize_for_analysis(input: &str) -> String {
    let lowered = normalize_circumflex(&turkish_lower(input));
    let no_dots: String = lowered.chars().filter(|value| *value != '.').collect();
    let selected = if no_dots.is_empty() { lowered } else { no_dots };
    normalize_apostrophes(&selected)
}

fn normalize_runtime_component(input: &str) -> String {
    normalize_circumflex(&turkish_lower(input))
        .chars()
        .map(foreign_diacritic_to_turkish)
        .map(|value| {
            if is_turkish_letter(value) || matches!(value, '.' | '-') {
                value
            } else {
                '?'
            }
        })
        .collect()
}

const fn foreign_diacritic_to_turkish(value: char) -> char {
    match value {
        'à' | 'á' | 'ã' | 'ä' | 'å' => 'a',
        'è' | 'é' | 'ê' | 'ë' => 'e',
        'ì' | 'í' | 'ï' => 'i',
        'ñ' => 'n',
        'ò' | 'ó' | 'ô' | 'õ' => 'o',
        'ù' | 'ú' => 'u',
        _ => value,
    }
}

fn normalize_circumflex(input: &str) -> String {
    input
        .chars()
        .map(|value| match value {
            'â' => 'a',
            'î' => 'i',
            'û' => 'u',
            'Â' => 'A',
            'Î' => 'İ',
            'Û' => 'U',
            _ => value,
        })
        .collect()
}

fn normalize_apostrophes(input: &str) -> String {
    input
        .chars()
        .map(|value| {
            if matches!(value, '\'' | '\u{2032}' | '´' | '`' | '’' | '‘') {
                '\''
            } else {
                value
            }
        })
        .collect()
}

fn turkish_lower(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    for value in input.chars() {
        match value {
            'I' => output.push('ı'),
            'İ' => output.push('i'),
            _ => output.extend(value.to_lowercase()),
        }
    }
    output
}

fn turkish_capitalize(input: &str) -> String {
    let lowered = turkish_lower(input);
    let mut characters = lowered.chars();
    let Some(first) = characters.next() else {
        return lowered;
    };
    let mut output = String::with_capacity(lowered.len());
    match first {
        'i' => output.push('İ'),
        'ı' => output.push('I'),
        _ => output.extend(first.to_uppercase()),
    }
    output.extend(characters);
    output
}

const fn is_turkish_letter(value: char) -> bool {
    matches!(
        value,
        'a' | 'b'
            | 'c'
            | 'ç'
            | 'd'
            | 'e'
            | 'f'
            | 'g'
            | 'ğ'
            | 'h'
            | 'ı'
            | 'i'
            | 'j'
            | 'k'
            | 'l'
            | 'm'
            | 'n'
            | 'o'
            | 'ö'
            | 'p'
            | 'r'
            | 's'
            | 'ş'
            | 't'
            | 'u'
            | 'ü'
            | 'v'
            | 'y'
            | 'z'
            | 'x'
            | 'w'
            | 'q'
    )
}

fn is_signed_digits(input: &str) -> bool {
    let body = input
        .strip_prefix('+')
        .or_else(|| input.strip_prefix('-'))
        .unwrap_or(input);
    !body.is_empty() && body.chars().all(|value| value.is_ascii_digit())
}

fn is_numeric_pair(input: &str, separator: char) -> bool {
    let body = input
        .strip_prefix('+')
        .or_else(|| input.strip_prefix('-'))
        .unwrap_or(input);
    let mut parts = body.split(separator);
    let Some(left) = parts.next() else {
        return false;
    };
    let Some(right) = parts.next() else {
        return false;
    };
    parts.next().is_none()
        && !left.is_empty()
        && !right.is_empty()
        && left.chars().all(|value| value.is_ascii_digit())
        && right.chars().all(|value| value.is_ascii_digit())
}

fn is_real(input: &str) -> bool {
    is_numeric_pair(input, ',') || is_numeric_pair(input, '.')
}

fn is_distribution(input: &str) -> bool {
    let digit_count = input.chars().take_while(char::is_ascii_digit).count();
    digit_count > 0
        && digit_count < input.chars().count()
        && input
            .chars()
            .skip(digit_count)
            .all(|value| !value.is_ascii_digit())
}

fn is_percentage(input: &str) -> bool {
    let body = input
        .strip_prefix('+')
        .or_else(|| input.strip_prefix('-'))
        .unwrap_or(input);
    let Some(number) = body.strip_prefix('%') else {
        return false;
    };
    is_signed_digits(number) || is_real(number)
}

fn is_clock(input: &str) -> bool {
    for separator in [':', '.'] {
        let mut parts = input.split(separator);
        let (Some(hour), Some(minute), None) = (parts.next(), parts.next(), parts.next()) else {
            continue;
        };
        if minute.len() != 2 || !minute.chars().all(|value| value.is_ascii_digit()) {
            continue;
        }
        let Ok(hour_value) = hour.parse::<u8>() else {
            continue;
        };
        let Ok(minute_value) = minute.parse::<u8>() else {
            continue;
        };
        if (1..=29).contains(&hour_value) && minute_value <= 59 {
            return true;
        }
    }
    false
}

fn is_date(input: &str) -> bool {
    for separator in ['.', '/'] {
        let mut parts = input.split(separator);
        let (Some(day), Some(month), Some(year), None) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        if year.len() != 4 || !year.chars().all(|value| value.is_ascii_digit()) {
            continue;
        }
        let (Ok(day), Ok(month)) = (day.parse::<u8>(), month.parse::<u8>()) else {
            continue;
        };
        if day <= 39 && month <= 19 && day.to_string().len() <= 2 && month.to_string().len() <= 2 {
            return true;
        }
    }
    false
}

fn apostrophe_range(input: &str) -> Option<(usize, usize)> {
    input.char_indices().find_map(|(start, value)| {
        matches!(value, '\'' | '\u{2032}' | '´' | '`' | '’' | '‘')
            .then_some((start, start + value.len_utf8()))
    })
}

fn realize_token(
    output: &mut String,
    opcode: u8,
    append: u8,
    letter: Option<char>,
    token_index: usize,
    predecessor: u32,
    attributes: u32,
) -> Result<(), BinaryError> {
    match opcode {
        1 => realize_i_vowel(output, token_index, predecessor, attributes),
        2 => realize_a_vowel(output, token_index, predecessor, attributes),
        3 => {
            let mut value = require_runtime_letter(letter, "devoice token")?;
            if attributes & LAST_LETTER_VOICELESS != 0 {
                value = devoice(value);
            }
            output.push(value);
            Ok(())
        }
        4 | 5 | 7 => {
            output.push(require_runtime_letter(letter, "literal template token")?);
            Ok(())
        }
        6 => {
            if attributes & LAST_LETTER_VOWEL != 0 {
                output.push(require_runtime_letter(letter, "append template token")?);
            }
            Ok(())
        }
        _ => Err(invalid(format!("runtime unknown template opcode {opcode}"))),
    }?;
    if append > 1 {
        return Err(invalid("runtime template append flag is not boolean"));
    }
    Ok(())
}

fn realize_a_vowel(
    output: &mut String,
    token_index: usize,
    predecessor: u32,
    attributes: u32,
) -> Result<(), BinaryError> {
    if token_index == 0 && predecessor & LAST_LETTER_VOWEL != 0 {
        return Ok(());
    }
    if attributes & LAST_VOWEL_BACK != 0 {
        output.push('a');
    } else if attributes & LAST_VOWEL_FRONTAL != 0 {
        output.push('e');
    } else {
        return Err(invalid("runtime cannot generate A-vowel harmony"));
    }
    Ok(())
}

fn realize_i_vowel(
    output: &mut String,
    token_index: usize,
    predecessor: u32,
    attributes: u32,
) -> Result<(), BinaryError> {
    if token_index == 0 && predecessor & LAST_LETTER_VOWEL != 0 {
        return Ok(());
    }
    let value = if attributes & LAST_VOWEL_FRONTAL != 0 && attributes & LAST_VOWEL_UNROUNDED != 0 {
        'i'
    } else if attributes & LAST_VOWEL_BACK != 0 && attributes & LAST_VOWEL_UNROUNDED != 0 {
        'ı'
    } else if attributes & LAST_VOWEL_BACK != 0 && attributes & LAST_VOWEL_ROUNDED != 0 {
        'u'
    } else if attributes & LAST_VOWEL_FRONTAL != 0 && attributes & LAST_VOWEL_ROUNDED != 0 {
        'ü'
    } else {
        return Err(invalid("runtime cannot generate I-vowel harmony"));
    };
    output.push(value);
    Ok(())
}

fn morphemic_attributes(surface: &str, predecessor: u32) -> u32 {
    if surface.is_empty() {
        return predecessor;
    }
    let mut bits = if surface.chars().any(is_vowel) {
        attributes_with_vowel(surface)
    } else {
        let mut inherited = predecessor;
        inherited |= LAST_LETTER_CONSONANT | FIRST_LETTER_CONSONANT | HAS_NO_VOWEL;
        inherited &= !(LAST_LETTER_VOWEL | EXPECTS_CONSONANT);
        inherited
    };
    let last = surface.chars().next_back().unwrap_or('\0');
    if is_voiceless(last) {
        bits |= LAST_LETTER_VOICELESS;
        if is_stop_consonant(last) {
            bits |= LAST_LETTER_VOICELESS_STOP;
        }
    } else {
        bits |= LAST_LETTER_VOICED;
    }
    bits
}

fn attributes_with_vowel(surface: &str) -> u32 {
    let first = surface.chars().next().unwrap_or('\0');
    let last = surface.chars().next_back().unwrap_or('\0');
    let last_vowel = surface
        .chars()
        .rev()
        .find(|value| is_vowel(*value))
        .unwrap_or('\0');
    let mut bits = if is_vowel(last) {
        LAST_LETTER_VOWEL
    } else {
        LAST_LETTER_CONSONANT
    };
    bits |= if is_frontal(last_vowel) {
        LAST_VOWEL_FRONTAL
    } else {
        LAST_VOWEL_BACK
    };
    bits |= if is_rounded(last_vowel) {
        LAST_VOWEL_ROUNDED
    } else {
        LAST_VOWEL_UNROUNDED
    };
    bits |= if is_vowel(first) {
        FIRST_LETTER_VOWEL
    } else {
        FIRST_LETTER_CONSONANT
    };
    bits
}

const fn is_vowel(value: char) -> bool {
    matches!(
        value,
        'a' | 'A'
            | 'e'
            | 'E'
            | 'ı'
            | 'I'
            | 'i'
            | 'İ'
            | 'o'
            | 'O'
            | 'ö'
            | 'Ö'
            | 'u'
            | 'U'
            | 'ü'
            | 'Ü'
            | 'â'
            | 'Â'
            | 'î'
            | 'Î'
            | 'û'
            | 'Û'
    )
}

const fn is_frontal(value: char) -> bool {
    matches!(
        value,
        'e' | 'E' | 'i' | 'İ' | 'ö' | 'Ö' | 'ü' | 'Ü' | 'î' | 'Î' | 'û' | 'Û'
    )
}

const fn is_rounded(value: char) -> bool {
    matches!(
        value,
        'o' | 'O' | 'ö' | 'Ö' | 'u' | 'U' | 'ü' | 'Ü' | 'û' | 'Û'
    )
}

const fn is_voiceless(value: char) -> bool {
    matches!(
        value,
        'ç' | 'Ç'
            | 'f'
            | 'F'
            | 'h'
            | 'H'
            | 'k'
            | 'K'
            | 'p'
            | 'P'
            | 's'
            | 'S'
            | 'ş'
            | 'Ş'
            | 't'
            | 'T'
    )
}

const fn is_stop_consonant(value: char) -> bool {
    matches!(value, 'ç' | 'Ç' | 'k' | 'K' | 'p' | 'P' | 't' | 'T')
}

const fn devoice(value: char) -> char {
    match value {
        'b' => 'p',
        'B' => 'P',
        'c' => 'ç',
        'C' => 'Ç',
        'd' => 't',
        'D' => 'T',
        'g' | 'ğ' => 'k',
        'G' | 'Ğ' => 'K',
        _ => value,
    }
}

fn require_runtime_letter(value: Option<char>, label: &str) -> Result<char, BinaryError> {
    value.ok_or_else(|| invalid(format!("runtime {label} has no letter")))
}

fn path_surface(arena: &[PathNode], node: usize) -> String {
    let mut indices = history_indices(arena, node);
    indices.reverse();
    let byte_count: usize = indices
        .iter()
        .map(|index| arena[*index].surface.len())
        .sum();
    let mut output = String::with_capacity(byte_count);
    for index in indices {
        output.push_str(&arena[index].surface);
    }
    output
}

fn history_indices(arena: &[PathNode], node: usize) -> Vec<usize> {
    let mut output = Vec::with_capacity(arena[node].depth);
    let mut current = Some(node);
    while let Some(index) = current {
        output.push(index);
        current = arena[index].parent;
    }
    output
}

fn deduplicate_analyses_preserving_order(analyses: &mut Vec<NativeAnalysis>) {
    let mut seen = HashSet::with_capacity(analyses.len());
    analyses.retain(|analysis| seen.insert(analysis.canonical.clone()));
}

fn validate_limits(limits: AnalysisLimits) -> Result<(), BinaryError> {
    if limits.max_active_paths == 0 || limits.max_path_nodes == 0 || limits.max_results == 0 {
        return Err(invalid("analysis limits must be greater than zero"));
    }
    Ok(())
}

fn check_path_capacity(current: usize, limit: usize, label: &str) -> Result<(), BinaryError> {
    if current >= limit {
        Err(invalid(format!("{label} limit {limit} exceeded")))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{morphemic_attributes, realize_a_vowel, realize_i_vowel};
    use super::{LAST_LETTER_VOICELESS, LAST_VOWEL_BACK, LAST_VOWEL_FRONTAL};
    use super::{LAST_VOWEL_ROUNDED, LAST_VOWEL_UNROUNDED};

    #[test]
    fn runtime_component_matches_foreign_diacritic_and_circumflex_rules() {
        assert_eq!(super::normalize_runtime_component("ÂÁÑŔ"), "aan?");
    }

    #[test]
    fn computes_turkish_harmony_attributes() {
        let front = morphemic_attributes("ev", 0);
        assert_ne!(front & LAST_VOWEL_FRONTAL, 0);
        assert_ne!(front & LAST_VOWEL_UNROUNDED, 0);
        let back = morphemic_attributes("kitap", 0);
        assert_ne!(back & LAST_VOWEL_BACK, 0);
        assert_ne!(back & LAST_LETTER_VOICELESS, 0);
    }

    #[test]
    fn realizes_a_and_i_harmony() -> Result<(), super::BinaryError> {
        let mut output = String::new();
        realize_a_vowel(&mut output, 1, 0, LAST_VOWEL_FRONTAL)?;
        assert_eq!(output, "e");
        output.clear();
        realize_i_vowel(&mut output, 1, 0, LAST_VOWEL_ROUNDED | LAST_VOWEL_BACK)?;
        assert_eq!(output, "u");
        Ok(())
    }
}
