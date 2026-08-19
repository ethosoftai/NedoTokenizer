//! Strict loader and semantic validator for Nedo morphology reference bundles.

#![forbid(unsafe_code)]

use core::fmt;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::Deserialize;

mod binary;
mod disambiguation;

pub use disambiguation::{
    ambiguity_word_data, AmbiguityScoringCode, AmbiguityWordData, DisambiguationError,
    DisambiguationScoreCache, NativeDisambiguation, NativeDisambiguator, PerceptronModel,
    SharedPreparedWordCache,
};

pub use binary::{
    compile_binary, AnalysisLimits, BinaryBundleView, BinaryError, BinarySummary, NativeAnalysis,
    NativeGenerator, NativeMorpheme, NativeMorphology, NativeStem, StemMatches,
};

/// Current supported graph-bundle schema.
pub const BUNDLE_SCHEMA_VERSION: u32 = 1;
/// Exact upstream source identity accepted by schema v1.
pub const ZEMBEREK_COMMIT: &str = "ae2fbe31438dda4dddc674a2a8991d518984d392";

/// A fully parsed and cross-reference-validated morphology bundle.
#[derive(Debug)]
pub struct MorphBundle {
    manifest: BundleManifest,
    morphemes: Vec<MorphemeRecord>,
    dictionary: Vec<DictionaryRecord>,
    stems: Vec<StemRecord>,
    states: Vec<StateRecord>,
    edges: Vec<EdgeRecord>,
}

impl MorphBundle {
    /// Loads all bundle files and validates every schema and graph invariant.
    ///
    /// # Errors
    ///
    /// Returns an error for I/O failure, malformed or unknown JSON fields,
    /// unsupported schema identity, count mismatch, invalid template/condition,
    /// duplicate identity, or a broken cross-reference.
    pub fn load_directory(path: &Path) -> Result<Self, BundleError> {
        let manifest: BundleManifest = read_json(&path.join("manifest.json"))?;
        let morphemes = read_jsonl(&path.join("morphemes.jsonl"))?;
        let dictionary = read_jsonl(&path.join("dictionary.jsonl"))?;
        let stems = read_jsonl(&path.join("stems.jsonl"))?;
        let states = read_jsonl(&path.join("states.jsonl"))?;
        let edges = read_jsonl(&path.join("edges.jsonl"))?;
        let bundle = Self {
            manifest,
            morphemes,
            dictionary,
            stems,
            states,
            edges,
        };
        bundle.validate()?;
        Ok(bundle)
    }

    /// Returns a compact validated bundle summary.
    #[must_use]
    pub const fn summary(&self) -> BundleSummary {
        BundleSummary {
            morphemes: self.morphemes.len(),
            dictionary: self.dictionary.len(),
            stems: self.stems.len(),
            states: self.states.len(),
            edges: self.edges.len(),
            owner_mismatches: self.manifest.owner_declared_from_mismatch_count,
            duplicate_state_ids: self.manifest.duplicate_zemberek_state_id_count,
            condition_kinds: self.manifest.condition_kind_counts.len(),
        }
    }

    /// Returns the validated manifest.
    #[must_use]
    pub const fn manifest(&self) -> &BundleManifest {
        &self.manifest
    }

    fn validate(&self) -> Result<(), BundleError> {
        self.validate_manifest_identity()?;
        self.validate_manifest_counts()?;

        let morpheme_ids =
            validate_sorted_unique(&self.morphemes, |record| record.id.as_str(), "morpheme id")?;
        let dictionary_ids = validate_sorted_unique(
            &self.dictionary,
            |record| record.id.as_str(),
            "dictionary id",
        )?;
        validate_sorted_unique(&self.states, |record| record.key.as_str(), "state key")?;
        validate_sequential_edge_ids(&self.edges)?;

        let morpheme_by_id: HashMap<&str, &MorphemeRecord> = self
            .morphemes
            .iter()
            .map(|record| (record.id.as_str(), record))
            .collect();
        let dictionary_by_id: HashMap<&str, &DictionaryRecord> = self
            .dictionary
            .iter()
            .map(|record| (record.id.as_str(), record))
            .collect();
        let state_by_key: HashMap<&str, &StateRecord> = self
            .states
            .iter()
            .map(|record| (record.key.as_str(), record))
            .collect();

        for record in &self.morphemes {
            record.validate(&morpheme_ids)?;
        }
        for record in &self.dictionary {
            record.validate(&dictionary_ids)?;
        }
        for record in &self.states {
            record.validate(&morpheme_ids)?;
        }
        for record in &self.stems {
            record.validate(&dictionary_by_id, &state_by_key)?;
        }

        self.validate_graph_aggregates(&morpheme_by_id, &dictionary_by_id, &state_by_key)?;
        Ok(())
    }

    fn validate_graph_aggregates(
        &self,
        morpheme_by_id: &HashMap<&str, &MorphemeRecord>,
        dictionary_by_id: &HashMap<&str, &DictionaryRecord>,
        state_by_key: &HashMap<&str, &StateRecord>,
    ) -> Result<(), BundleError> {
        let mut owner_edge_counts: HashMap<&str, usize> = HashMap::new();
        let mut owner_mismatches = 0_usize;
        let mut condition_counts: BTreeMap<&'static str, usize> = BTreeMap::new();
        for edge in &self.edges {
            edge.validate(
                morpheme_by_id,
                dictionary_by_id,
                state_by_key,
                &mut condition_counts,
            )?;
            let count = owner_edge_counts
                .entry(edge.owner_state.as_str())
                .or_insert(0);
            *count = count
                .checked_add(1)
                .ok_or_else(|| invalid("owner edge count overflow"))?;
            if !edge.owner_matches_declared_from {
                owner_mismatches = owner_mismatches
                    .checked_add(1)
                    .ok_or_else(|| invalid("owner mismatch count overflow"))?;
            }
        }

        for state in &self.states {
            let actual = owner_edge_counts
                .get(state.key.as_str())
                .copied()
                .unwrap_or(0);
            if state.outgoing_count != actual {
                return Err(invalid(format!(
                    "state {} outgoing_count {} != actual {actual}",
                    state.key, state.outgoing_count
                )));
            }
        }
        if owner_mismatches != self.manifest.owner_declared_from_mismatch_count {
            return Err(invalid(format!(
                "owner mismatch count {} != manifest {}",
                owner_mismatches, self.manifest.owner_declared_from_mismatch_count
            )));
        }

        let duplicate_state_ids =
            duplicate_excess(self.states.iter().map(|state| state.zemberek_id.as_str()))?;
        if duplicate_state_ids != self.manifest.duplicate_zemberek_state_id_count {
            return Err(invalid(format!(
                "duplicate state id count {duplicate_state_ids} != manifest {}",
                self.manifest.duplicate_zemberek_state_id_count
            )));
        }
        let aliasless_states = self
            .states
            .iter()
            .filter(|state| state.declared_fields.is_empty())
            .count();
        if aliasless_states != self.manifest.aliasless_state_count {
            return Err(invalid(format!(
                "aliasless state count {aliasless_states} != manifest {}",
                self.manifest.aliasless_state_count
            )));
        }
        let reflected_field_count = self.states.iter().try_fold(0_usize, |total, state| {
            total
                .checked_add(state.declared_fields.len())
                .ok_or_else(|| invalid("reflected field count overflow"))
        })?;
        if reflected_field_count != self.manifest.reflected_state_field_count {
            return Err(invalid(format!(
                "reflected state field count {reflected_field_count} != manifest {}",
                self.manifest.reflected_state_field_count
            )));
        }

        let manifest_condition_counts = self.manifest.condition_count_map()?;
        if condition_counts != manifest_condition_counts {
            return Err(invalid(format!(
                "condition inventory mismatch: actual={condition_counts:?} manifest={manifest_condition_counts:?}"
            )));
        }
        Ok(())
    }

    fn validate_manifest_identity(&self) -> Result<(), BundleError> {
        if self.manifest.schema_version != BUNDLE_SCHEMA_VERSION {
            return Err(invalid(format!(
                "unsupported bundle schema {}",
                self.manifest.schema_version
            )));
        }
        if self.manifest.zemberek_commit != ZEMBEREK_COMMIT {
            return Err(invalid(format!(
                "unsupported Zemberek commit {}",
                self.manifest.zemberek_commit
            )));
        }
        if self.manifest.zemberek_version != "0.17.2" {
            return Err(invalid(format!(
                "unsupported Zemberek version {}",
                self.manifest.zemberek_version
            )));
        }
        if self.manifest.exporter != "nedo-zemberek-graph-bundle-v1" {
            return Err(invalid(format!(
                "unsupported exporter {}",
                self.manifest.exporter
            )));
        }
        Ok(())
    }

    fn validate_manifest_counts(&self) -> Result<(), BundleError> {
        let pairs = [
            (
                "morpheme_count",
                self.manifest.morpheme_count,
                self.morphemes.len(),
            ),
            (
                "dictionary_count",
                self.manifest.dictionary_count,
                self.dictionary.len(),
            ),
            ("stem_count", self.manifest.stem_count, self.stems.len()),
            ("state_count", self.manifest.state_count, self.states.len()),
            ("edge_count", self.manifest.edge_count, self.edges.len()),
        ];
        for (name, declared, actual) in pairs {
            if declared != actual {
                return Err(invalid(format!(
                    "manifest {name} {declared} != actual {actual}"
                )));
            }
        }
        Ok(())
    }
}

/// Validated high-level bundle counts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleSummary {
    /// Number of morphemes.
    pub morphemes: usize,
    /// Number of dictionary entries.
    pub dictionary: usize,
    /// Number of generated stem transitions.
    pub stems: usize,
    /// Number of distinct state objects.
    pub states: usize,
    /// Number of outgoing-owner graph edges.
    pub edges: usize,
    /// Number of edges whose owner differs from the Java object's `from` field.
    pub owner_mismatches: usize,
    /// Number of duplicate upstream textual state IDs beyond the first occurrence.
    pub duplicate_state_ids: usize,
    /// Number of distinct source condition implementation kinds.
    pub condition_kinds: usize,
}

/// Strict graph bundle manifest.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleManifest {
    schema_version: u32,
    zemberek_commit: String,
    zemberek_version: String,
    exporter: String,
    morpheme_count: usize,
    dictionary_count: usize,
    stem_count: usize,
    state_count: usize,
    edge_count: usize,
    owner_declared_from_mismatch_count: usize,
    reflected_state_field_count: usize,
    duplicate_zemberek_state_id_count: usize,
    aliasless_state_count: usize,
    condition_kind_counts: Vec<ConditionKindCount>,
}

impl BundleManifest {
    fn condition_count_map(&self) -> Result<BTreeMap<&'static str, usize>, BundleError> {
        let mut result = BTreeMap::new();
        let mut previous = None;
        for entry in &self.condition_kind_counts {
            let kind = known_condition_manifest_kind(&entry.kind).ok_or_else(|| {
                invalid(format!("unknown manifest condition kind {}", entry.kind))
            })?;
            if previous.is_some_and(|value: &str| value >= entry.kind.as_str()) {
                return Err(invalid("manifest condition kinds are not strictly sorted"));
            }
            previous = Some(entry.kind.as_str());
            if result.insert(kind, entry.count).is_some() {
                return Err(invalid(format!(
                    "duplicate manifest condition kind {}",
                    entry.kind
                )));
            }
        }
        Ok(result)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConditionKindCount {
    kind: String,
    count: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MorphemeRecord {
    id: String,
    name: String,
    pos: Option<String>,
    derivational: bool,
    informal: bool,
    mapped_id: Option<String>,
}

impl MorphemeRecord {
    fn validate(&self, morpheme_ids: &HashSet<&str>) -> Result<(), BundleError> {
        if self.id.is_empty() || self.name.is_empty() {
            return Err(invalid("empty morpheme id or name"));
        }
        if let Some(pos) = &self.pos {
            if !PRIMARY_POS_SHORT.contains(&pos.as_str()) {
                return Err(invalid(format!("unknown morpheme POS {pos}")));
            }
        }
        if let Some(mapped_id) = &self.mapped_id {
            require_reference(morpheme_ids, mapped_id, "mapped morpheme")?;
        }
        let _ = (self.derivational, self.informal);
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DictionaryRecord {
    id: String,
    lemma: String,
    root: String,
    pronunciation: String,
    primary_pos: String,
    secondary_pos: String,
    attributes: Vec<String>,
    reference_id: Option<String>,
    index: i32,
}

impl DictionaryRecord {
    fn validate(&self, dictionary_ids: &HashSet<&str>) -> Result<(), BundleError> {
        if self.id.is_empty() {
            return Err(invalid("empty dictionary id"));
        }
        if !PRIMARY_POS_SHORT.contains(&self.primary_pos.as_str()) {
            return Err(invalid(format!(
                "unknown dictionary primary POS {} for {}",
                self.primary_pos, self.id
            )));
        }
        if !SECONDARY_POS_SHORT.contains(&self.secondary_pos.as_str()) {
            return Err(invalid(format!(
                "unknown dictionary secondary POS {} for {}",
                self.secondary_pos, self.id
            )));
        }
        validate_sorted_known(&self.attributes, ROOT_ATTRIBUTES, "root attribute")?;
        if let Some(reference_id) = &self.reference_id {
            require_reference(dictionary_ids, reference_id, "dictionary reference")?;
        }
        let _ = (
            self.lemma.as_str(),
            self.root.as_str(),
            self.pronunciation.as_str(),
            self.index,
        );
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StemRecord {
    surface: String,
    source_order: usize,
    dictionary_id: String,
    target_state: String,
    target_zemberek_id: String,
    phonetic_bits: i32,
    phonetic_attributes: Vec<String>,
}

impl StemRecord {
    fn validate(
        &self,
        dictionary: &HashMap<&str, &DictionaryRecord>,
        states: &HashMap<&str, &StateRecord>,
    ) -> Result<(), BundleError> {
        if !dictionary.contains_key(self.dictionary_id.as_str()) {
            return Err(invalid(format!(
                "stem references missing dictionary item {}",
                self.dictionary_id
            )));
        }
        let state = states.get(self.target_state.as_str()).ok_or_else(|| {
            invalid(format!(
                "stem references missing state {}",
                self.target_state
            ))
        })?;
        if state.zemberek_id != self.target_zemberek_id {
            return Err(invalid(format!(
                "stem target textual id mismatch for {}",
                self.target_state
            )));
        }
        validate_unique(&self.phonetic_attributes, "phonetic attributes")?;
        let mut bits = 0_i32;
        for attribute in &self.phonetic_attributes {
            let ordinal = phonetic_ordinal(attribute)
                .ok_or_else(|| invalid(format!("unknown phonetic attribute {attribute}")))?;
            bits |= 1_i32
                .checked_shl(ordinal)
                .ok_or_else(|| invalid("phonetic bit shift overflow"))?;
        }
        if bits != self.phonetic_bits {
            return Err(invalid(format!(
                "stem phonetic bits {bits} != declared {} for {}",
                self.phonetic_bits, self.dictionary_id
            )));
        }
        let _ = (self.surface.as_str(), self.source_order);
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StateRecord {
    key: String,
    zemberek_id: String,
    morpheme_id: String,
    terminal: bool,
    derivative: bool,
    pos_root: bool,
    outgoing_count: usize,
    incoming_count: usize,
    declared_fields: Vec<String>,
}

impl StateRecord {
    fn validate(&self, morpheme_ids: &HashSet<&str>) -> Result<(), BundleError> {
        require_reference(morpheme_ids, &self.morpheme_id, "state morpheme")?;
        validate_sorted_unique_strings(&self.declared_fields, "state declared fields")?;
        if let Some(first) = self.declared_fields.first() {
            if first != &self.key {
                return Err(invalid(format!(
                    "state key {} differs from canonical field {first}",
                    self.key
                )));
            }
        } else if !self.key.starts_with("unbound:") {
            return Err(invalid(format!(
                "aliasless state has non-unbound key {}",
                self.key
            )));
        }
        let _ = (
            self.zemberek_id.as_str(),
            self.terminal,
            self.derivative,
            self.pos_root,
            self.incoming_count,
        );
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EdgeRecord {
    id: String,
    source_order: usize,
    owner_state: String,
    owner_zemberek_id: String,
    declared_from: String,
    declared_from_zemberek_id: String,
    owner_matches_declared_from: bool,
    to_state: String,
    to_zemberek_id: String,
    morpheme_id: String,
    surface_template: String,
    template_tokens: Vec<TemplateToken>,
    condition_count: usize,
    condition: Option<Condition>,
}

impl EdgeRecord {
    fn validate(
        &self,
        morphemes: &HashMap<&str, &MorphemeRecord>,
        dictionary: &HashMap<&str, &DictionaryRecord>,
        states: &HashMap<&str, &StateRecord>,
        condition_counts: &mut BTreeMap<&'static str, usize>,
    ) -> Result<(), BundleError> {
        let owner = require_state(states, &self.owner_state, "edge owner")?;
        let declared_from = require_state(states, &self.declared_from, "edge declared_from")?;
        let target = require_state(states, &self.to_state, "edge target")?;
        if owner.zemberek_id != self.owner_zemberek_id
            || declared_from.zemberek_id != self.declared_from_zemberek_id
            || target.zemberek_id != self.to_zemberek_id
        {
            return Err(invalid(format!(
                "edge {} textual state ID mismatch",
                self.id
            )));
        }
        if self.owner_matches_declared_from != (self.owner_state == self.declared_from) {
            return Err(invalid(format!(
                "edge {} owner match flag is inconsistent",
                self.id
            )));
        }
        if target.morpheme_id != self.morpheme_id {
            return Err(invalid(format!(
                "edge {} target morpheme mismatch",
                self.id
            )));
        }
        if !morphemes.contains_key(self.morpheme_id.as_str()) {
            return Err(invalid(format!(
                "edge {} references missing morpheme {}",
                self.id, self.morpheme_id
            )));
        }
        let _ = self.source_order;
        let reconstructed = reconstruct_template(&self.template_tokens)?;
        if reconstructed != self.surface_template {
            return Err(invalid(format!(
                "edge {} template {:?} != reconstructed {:?}",
                self.id, self.surface_template, reconstructed
            )));
        }
        let actual_condition_count = zemberek_condition_count(self.condition.as_ref());
        if actual_condition_count != self.condition_count {
            return Err(invalid(format!(
                "edge {} condition_count {} != recomputed {actual_condition_count}",
                self.id, self.condition_count
            )));
        }
        if let Some(condition) = &self.condition {
            condition.validate(morphemes, dictionary, states, condition_counts)?;
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TemplateToken {
    #[serde(rename = "type")]
    token_type: TemplateTokenType,
    letter: String,
    append: bool,
}

#[derive(Clone, Copy, Debug, Deserialize)]
enum TemplateTokenType {
    #[serde(rename = "I_WOVEL")]
    IVowel,
    #[serde(rename = "A_WOVEL")]
    AVowel,
    #[serde(rename = "DEVOICE_FIRST")]
    DevoiceFirst,
    #[serde(rename = "LAST_VOICED")]
    LastVoiced,
    #[serde(rename = "LAST_NOT_VOICED")]
    LastNotVoiced,
    #[serde(rename = "APPEND")]
    Append,
    #[serde(rename = "LETTER")]
    Letter,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
enum Condition {
    #[serde(rename = "AND")]
    And {
        children: Vec<Self>,
    },
    #[serde(rename = "OR")]
    Or {
        children: Vec<Self>,
    },
    #[serde(rename = "NOT")]
    Not {
        child: Box<Self>,
    },
    HasRootAttribute {
        attribute: String,
    },
    HasAnyRootAttribute {
        attributes: Vec<String>,
    },
    HasPhoneticAttribute {
        attribute: String,
    },
    DictionaryItemIs {
        item: String,
    },
    RootPrimaryPosIs {
        pos: String,
    },
    SecondaryPosIs {
        pos: String,
    },
    DictionaryItemIsAny {
        items: Vec<Option<String>>,
    },
    DictionaryItemIsNone {
        items: Vec<Option<String>>,
    },
    HasAnySuffixSurface,
    HasTail,
    HasNoTail,
    HasTailSequence {
        morphemes: Vec<String>,
    },
    ContainsMorphemeSequence {
        morphemes: Vec<String>,
    },
    CurrentMorphemeIs {
        morpheme: String,
    },
    PreviousMorphemeIs {
        morpheme: String,
    },
    PreviousStateIs {
        state: String,
    },
    PreviousStateIsNot {
        state: String,
    },
    RootSurfaceIs {
        surface: String,
    },
    RootSurfaceIsAny {
        surfaces: Vec<String>,
    },
    CurrentStateIs {
        state: String,
    },
    CurrentStateIsNot {
        state: String,
    },
    LastDerivationIs {
        state: String,
    },
    HasDerivation,
    LastDerivationIsAny {
        states: Vec<String>,
    },
    CurrentGroupContainsAny {
        states: Vec<String>,
    },
    PreviousGroupContains {
        states: Vec<String>,
    },
    PreviousGroupContainsMorpheme {
        morphemes: Vec<String>,
    },
    NoSurfaceAfterDerivation,
    ContainsMorpheme {
        morphemes: Vec<String>,
    },
    PreviousMorphemeIsAny {
        morphemes: Vec<String>,
    },
    CurrentMorphemeIsAny {
        morphemes: Vec<String>,
    },
    PreviousStateIsAny {
        states: Vec<String>,
    },
}

impl Condition {
    fn validate(
        &self,
        morphemes: &HashMap<&str, &MorphemeRecord>,
        dictionary: &HashMap<&str, &DictionaryRecord>,
        states: &HashMap<&str, &StateRecord>,
        counts: &mut BTreeMap<&'static str, usize>,
    ) -> Result<(), BundleError> {
        increment(counts, self.manifest_kind())?;
        match self {
            Self::And { children } | Self::Or { children } => {
                if children.len() < 2 {
                    return Err(invalid("combined condition has fewer than two children"));
                }
                for child in children {
                    child.validate(morphemes, dictionary, states, counts)?;
                }
            }
            Self::Not { child } => child.validate(morphemes, dictionary, states, counts)?,
            Self::HasRootAttribute { attribute } => {
                require_known(ROOT_ATTRIBUTES, attribute, "root attribute")?;
            }
            Self::HasAnyRootAttribute { attributes } => {
                for attribute in attributes {
                    require_known(ROOT_ATTRIBUTES, attribute, "root attribute")?;
                }
            }
            Self::HasPhoneticAttribute { attribute } => {
                if phonetic_ordinal(attribute).is_none() {
                    return Err(invalid(format!("unknown phonetic attribute {attribute}")));
                }
            }
            Self::DictionaryItemIs { item } => {
                require_map_reference(dictionary, item, "condition dictionary item")?;
            }
            Self::RootPrimaryPosIs { pos } => {
                require_known(PRIMARY_POS_NAMES, pos, "primary POS")?;
            }
            Self::SecondaryPosIs { pos } => {
                require_known(SECONDARY_POS_NAMES, pos, "secondary POS")?;
            }
            Self::DictionaryItemIsAny { items } | Self::DictionaryItemIsNone { items } => {
                validate_nullable_dictionary_set(items, dictionary)?;
            }
            Self::HasTailSequence { morphemes: values }
            | Self::ContainsMorphemeSequence { morphemes: values } => {
                for value in values {
                    require_map_reference(morphemes, value, "condition morpheme")?;
                }
            }
            Self::CurrentMorphemeIs { morpheme } | Self::PreviousMorphemeIs { morpheme } => {
                require_map_reference(morphemes, morpheme, "condition morpheme")?;
            }
            Self::PreviousStateIs { state }
            | Self::PreviousStateIsNot { state }
            | Self::CurrentStateIs { state }
            | Self::CurrentStateIsNot { state }
            | Self::LastDerivationIs { state } => {
                require_map_reference(states, state, "condition state")?;
            }
            Self::RootSurfaceIs { surface } => {
                let _ = surface.as_str();
            }
            Self::RootSurfaceIsAny { surfaces } => {
                let _ = surfaces.as_slice();
            }
            Self::LastDerivationIsAny { states: values }
            | Self::CurrentGroupContainsAny { states: values }
            | Self::PreviousGroupContains { states: values }
            | Self::PreviousStateIsAny { states: values } => {
                validate_sorted_unique_strings(values, "condition state set")?;
                for value in values {
                    require_map_reference(states, value, "condition state")?;
                }
            }
            Self::PreviousGroupContainsMorpheme { morphemes: values }
            | Self::ContainsMorpheme { morphemes: values }
            | Self::PreviousMorphemeIsAny { morphemes: values }
            | Self::CurrentMorphemeIsAny { morphemes: values } => {
                validate_sorted_unique_strings(values, "condition morpheme set")?;
                for value in values {
                    require_map_reference(morphemes, value, "condition morpheme")?;
                }
            }
            Self::HasAnySuffixSurface
            | Self::HasTail
            | Self::HasNoTail
            | Self::HasDerivation
            | Self::NoSurfaceAfterDerivation => {}
        }
        Ok(())
    }

    const fn manifest_kind(&self) -> &'static str {
        match self {
            Self::And { .. } | Self::Or { .. } => "CombinedCondition",
            Self::Not { .. } => "NotCondition",
            Self::HasRootAttribute { .. } => "HasRootAttribute",
            Self::HasAnyRootAttribute { .. } => "HasAnyRootAttribute",
            Self::HasPhoneticAttribute { .. } => "HasPhoneticAttribute",
            Self::DictionaryItemIs { .. } => "DictionaryItemIs",
            Self::RootPrimaryPosIs { .. } => "RootPrimaryPosIs",
            Self::SecondaryPosIs { .. } => "SecondaryPosIs",
            Self::DictionaryItemIsAny { .. } => "DictionaryItemIsAny",
            Self::DictionaryItemIsNone { .. } => "DictionaryItemIsNone",
            Self::HasAnySuffixSurface => "HasAnySuffixSurface",
            Self::HasTail => "HasTail",
            Self::HasNoTail => "HasNoTail",
            Self::HasTailSequence { .. } => "HasTailSequence",
            Self::ContainsMorphemeSequence { .. } => "ContainsMorphemeSequence",
            Self::CurrentMorphemeIs { .. } => "CurrentMorphemeIs",
            Self::PreviousMorphemeIs { .. } => "PreviousMorphemeIs",
            Self::PreviousStateIs { .. } => "PreviousStateIs",
            Self::PreviousStateIsNot { .. } => "PreviousStateIsNot",
            Self::RootSurfaceIs { .. } => "RootSurfaceIs",
            Self::RootSurfaceIsAny { .. } => "RootSurfaceIsAny",
            Self::CurrentStateIs { .. } => "CurrentStateIs",
            Self::CurrentStateIsNot { .. } => "CurrentStateIsNot",
            Self::LastDerivationIs { .. } => "LastDerivationIs",
            Self::HasDerivation => "HasDerivation",
            Self::LastDerivationIsAny { .. } => "LastDerivationIsAny",
            Self::CurrentGroupContainsAny { .. } => "CurrentGroupContainsAny",
            Self::PreviousGroupContains { .. } => "PreviousGroupContains",
            Self::PreviousGroupContainsMorpheme { .. } => "PreviousGroupContainsMorpheme",
            Self::NoSurfaceAfterDerivation => "NoSurfaceAfterDerivation",
            Self::ContainsMorpheme { .. } => "ContainsMorpheme",
            Self::PreviousMorphemeIsAny { .. } => "PreviousMorphemeIsAny",
            Self::CurrentMorphemeIsAny { .. } => "CurrentMorphemeIsAny",
            Self::PreviousStateIsAny { .. } => "PreviousStateIsAny",
        }
    }
}

fn zemberek_condition_count(condition: Option<&Condition>) -> usize {
    match condition {
        None => 0,
        Some(Condition::And { children } | Condition::Or { children }) => children
            .iter()
            .map(|child| match child {
                Condition::And { .. } | Condition::Or { .. } => {
                    zemberek_condition_count(Some(child))
                }
                _ => 1,
            })
            .sum(),
        Some(_) => 1,
    }
}

fn reconstruct_template(tokens: &[TemplateToken]) -> Result<String, BundleError> {
    let mut output = String::new();
    for token in tokens {
        match token.token_type {
            TemplateTokenType::IVowel => {
                require_empty_letter(token)?;
                output.push_str(if token.append { "+I" } else { "I" });
            }
            TemplateTokenType::AVowel => {
                require_empty_letter(token)?;
                output.push_str(if token.append { "+A" } else { "A" });
            }
            TemplateTokenType::Append => {
                require_not_append_flag(token)?;
                output.push('+');
                output.push(single_letter(token)?);
            }
            TemplateTokenType::DevoiceFirst => {
                require_not_append_flag(token)?;
                output.push('>');
                output.push(single_letter(token)?);
            }
            TemplateTokenType::LastVoiced => {
                require_not_append_flag(token)?;
                output.push('~');
                output.push(single_letter(token)?);
            }
            TemplateTokenType::LastNotVoiced => {
                require_not_append_flag(token)?;
                output.push('!');
                output.push(single_letter(token)?);
            }
            TemplateTokenType::Letter => {
                require_not_append_flag(token)?;
                output.push(single_letter(token)?);
            }
        }
    }
    Ok(output)
}

fn single_letter(token: &TemplateToken) -> Result<char, BundleError> {
    let mut characters = token.letter.chars();
    let first = characters
        .next()
        .ok_or_else(|| invalid("template token requires one letter"))?;
    if characters.next().is_some() {
        return Err(invalid("template token contains multiple letters"));
    }
    Ok(first)
}

fn require_empty_letter(token: &TemplateToken) -> Result<(), BundleError> {
    if token.letter.is_empty() {
        Ok(())
    } else {
        Err(invalid("vowel template token contains a literal letter"))
    }
}

fn require_not_append_flag(token: &TemplateToken) -> Result<(), BundleError> {
    if token.append {
        Err(invalid("non-vowel template token has append=true"))
    } else {
        Ok(())
    }
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, BundleError> {
    let file = File::open(path).map_err(|error| io_error(path, &error))?;
    serde_json::from_reader(BufReader::new(file)).map_err(|error| BundleError::Json {
        path: path.to_path_buf(),
        line: error.line(),
        message: error.to_string(),
    })
}

fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>, BundleError> {
    let file = File::open(path).map_err(|error| io_error(path, &error))?;
    let mut records = Vec::new();
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line_number = line_index + 1;
        let line = line.map_err(|error| io_error(path, &error))?;
        if line.is_empty() {
            return Err(BundleError::Json {
                path: path.to_path_buf(),
                line: line_number,
                message: "empty JSONL line".to_owned(),
            });
        }
        let record = serde_json::from_str(&line).map_err(|error| BundleError::Json {
            path: path.to_path_buf(),
            line: line_number,
            message: error.to_string(),
        })?;
        records.push(record);
    }
    Ok(records)
}

fn validate_sorted_unique<'a, T, F>(
    records: &'a [T],
    key: F,
    label: &str,
) -> Result<HashSet<&'a str>, BundleError>
where
    F: Fn(&'a T) -> &'a str,
{
    let mut result = HashSet::with_capacity(records.len());
    let mut previous = None;
    for record in records {
        let value = key(record);
        if previous.is_some_and(|prior| prior >= value) {
            return Err(invalid(format!(
                "{label} is not strictly sorted at {value}"
            )));
        }
        previous = Some(value);
        if !result.insert(value) {
            return Err(invalid(format!("duplicate {label}: {value}")));
        }
    }
    Ok(result)
}

fn validate_sequential_edge_ids(edges: &[EdgeRecord]) -> Result<(), BundleError> {
    for (index, edge) in edges.iter().enumerate() {
        let expected = format!("e{index:06}");
        if edge.id != expected {
            return Err(invalid(format!(
                "edge id {} != expected {expected}",
                edge.id
            )));
        }
    }
    Ok(())
}

fn validate_nullable_dictionary_set<T>(
    values: &[Option<String>],
    dictionary: &HashMap<&str, T>,
) -> Result<(), BundleError> {
    let mut previous: Option<&str> = None;
    let mut saw_null = false;
    for value in values {
        if let Some(item) = value {
            if saw_null {
                return Err(invalid(
                    "condition dictionary null sentinel must be the final element",
                ));
            }
            if previous.is_some_and(|prior| prior >= item.as_str()) {
                return Err(invalid(
                    "condition dictionary item set is not strictly sorted",
                ));
            }
            require_map_reference(dictionary, item, "condition dictionary item")?;
            previous = Some(item);
        } else {
            if saw_null {
                return Err(invalid(
                    "condition dictionary item set contains multiple null sentinels",
                ));
            }
            saw_null = true;
        }
    }
    Ok(())
}

fn validate_sorted_unique_strings(values: &[String], label: &str) -> Result<(), BundleError> {
    for pair in values.windows(2) {
        if pair[0] >= pair[1] {
            return Err(invalid(format!("{label} is not strictly sorted")));
        }
    }
    Ok(())
}

fn validate_unique(values: &[String], label: &str) -> Result<(), BundleError> {
    let mut seen = HashSet::with_capacity(values.len());
    for value in values {
        if !seen.insert(value) {
            return Err(invalid(format!("duplicate {label}: {value}")));
        }
    }
    Ok(())
}

fn validate_sorted_known(
    values: &[String],
    known: &[&str],
    label: &str,
) -> Result<(), BundleError> {
    validate_sorted_unique_strings(values, label)?;
    for value in values {
        require_known(known, value, label)?;
    }
    Ok(())
}

fn require_known(known: &[&str], value: &str, label: &str) -> Result<(), BundleError> {
    if known.contains(&value) {
        Ok(())
    } else {
        Err(invalid(format!("unknown {label}: {value}")))
    }
}

fn require_reference(known: &HashSet<&str>, value: &str, label: &str) -> Result<(), BundleError> {
    if known.contains(value) {
        Ok(())
    } else {
        Err(invalid(format!("missing {label}: {value}")))
    }
}

fn require_map_reference<T>(
    known: &HashMap<&str, T>,
    value: &str,
    label: &str,
) -> Result<(), BundleError> {
    if known.contains_key(value) {
        Ok(())
    } else {
        Err(invalid(format!("missing {label}: {value}")))
    }
}

fn require_state<'a>(
    states: &'a HashMap<&str, &StateRecord>,
    key: &str,
    label: &str,
) -> Result<&'a StateRecord, BundleError> {
    states
        .get(key)
        .copied()
        .ok_or_else(|| invalid(format!("missing {label}: {key}")))
}

fn duplicate_excess<'a>(values: impl Iterator<Item = &'a str>) -> Result<usize, BundleError> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for value in values {
        let count = counts.entry(value).or_insert(0);
        *count = count
            .checked_add(1)
            .ok_or_else(|| invalid("duplicate count overflow"))?;
    }
    counts.values().try_fold(0_usize, |total, count| {
        total
            .checked_add(count.saturating_sub(1))
            .ok_or_else(|| invalid("duplicate excess overflow"))
    })
}

fn increment(
    counts: &mut BTreeMap<&'static str, usize>,
    kind: &'static str,
) -> Result<(), BundleError> {
    let count = counts.entry(kind).or_insert(0);
    *count = count
        .checked_add(1)
        .ok_or_else(|| invalid("condition count overflow"))?;
    Ok(())
}

fn known_condition_manifest_kind(value: &str) -> Option<&'static str> {
    CONDITION_MANIFEST_KINDS
        .iter()
        .copied()
        .find(|known| *known == value)
}

fn phonetic_ordinal(value: &str) -> Option<u32> {
    PHONETIC_ATTRIBUTES
        .iter()
        .position(|known| *known == value)
        .and_then(|index| u32::try_from(index).ok())
}

fn io_error(path: &Path, error: &std::io::Error) -> BundleError {
    BundleError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    }
}

fn invalid(message: impl Into<String>) -> BundleError {
    BundleError::Invalid {
        message: message.into(),
    }
}

/// Bundle load or semantic validation error.
#[derive(Debug)]
pub enum BundleError {
    /// File I/O failed.
    Io { path: PathBuf, message: String },
    /// JSON parsing or strict schema validation failed.
    Json {
        path: PathBuf,
        line: usize,
        message: String,
    },
    /// Parsed data violated a bundle invariant.
    Invalid { message: String },
}

impl fmt::Display for BundleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for BundleError {}

const ROOT_ATTRIBUTES: &[&str] = &[
    "Aorist_I",
    "Aorist_A",
    "ProgressiveVowelDrop",
    "Passive_In",
    "Causative_t",
    "Voicing",
    "NoVoicing",
    "InverseHarmony",
    "Doubling",
    "LastVowelDrop",
    "CompoundP3sg",
    "NoSuffix",
    "NounConsInsert_n",
    "NoQuote",
    "CompoundP3sgRoot",
    "Reflexive",
    "Reciprocal",
    "NonReciprocal",
    "Ext",
    "Runtime",
    "Dummy",
    "ImplicitDative",
    "ImplicitPlural",
    "ImplicitP1sg",
    "ImplicitP2sg",
    "FamilyMember",
    "PronunciationGuessed",
    "Informal",
    "LocaleEn",
];

const PHONETIC_ATTRIBUTES: &[&str] = &[
    "LastLetterVowel",
    "LastLetterConsonant",
    "LastVowelFrontal",
    "LastVowelBack",
    "LastVowelRounded",
    "LastVowelUnrounded",
    "LastLetterVoiceless",
    "LastLetterVoiced",
    "LastLetterVoicelessStop",
    "FirstLetterVowel",
    "FirstLetterConsonant",
    "HasNoVowel",
    "ExpectsVowel",
    "ExpectsConsonant",
    "ModifiedPronoun",
    "UnModifiedPronoun",
    "LastLetterDropped",
    "CannotTerminate",
];

const PRIMARY_POS_SHORT: &[&str] = &[
    "Noun", "Adj", "Adv", "Conj", "Interj", "Verb", "Pron", "Num", "Det", "Postp", "Ques", "Dup",
    "Punc", "Unk",
];

const PRIMARY_POS_NAMES: &[&str] = &[
    "Noun",
    "Adjective",
    "Adverb",
    "Conjunction",
    "Interjection",
    "Verb",
    "Pronoun",
    "Numeral",
    "Determiner",
    "PostPositive",
    "Question",
    "Duplicator",
    "Punctuation",
    "Unknown",
];

const SECONDARY_POS_SHORT: &[&str] = &[
    "Unk",
    "Demons",
    "Time",
    "Quant",
    "Ques",
    "Prop",
    "Pers",
    "Reflex",
    "None",
    "Ord",
    "Card",
    "Percent",
    "Ratio",
    "Range",
    "Real",
    "Dist",
    "Clock",
    "Date",
    "Email",
    "Url",
    "Mention",
    "HashTag",
    "Emoticon",
    "RomanNumeral",
    "RegAbbrv",
    "Abbrv",
    "PCDat",
    "PCAcc",
    "PCIns",
    "PCNom",
    "PCGen",
    "PCAbl",
];

const SECONDARY_POS_NAMES: &[&str] = &[
    "UnknownSec",
    "DemonstrativePron",
    "Time",
    "QuantitivePron",
    "QuestionPron",
    "ProperNoun",
    "PersonalPron",
    "ReflexivePron",
    "None",
    "Ordinal",
    "Cardinal",
    "Percentage",
    "Ratio",
    "Range",
    "Real",
    "Distribution",
    "Clock",
    "Date",
    "Email",
    "Url",
    "Mention",
    "HashTag",
    "Emoticon",
    "RomanNumeral",
    "RegularAbbreviation",
    "Abbreviation",
    "PCDat",
    "PCAcc",
    "PCIns",
    "PCNom",
    "PCGen",
    "PCAbl",
];

const CONDITION_MANIFEST_KINDS: &[&str] = &[
    "CombinedCondition",
    "ContainsMorpheme",
    "ContainsMorphemeSequence",
    "CurrentGroupContainsAny",
    "CurrentMorphemeIs",
    "CurrentMorphemeIsAny",
    "CurrentStateIs",
    "CurrentStateIsNot",
    "DictionaryItemIs",
    "DictionaryItemIsAny",
    "DictionaryItemIsNone",
    "HasAnyRootAttribute",
    "HasAnySuffixSurface",
    "HasDerivation",
    "HasNoTail",
    "HasPhoneticAttribute",
    "HasRootAttribute",
    "HasTail",
    "HasTailSequence",
    "LastDerivationIs",
    "LastDerivationIsAny",
    "NoSurfaceAfterDerivation",
    "NotCondition",
    "PreviousGroupContains",
    "PreviousGroupContainsMorpheme",
    "PreviousMorphemeIs",
    "PreviousMorphemeIsAny",
    "PreviousStateIs",
    "PreviousStateIsAny",
    "PreviousStateIsNot",
    "RootPrimaryPosIs",
    "RootSurfaceIs",
    "RootSurfaceIsAny",
    "SecondaryPosIs",
];

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{
        reconstruct_template, validate_nullable_dictionary_set, zemberek_condition_count,
        Condition, TemplateToken, TemplateTokenType,
    };

    #[test]
    fn reconstructs_suffix_template_tokens() -> Result<(), super::BundleError> {
        let tokens = vec![
            TemplateToken {
                token_type: TemplateTokenType::Append,
                letter: "y".to_owned(),
                append: false,
            },
            TemplateToken {
                token_type: TemplateTokenType::DevoiceFirst,
                letter: "d".to_owned(),
                append: false,
            },
            TemplateToken {
                token_type: TemplateTokenType::IVowel,
                letter: String::new(),
                append: false,
            },
        ];
        let reconstructed = reconstruct_template(&tokens)?;
        assert_eq!(reconstructed, "+y>dI");
        Ok(())
    }

    #[test]
    fn accepts_one_trailing_dictionary_null_sentinel() -> Result<(), serde_json::Error> {
        let condition: Condition =
            serde_json::from_str(r#"{"kind":"DictionaryItemIsAny","items":["a_Noun",null]}"#)?;
        assert!(matches!(
            condition,
            Condition::DictionaryItemIsAny { ref items }
                if items == &vec![Some("a_Noun".to_owned()), None]
        ));
        Ok(())
    }

    #[test]
    fn rejects_non_trailing_or_duplicate_null_sentinels() {
        let dictionary = HashMap::from([("a_Noun", 1_u8)]);
        let non_trailing = vec![None, Some("a_Noun".to_owned())];
        let duplicate = vec![Some("a_Noun".to_owned()), None, None];
        assert!(validate_nullable_dictionary_set(&non_trailing, &dictionary).is_err());
        assert!(validate_nullable_dictionary_set(&duplicate, &dictionary).is_err());
    }

    #[test]
    fn rejects_unknown_condition_kind() {
        let value = r#"{"kind":"UnknownCondition"}"#;
        assert!(serde_json::from_str::<Condition>(value).is_err());
    }

    #[test]
    fn matches_zemberek_top_level_condition_count_semantics() -> Result<(), serde_json::Error> {
        let condition: Condition = serde_json::from_str(
            r#"{"kind":"NOT","child":{"kind":"AND","children":[{"kind":"HasTail"},{"kind":"HasNoTail"}]}}"#,
        )?;
        assert_eq!(zemberek_condition_count(Some(&condition)), 1);

        let combined: Condition = serde_json::from_str(
            r#"{"kind":"AND","children":[{"kind":"HasTail"},{"kind":"OR","children":[{"kind":"HasNoTail"},{"kind":"HasDerivation"}]}]}"#,
        )?;
        assert_eq!(zemberek_condition_count(Some(&combined)), 3);
        Ok(())
    }
}
