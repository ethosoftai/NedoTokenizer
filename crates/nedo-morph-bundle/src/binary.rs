//! Stable mmap-oriented binary compiler and corruption-safe zero-copy validator.

use core::fmt;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use sha2::{Digest, Sha256};

use super::{
    phonetic_ordinal, BundleError, Condition, EdgeRecord, MorphBundle, StemRecord, TemplateToken,
    TemplateTokenType, PHONETIC_ATTRIBUTES, PRIMARY_POS_NAMES, PRIMARY_POS_SHORT, ROOT_ATTRIBUTES,
    SECONDARY_POS_NAMES, SECONDARY_POS_SHORT, ZEMBEREK_COMMIT,
};

mod runtime;

pub use runtime::{
    AnalysisLimits, NativeAnalysis, NativeGenerator, NativeMorpheme, NativeMorphology, NativeStem,
    StemMatches,
};

const MAGIC: &[u8; 8] = b"NDMORF01";
const ENDIAN_MARKER: u32 = 0x0102_0304;
const BINARY_SCHEMA_VERSION: u32 = 1;
const HEADER_SIZE: usize = 256;
const SECTION_COUNT: usize = 10;
const DIRECTORY_OFFSET: usize = 128;
const DIRECTORY_ENTRY_SIZE: usize = 12;
const PAYLOAD_HASH_OFFSET: usize = 48;
const FILE_LENGTH_OFFSET: usize = 80;
const COUNTS_OFFSET: usize = 88;
const SOURCE_COMMIT_OFFSET: usize = 24;
const NONE_U32: u32 = u32::MAX;
const NONE_U16: u16 = u16::MAX;

const MORPHEME_RECORD_SIZE: usize = 24;
const DICTIONARY_RECORD_SIZE: usize = 40;
const STEM_RECORD_SIZE: usize = 16;
const STATE_RECORD_SIZE: usize = 36;
const EDGE_RECORD_SIZE: usize = 36;
const TEMPLATE_RECORD_SIZE: usize = 8;

#[derive(Clone, Copy, Debug)]
#[repr(usize)]
enum Section {
    StringOffsets = 0,
    StringBlob = 1,
    Morphemes = 2,
    Dictionary = 3,
    Stems = 4,
    States = 5,
    Edges = 6,
    Aliases = 7,
    Templates = 8,
    Conditions = 9,
}

impl Section {
    const ALL: [Self; SECTION_COUNT] = [
        Self::StringOffsets,
        Self::StringBlob,
        Self::Morphemes,
        Self::Dictionary,
        Self::Stems,
        Self::States,
        Self::Edges,
        Self::Aliases,
        Self::Templates,
        Self::Conditions,
    ];
}

/// Compiles one validated reference bundle into the stable native binary format.
///
/// # Errors
///
/// Returns an error when a count or offset exceeds the binary schema, a source
/// reference cannot be resolved, or an input semantic value is unsupported.
pub fn compile_binary(bundle: &MorphBundle) -> Result<Vec<u8>, BinaryError> {
    bundle.validate().map_err(BinaryError::SourceBundle)?;
    let mut compiler = Compiler::new(bundle)?;
    compiler.compile()
}

/// A validated zero-copy view over native morphology bytes.
#[derive(Clone, Copy)]
pub struct BinaryBundleView<'a> {
    bytes: &'a [u8],
    header: BinaryHeader,
}

impl<'a> BinaryBundleView<'a> {
    /// Parses and validates a complete native morphology binary.
    ///
    /// # Errors
    ///
    /// Returns an error for bad identity, checksum mismatch, malformed section
    /// layout, invalid UTF-8 strings, out-of-range references, or corrupt
    /// template/condition bytecode.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, BinaryError> {
        let header = BinaryHeader::parse(bytes)?;
        let view = Self { bytes, header };
        view.validate()?;
        Ok(view)
    }

    /// Returns validated high-level counts.
    #[must_use]
    pub const fn summary(self) -> BinarySummary {
        BinarySummary {
            string_count: self.header.counts[0] as usize,
            morpheme_count: self.header.counts[1] as usize,
            dictionary_count: self.header.counts[2] as usize,
            stem_count: self.header.counts[3] as usize,
            state_count: self.header.counts[4] as usize,
            edge_count: self.header.counts[5] as usize,
            alias_count: self.header.counts[6] as usize,
            template_token_count: self.header.counts[7] as usize,
            condition_byte_count: self.header.counts[8] as usize,
            null_dictionary_sentinel_count: self.header.counts[9] as usize,
            file_byte_count: self.bytes.len(),
        }
    }

    /// Returns the underlying validated bytes.
    #[must_use]
    pub const fn as_bytes(self) -> &'a [u8] {
        self.bytes
    }

    fn validate(self) -> Result<(), BinaryError> {
        self.validate_payload_hash()?;
        self.validate_sections()?;
        let strings = StringTable::parse(self)?;
        self.validate_morphemes(&strings)?;
        self.validate_dictionary(&strings)?;
        self.validate_stems(&strings)?;
        self.validate_states(&strings)?;
        self.validate_edges(&strings)?;
        Ok(())
    }

    fn validate_payload_hash(self) -> Result<(), BinaryError> {
        let actual: [u8; 32] = Sha256::digest(&self.bytes[HEADER_SIZE..]).into();
        if actual != self.header.payload_hash {
            return Err(invalid("payload SHA-256 mismatch"));
        }
        Ok(())
    }

    fn validate_sections(self) -> Result<(), BinaryError> {
        let mut previous_end = HEADER_SIZE;
        for section in Section::ALL {
            let range = self.section_range(section)?;
            if range.start < previous_end {
                return Err(invalid(format!(
                    "section {section:?} overlaps or is out of order"
                )));
            }
            if !self.bytes[previous_end..range.start]
                .iter()
                .all(|byte| *byte == 0)
            {
                return Err(invalid(format!(
                    "section {section:?} has non-zero alignment padding"
                )));
            }
            previous_end = range.end;
        }
        if previous_end != self.bytes.len() {
            return Err(invalid("trailing bytes after final section"));
        }

        let expected_lengths = [
            checked_mul(
                checked_add(self.header.counts[0] as usize, 1, "string offset count")?,
                4,
                "string offsets length",
            )?,
            self.header.sections[Section::StringBlob as usize].length as usize,
            checked_mul(
                self.header.counts[1] as usize,
                MORPHEME_RECORD_SIZE,
                "morpheme section length",
            )?,
            checked_mul(
                self.header.counts[2] as usize,
                DICTIONARY_RECORD_SIZE,
                "dictionary section length",
            )?,
            checked_mul(
                self.header.counts[3] as usize,
                STEM_RECORD_SIZE,
                "stem section length",
            )?,
            checked_mul(
                self.header.counts[4] as usize,
                STATE_RECORD_SIZE,
                "state section length",
            )?,
            checked_mul(
                self.header.counts[5] as usize,
                EDGE_RECORD_SIZE,
                "edge section length",
            )?,
            checked_mul(self.header.counts[6] as usize, 4, "alias section length")?,
            checked_mul(
                self.header.counts[7] as usize,
                TEMPLATE_RECORD_SIZE,
                "template section length",
            )?,
            self.header.counts[8] as usize,
        ];
        for (index, expected) in expected_lengths.into_iter().enumerate() {
            let actual = self.header.sections[index].length as usize;
            if actual != expected {
                return Err(invalid(format!(
                    "section {index} length {actual} != expected {expected}"
                )));
            }
        }
        Ok(())
    }

    fn validate_morphemes(self, strings: &StringTable<'a>) -> Result<(), BinaryError> {
        let section = self.section(Section::Morphemes)?;
        let count = self.header.counts[1] as usize;
        let mut previous_id = None;
        for index in 0..count {
            let record = fixed_record(section, index, MORPHEME_RECORD_SIZE)?;
            let id = strings.get(read_u32(record, 0)?)?;
            strings.get(read_u32(record, 4)?)?;
            let pos = read_u16(record, 8)?;
            if pos != NONE_U16 && usize::from(pos) >= PRIMARY_POS_SHORT.len() {
                return Err(invalid(format!("morpheme {index} has invalid POS {pos}")));
            }
            let flags = read_u16(record, 10)?;
            if flags & !0b11 != 0 {
                return Err(invalid(format!("morpheme {index} has unknown flags")));
            }
            let mapped = read_u32(record, 12)?;
            require_optional_index(mapped, count, "mapped morpheme")?;
            require_zero(&record[16..24], "morpheme reserved bytes")?;
            require_strict_order(previous_id, id, "binary morpheme IDs")?;
            previous_id = Some(id);
        }
        Ok(())
    }

    fn validate_dictionary(self, strings: &StringTable<'a>) -> Result<(), BinaryError> {
        let section = self.section(Section::Dictionary)?;
        let count = self.header.counts[2] as usize;
        let mut previous_id = None;
        let valid_attribute_bits = if ROOT_ATTRIBUTES.len() == u32::BITS as usize {
            u32::MAX
        } else {
            (1_u32 << ROOT_ATTRIBUTES.len()) - 1
        };
        for index in 0..count {
            let record = fixed_record(section, index, DICTIONARY_RECORD_SIZE)?;
            let id = strings.get(read_u32(record, 0)?)?;
            strings.get(read_u32(record, 4)?)?;
            strings.get(read_u32(record, 8)?)?;
            strings.get(read_u32(record, 12)?)?;
            let primary = read_u16(record, 16)?;
            let secondary = read_u16(record, 18)?;
            if usize::from(primary) >= PRIMARY_POS_SHORT.len()
                || usize::from(secondary) >= SECONDARY_POS_SHORT.len()
            {
                return Err(invalid(format!(
                    "dictionary record {index} has invalid POS"
                )));
            }
            if read_u32(record, 20)? & !valid_attribute_bits != 0 {
                return Err(invalid(format!(
                    "dictionary record {index} has unknown root attribute bits"
                )));
            }
            require_optional_index(read_u32(record, 24)?, count, "dictionary reference")?;
            let _source_index = read_i32(record, 28)?;
            require_zero(&record[32..40], "dictionary reserved bytes")?;
            require_strict_order(previous_id, id, "binary dictionary IDs")?;
            previous_id = Some(id);
        }
        Ok(())
    }

    fn validate_stems(self, strings: &StringTable<'a>) -> Result<(), BinaryError> {
        let section = self.section(Section::Stems)?;
        let count = self.header.counts[3] as usize;
        let dictionary_count = self.header.counts[2] as usize;
        let state_count = self.header.counts[4] as usize;
        let valid_phonetic_bits = if PHONETIC_ATTRIBUTES.len() == u32::BITS as usize {
            u32::MAX
        } else {
            (1_u32 << PHONETIC_ATTRIBUTES.len()) - 1
        };
        let mut previous_surface = None;
        let mut seen = std::collections::HashSet::with_capacity(count);
        for index in 0..count {
            let record = fixed_record(section, index, STEM_RECORD_SIZE)?;
            let surface = read_u32(record, 0)?;
            strings.get(surface)?;
            let dictionary = read_u32(record, 4)?;
            let state = read_u32(record, 8)?;
            let phonetic_bits = read_u32(record, 12)?;
            require_index(dictionary, dictionary_count, "stem dictionary")?;
            require_index(state, state_count, "stem state")?;
            if phonetic_bits & !valid_phonetic_bits != 0 {
                return Err(invalid(format!("stem {index} has unknown phonetic bits")));
            }
            if previous_surface.is_some_and(|prior| prior > surface) {
                return Err(invalid(format!("stem surfaces are not sorted at {index}")));
            }
            let key = (surface, dictionary, state, phonetic_bits);
            if !seen.insert(key) {
                return Err(invalid(format!("duplicate stem record at {index}")));
            }
            previous_surface = Some(surface);
        }
        Ok(())
    }

    fn validate_states(self, strings: &StringTable<'a>) -> Result<(), BinaryError> {
        let section = self.section(Section::States)?;
        let count = self.header.counts[4] as usize;
        let morpheme_count = self.header.counts[1] as usize;
        let edge_count = self.header.counts[5] as usize;
        let alias_count = self.header.counts[6] as usize;
        let aliases = self.section(Section::Aliases)?;
        let mut previous_key = None;
        let mut expected_edge_start = 0_usize;
        let mut expected_alias_start = 0_usize;
        for index in 0..count {
            let record = fixed_record(section, index, STATE_RECORD_SIZE)?;
            let key = strings.get(read_u32(record, 0)?)?;
            strings.get(read_u32(record, 4)?)?;
            require_index(read_u32(record, 8)?, morpheme_count, "state morpheme")?;
            let flags = read_u32(record, 12)?;
            if flags & !0b111 != 0 {
                return Err(invalid(format!("state {index} has unknown flags")));
            }
            let edge_start = read_u32(record, 16)? as usize;
            let outgoing = read_u32(record, 20)? as usize;
            let _incoming = read_u32(record, 24)? as usize;
            let alias_start = read_u32(record, 28)? as usize;
            let aliases_for_state = read_u32(record, 32)? as usize;
            if edge_start != expected_edge_start
                || checked_add(edge_start, outgoing, "state edge range")? > edge_count
            {
                return Err(invalid(format!("state {index} has invalid edge range")));
            }
            expected_edge_start = edge_start + outgoing;
            if alias_start != expected_alias_start
                || checked_add(alias_start, aliases_for_state, "state alias range")? > alias_count
            {
                return Err(invalid(format!("state {index} has invalid alias range")));
            }
            let mut previous_alias = None;
            for alias_index in alias_start..alias_start + aliases_for_state {
                let string_id = read_u32(aliases, alias_index * 4)?;
                let alias = strings.get(string_id)?;
                require_strict_order(previous_alias, alias, "state aliases")?;
                previous_alias = Some(alias);
            }
            if let Some(first_alias) =
                previous_first_alias(aliases, alias_start, aliases_for_state, strings)?
            {
                if first_alias != key {
                    return Err(invalid(format!(
                        "state {index} key differs from canonical alias"
                    )));
                }
            } else if !key.starts_with("unbound:") {
                return Err(invalid(format!(
                    "state {index} has no alias and non-unbound key"
                )));
            }
            expected_alias_start = alias_start + aliases_for_state;
            require_strict_order(previous_key, key, "binary state keys")?;
            previous_key = Some(key);
        }
        if expected_edge_start != edge_count || expected_alias_start != alias_count {
            return Err(invalid("state ranges do not cover edge or alias sections"));
        }
        Ok(())
    }

    fn validate_edges(self, strings: &StringTable<'a>) -> Result<(), BinaryError> {
        let section = self.section(Section::Edges)?;
        let count = self.header.counts[5] as usize;
        let state_count = self.header.counts[4] as usize;
        let morpheme_count = self.header.counts[1] as usize;
        let template_count = self.header.counts[7] as usize;
        let condition_bytes = self.section(Section::Conditions)?;
        let templates = self.section(Section::Templates)?;
        let mut null_sentinels = 0_usize;
        for index in 0..count {
            let record = fixed_record(section, index, EDGE_RECORD_SIZE)?;
            require_index(read_u32(record, 0)?, state_count, "edge declared_from")?;
            require_index(read_u32(record, 4)?, state_count, "edge target")?;
            require_index(read_u32(record, 8)?, morpheme_count, "edge morpheme")?;
            let template_string = strings.get(read_u32(record, 12)?)?;
            let template_start = read_u32(record, 16)? as usize;
            let condition_start = read_u32(record, 20)? as usize;
            let condition_length = read_u32(record, 24)? as usize;
            let template_length = read_u16(record, 28)? as usize;
            let condition_count = read_u16(record, 30)? as usize;
            let flags = read_u32(record, 32)?;
            if flags & !1 != 0 {
                return Err(invalid(format!("edge {index} has unknown flags")));
            }
            if checked_add(template_start, template_length, "edge template range")? > template_count
            {
                return Err(invalid(format!("edge {index} has invalid template range")));
            }
            let reconstructed =
                reconstruct_binary_template(templates, template_start, template_length)?;
            if reconstructed != template_string {
                return Err(invalid(format!(
                    "edge {index} binary template differs from string table"
                )));
            }
            let condition_end =
                checked_add(condition_start, condition_length, "edge condition range")?;
            let program = condition_bytes
                .get(condition_start..condition_end)
                .ok_or_else(|| invalid(format!("edge {index} has invalid condition range")))?;
            let validation = validate_condition_program(
                program,
                self.header.counts[1] as usize,
                self.header.counts[2] as usize,
                self.header.counts[4] as usize,
                self.header.counts[0] as usize,
            )?;
            if validation.zemberek_count != condition_count {
                return Err(invalid(format!(
                    "edge {index} condition count {} != binary {}",
                    condition_count, validation.zemberek_count
                )));
            }
            null_sentinels = checked_add(
                null_sentinels,
                validation.null_dictionary_sentinels,
                "null sentinel total",
            )?;
        }
        if null_sentinels != self.header.counts[9] as usize {
            return Err(invalid(format!(
                "null dictionary sentinel count {null_sentinels} != header {}",
                self.header.counts[9]
            )));
        }
        Ok(())
    }

    fn section(self, section: Section) -> Result<&'a [u8], BinaryError> {
        let range = self.section_range(section)?;
        self.bytes
            .get(range)
            .ok_or_else(|| invalid(format!("section {section:?} is out of bounds")))
    }

    fn section_range(self, section: Section) -> Result<std::ops::Range<usize>, BinaryError> {
        let entry = self.header.sections[section as usize];
        let start = usize::try_from(entry.offset)
            .map_err(|_| invalid(format!("section {section:?} offset is too large")))?;
        let end = checked_add(start, entry.length as usize, "section end")?;
        if start < HEADER_SIZE || end > self.bytes.len() {
            return Err(invalid(format!("section {section:?} is outside the file")));
        }
        Ok(start..end)
    }
}

impl fmt::Debug for BinaryBundleView<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BinaryBundleView")
            .field("summary", &self.summary())
            .finish()
    }
}

/// Counts and byte size from a validated native binary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BinarySummary {
    /// Number of interned strings.
    pub string_count: usize,
    /// Number of morphemes.
    pub morpheme_count: usize,
    /// Number of dictionary entries.
    pub dictionary_count: usize,
    /// Number of stems.
    pub stem_count: usize,
    /// Number of states.
    pub state_count: usize,
    /// Number of edges.
    pub edge_count: usize,
    /// Number of state aliases.
    pub alias_count: usize,
    /// Number of suffix-template tokens.
    pub template_token_count: usize,
    /// Number of condition bytecode bytes.
    pub condition_byte_count: usize,
    /// Number of explicit upstream null dictionary lookup sentinels.
    pub null_dictionary_sentinel_count: usize,
    /// Complete file size.
    pub file_byte_count: usize,
}

struct Compiler<'a> {
    bundle: &'a MorphBundle,
    strings: StringInterner,
    morpheme_index: HashMap<&'a str, u32>,
    dictionary_index: HashMap<&'a str, u32>,
    state_index: HashMap<&'a str, u32>,
    aliases: Vec<u32>,
    templates: Vec<u8>,
    conditions: Vec<u8>,
    binary_edges: Vec<BinaryEdge>,
    state_edge_ranges: Vec<(u32, u32)>,
    null_dictionary_sentinels: u32,
}

impl<'a> Compiler<'a> {
    fn new(bundle: &'a MorphBundle) -> Result<Self, BinaryError> {
        let strings = StringInterner::collect(bundle)?;
        let morpheme_index = index_by(
            &bundle.morphemes,
            |record| record.id.as_str(),
            "morpheme index",
        )?;
        let dictionary_index = index_by(
            &bundle.dictionary,
            |record| record.id.as_str(),
            "dictionary index",
        )?;
        let state_index = index_by(&bundle.states, |record| record.key.as_str(), "state index")?;
        Ok(Self {
            bundle,
            strings,
            morpheme_index,
            dictionary_index,
            state_index,
            aliases: Vec::new(),
            templates: Vec::new(),
            conditions: Vec::new(),
            binary_edges: Vec::new(),
            state_edge_ranges: Vec::new(),
            null_dictionary_sentinels: 0,
        })
    }

    fn compile(&mut self) -> Result<Vec<u8>, BinaryError> {
        self.compile_edges_and_state_ranges()?;
        let sections = [
            self.compile_string_offsets(),
            self.strings.blob.clone(),
            self.compile_morphemes()?,
            self.compile_dictionary()?,
            self.compile_stems()?,
            self.compile_states()?,
            self.compile_edges(),
            u32_slice_bytes(&self.aliases),
            self.templates.clone(),
            self.conditions.clone(),
        ];
        self.assemble(&sections)
    }

    fn compile_string_offsets(&self) -> Vec<u8> {
        let mut output = Vec::with_capacity((self.strings.offsets.len()) * 4);
        for &offset in &self.strings.offsets {
            push_u32(&mut output, offset);
        }
        output
    }

    fn compile_morphemes(&self) -> Result<Vec<u8>, BinaryError> {
        let mut output = Vec::with_capacity(self.bundle.morphemes.len() * MORPHEME_RECORD_SIZE);
        for record in &self.bundle.morphemes {
            push_u32(&mut output, self.strings.id(&record.id)?);
            push_u32(&mut output, self.strings.id(&record.name)?);
            let pos = record
                .pos
                .as_deref()
                .map(|value| index_u16(PRIMARY_POS_SHORT, value, "morpheme POS"))
                .transpose()?
                .unwrap_or(NONE_U16);
            push_u16(&mut output, pos);
            let flags = u16::from(record.derivational) | (u16::from(record.informal) << 1);
            push_u16(&mut output, flags);
            push_u32(
                &mut output,
                optional_index(
                    record.mapped_id.as_deref(),
                    &self.morpheme_index,
                    "mapped morpheme",
                )?,
            );
            output.extend_from_slice(&[0; 8]);
        }
        Ok(output)
    }

    fn compile_dictionary(&self) -> Result<Vec<u8>, BinaryError> {
        let mut output = Vec::with_capacity(self.bundle.dictionary.len() * DICTIONARY_RECORD_SIZE);
        for record in &self.bundle.dictionary {
            push_u32(&mut output, self.strings.id(&record.id)?);
            push_u32(&mut output, self.strings.id(&record.lemma)?);
            push_u32(&mut output, self.strings.id(&record.root)?);
            push_u32(&mut output, self.strings.id(&record.pronunciation)?);
            push_u16(
                &mut output,
                index_u16(
                    PRIMARY_POS_SHORT,
                    &record.primary_pos,
                    "dictionary primary POS",
                )?,
            );
            push_u16(
                &mut output,
                index_u16(
                    SECONDARY_POS_SHORT,
                    &record.secondary_pos,
                    "dictionary secondary POS",
                )?,
            );
            push_u32(
                &mut output,
                bitset(ROOT_ATTRIBUTES, &record.attributes, "root attributes")?,
            );
            push_u32(
                &mut output,
                optional_index(
                    record.reference_id.as_deref(),
                    &self.dictionary_index,
                    "dictionary reference",
                )?,
            );
            push_i32(&mut output, record.index);
            output.extend_from_slice(&[0; 8]);
        }
        Ok(output)
    }

    fn compile_stems(&self) -> Result<Vec<u8>, BinaryError> {
        let mut output = Vec::with_capacity(self.bundle.stems.len() * STEM_RECORD_SIZE);
        let mut stems: Vec<&StemRecord> = self.bundle.stems.iter().collect();
        stems.sort_by(|left, right| {
            left.surface
                .cmp(&right.surface)
                .then_with(|| left.source_order.cmp(&right.source_order))
        });
        for record in stems {
            push_u32(&mut output, self.strings.id(&record.surface)?);
            push_u32(
                &mut output,
                required_index(
                    &self.dictionary_index,
                    &record.dictionary_id,
                    "stem dictionary",
                )?,
            );
            push_u32(
                &mut output,
                required_index(&self.state_index, &record.target_state, "stem state")?,
            );
            push_u32(
                &mut output,
                u32::try_from(record.phonetic_bits)
                    .map_err(|_| invalid("negative stem phonetic bits"))?,
            );
        }
        Ok(output)
    }

    fn compile_states(&mut self) -> Result<Vec<u8>, BinaryError> {
        let mut output = Vec::with_capacity(self.bundle.states.len() * STATE_RECORD_SIZE);
        for (index, record) in self.bundle.states.iter().enumerate() {
            let alias_start = usize_to_u32(self.aliases.len(), "alias start")?;
            for alias in &record.declared_fields {
                self.aliases.push(self.strings.id(alias)?);
            }
            let alias_count = usize_to_u32(record.declared_fields.len(), "alias count")?;
            let (edge_start, edge_count) = self.state_edge_ranges[index];
            push_u32(&mut output, self.strings.id(&record.key)?);
            push_u32(&mut output, self.strings.id(&record.zemberek_id)?);
            push_u32(
                &mut output,
                required_index(&self.morpheme_index, &record.morpheme_id, "state morpheme")?,
            );
            let flags = u32::from(record.terminal)
                | (u32::from(record.derivative) << 1)
                | (u32::from(record.pos_root) << 2);
            push_u32(&mut output, flags);
            push_u32(&mut output, edge_start);
            push_u32(&mut output, edge_count);
            push_u32(
                &mut output,
                usize_to_u32(record.incoming_count, "state incoming count")?,
            );
            push_u32(&mut output, alias_start);
            push_u32(&mut output, alias_count);
        }
        Ok(output)
    }

    fn compile_edges_and_state_ranges(&mut self) -> Result<(), BinaryError> {
        let mut edges_by_owner: HashMap<&str, Vec<&EdgeRecord>> = HashMap::new();
        for edge in &self.bundle.edges {
            edges_by_owner
                .entry(edge.owner_state.as_str())
                .or_default()
                .push(edge);
        }
        for state in &self.bundle.states {
            let start = usize_to_u32(self.binary_edges.len(), "state edge start")?;
            let mut edges = edges_by_owner
                .remove(state.key.as_str())
                .unwrap_or_default();
            edges.sort_by_key(|edge| edge.source_order);
            for edge in edges {
                let compiled = self.compile_edge(edge)?;
                self.binary_edges.push(compiled);
            }
            let count = usize_to_u32(self.binary_edges.len() - start as usize, "state edge count")?;
            self.state_edge_ranges.push((start, count));
        }
        if !edges_by_owner.is_empty() {
            return Err(invalid(
                "edges reference owner states outside the state table",
            ));
        }
        Ok(())
    }

    fn compile_edge(&mut self, edge: &EdgeRecord) -> Result<BinaryEdge, BinaryError> {
        let template_start = usize_to_u32(
            self.templates.len() / TEMPLATE_RECORD_SIZE,
            "template start",
        )?;
        for token in &edge.template_tokens {
            compile_template_token(&mut self.templates, token)?;
        }
        let template_count = usize_to_u16(edge.template_tokens.len(), "template count")?;
        let condition_start = usize_to_u32(self.conditions.len(), "condition offset")?;
        if let Some(condition) = &edge.condition {
            let mut compiler = ConditionCompiler {
                output: &mut self.conditions,
                morphemes: &self.morpheme_index,
                dictionary: &self.dictionary_index,
                states: &self.state_index,
                strings: &self.strings,
                null_sentinels: &mut self.null_dictionary_sentinels,
            };
            compiler.compile(condition)?;
        }
        let condition_length = usize_to_u32(
            self.conditions.len() - condition_start as usize,
            "condition length",
        )?;
        Ok(BinaryEdge {
            declared_from: required_index(
                &self.state_index,
                &edge.declared_from,
                "edge declared_from",
            )?,
            to_state: required_index(&self.state_index, &edge.to_state, "edge target")?,
            morpheme: required_index(&self.morpheme_index, &edge.morpheme_id, "edge morpheme")?,
            template_string: self.strings.id(&edge.surface_template)?,
            template_start,
            condition_start,
            condition_length,
            template_count,
            condition_count: usize_to_u16(edge.condition_count, "condition count")?,
            flags: u32::from(edge.owner_matches_declared_from),
        })
    }

    fn compile_edges(&self) -> Vec<u8> {
        let mut output = Vec::with_capacity(self.binary_edges.len() * EDGE_RECORD_SIZE);
        for edge in &self.binary_edges {
            push_u32(&mut output, edge.declared_from);
            push_u32(&mut output, edge.to_state);
            push_u32(&mut output, edge.morpheme);
            push_u32(&mut output, edge.template_string);
            push_u32(&mut output, edge.template_start);
            push_u32(&mut output, edge.condition_start);
            push_u32(&mut output, edge.condition_length);
            push_u16(&mut output, edge.template_count);
            push_u16(&mut output, edge.condition_count);
            push_u32(&mut output, edge.flags);
        }
        output
    }

    fn assemble(&self, sections: &[Vec<u8>; SECTION_COUNT]) -> Result<Vec<u8>, BinaryError> {
        let (mut output, directory) = Self::assemble_sections(sections)?;
        self.write_header(&mut output, &directory)?;
        let payload_hash: [u8; 32] = Sha256::digest(&output[HEADER_SIZE..]).into();
        output[PAYLOAD_HASH_OFFSET..PAYLOAD_HASH_OFFSET + 32].copy_from_slice(&payload_hash);
        BinaryBundleView::parse(&output)?;
        Ok(output)
    }

    fn assemble_sections(
        sections: &[Vec<u8>; SECTION_COUNT],
    ) -> Result<(Vec<u8>, [SectionEntry; SECTION_COUNT]), BinaryError> {
        let mut output = vec![0_u8; HEADER_SIZE];
        let mut directory = [SectionEntry::default(); SECTION_COUNT];
        for (index, section) in sections.iter().enumerate() {
            align_eight(&mut output);
            directory[index] = SectionEntry {
                offset: usize_to_u64(output.len(), "section offset")?,
                length: usize_to_u32(section.len(), "section length")?,
            };
            output.extend_from_slice(section);
        }
        Ok((output, directory))
    }

    fn write_header(
        &self,
        output: &mut [u8],
        directory: &[SectionEntry; SECTION_COUNT],
    ) -> Result<(), BinaryError> {
        output[0..8].copy_from_slice(MAGIC);
        write_u32(output, 8, ENDIAN_MARKER)?;
        write_u32(output, 12, BINARY_SCHEMA_VERSION)?;
        write_u32(output, 16, usize_to_u32(HEADER_SIZE, "header size")?)?;
        write_u32(output, 20, usize_to_u32(SECTION_COUNT, "section count")?)?;
        output[SOURCE_COMMIT_OFFSET..SOURCE_COMMIT_OFFSET + 20]
            .copy_from_slice(&decode_hex_20(ZEMBEREK_COMMIT)?);
        let file_length = usize_to_u64(output.len(), "file length")?;
        write_u64(output, FILE_LENGTH_OFFSET, file_length)?;
        for (index, count) in self.header_counts()?.into_iter().enumerate() {
            write_u32(output, COUNTS_OFFSET + index * 4, count)?;
        }
        for (index, entry) in directory.iter().copied().enumerate() {
            let position = DIRECTORY_OFFSET + index * DIRECTORY_ENTRY_SIZE;
            write_u64(output, position, entry.offset)?;
            write_u32(output, position + 8, entry.length)?;
        }
        Ok(())
    }

    fn header_counts(&self) -> Result<[u32; 10], BinaryError> {
        Ok([
            usize_to_u32(self.strings.strings.len(), "string count")?,
            usize_to_u32(self.bundle.morphemes.len(), "morpheme count")?,
            usize_to_u32(self.bundle.dictionary.len(), "dictionary count")?,
            usize_to_u32(self.bundle.stems.len(), "stem count")?,
            usize_to_u32(self.bundle.states.len(), "state count")?,
            usize_to_u32(self.binary_edges.len(), "edge count")?,
            usize_to_u32(self.aliases.len(), "alias count")?,
            usize_to_u32(
                self.templates.len() / TEMPLATE_RECORD_SIZE,
                "template token count",
            )?,
            usize_to_u32(self.conditions.len(), "condition bytes")?,
            self.null_dictionary_sentinels,
        ])
    }
}

#[derive(Clone, Copy, Debug)]
struct BinaryEdge {
    declared_from: u32,
    to_state: u32,
    morpheme: u32,
    template_string: u32,
    template_start: u32,
    condition_start: u32,
    condition_length: u32,
    template_count: u16,
    condition_count: u16,
    flags: u32,
}

struct StringInterner {
    strings: Vec<String>,
    ids: HashMap<String, u32>,
    offsets: Vec<u32>,
    blob: Vec<u8>,
}

impl StringInterner {
    fn collect(bundle: &MorphBundle) -> Result<Self, BinaryError> {
        let mut values = BTreeSet::new();
        for record in &bundle.morphemes {
            values.insert(record.id.clone());
            values.insert(record.name.clone());
        }
        for record in &bundle.dictionary {
            values.insert(record.id.clone());
            values.insert(record.lemma.clone());
            values.insert(record.root.clone());
            values.insert(record.pronunciation.clone());
        }
        for record in &bundle.stems {
            values.insert(record.surface.clone());
        }
        for record in &bundle.states {
            values.insert(record.key.clone());
            values.insert(record.zemberek_id.clone());
            values.extend(record.declared_fields.iter().cloned());
        }
        for edge in &bundle.edges {
            values.insert(edge.surface_template.clone());
            if let Some(condition) = &edge.condition {
                collect_condition_strings(condition, &mut values);
            }
        }
        let strings: Vec<String> = values.into_iter().collect();
        let mut ids = HashMap::with_capacity(strings.len());
        let mut offsets = Vec::with_capacity(strings.len() + 1);
        let mut blob = Vec::new();
        offsets.push(0);
        for (index, value) in strings.iter().enumerate() {
            ids.insert(value.clone(), usize_to_u32(index, "string ID")?);
            blob.extend_from_slice(value.as_bytes());
            offsets.push(usize_to_u32(blob.len(), "string blob offset")?);
        }
        Ok(Self {
            strings,
            ids,
            offsets,
            blob,
        })
    }

    fn id(&self, value: &str) -> Result<u32, BinaryError> {
        self.ids
            .get(value)
            .copied()
            .ok_or_else(|| invalid(format!("string was not interned: {value:?}")))
    }
}

#[derive(Clone)]
struct StringTable<'a> {
    values: Arc<[&'a str]>,
}

impl<'a> StringTable<'a> {
    fn parse(view: BinaryBundleView<'a>) -> Result<Self, BinaryError> {
        let offsets = view.section(Section::StringOffsets)?;
        let blob = view.section(Section::StringBlob)?;
        let count = view.header.counts[0] as usize;
        let mut values = Vec::with_capacity(count);
        let mut previous_string = None;
        for index in 0..count {
            let start = read_u32(offsets, index * 4)? as usize;
            let end = read_u32(offsets, (index + 1) * 4)? as usize;
            if start > end || end > blob.len() {
                return Err(invalid(format!("string {index} has invalid offsets")));
            }
            let value = std::str::from_utf8(&blob[start..end])
                .map_err(|error| invalid(format!("string {index} is not UTF-8: {error}")))?;
            require_strict_order(previous_string, value, "binary string table")?;
            previous_string = Some(value);
            values.push(value);
        }
        let final_offset = read_u32(offsets, count * 4)? as usize;
        if final_offset != blob.len() {
            return Err(invalid("final string offset does not equal blob length"));
        }
        Ok(Self {
            values: values.into(),
        })
    }

    fn find(&self, needle: &str) -> Result<Option<u32>, BinaryError> {
        match self.values.binary_search(&needle) {
            Ok(index) => Ok(Some(usize_to_u32(index, "string search result")?)),
            Err(_) => Ok(None),
        }
    }

    fn get(&self, id: u32) -> Result<&'a str, BinaryError> {
        let index = require_index(id, self.values.len(), "string ID")?;
        self.values
            .get(index)
            .copied()
            .ok_or_else(|| invalid(format!("string ID {id} is out of bounds")))
    }
}

#[derive(Clone, Copy)]
struct BinaryHeader {
    payload_hash: [u8; 32],
    counts: [u32; 10],
    sections: [SectionEntry; SECTION_COUNT],
}

impl BinaryHeader {
    fn parse(bytes: &[u8]) -> Result<Self, BinaryError> {
        Self::validate_identity(bytes)?;
        let mut payload_hash = [0_u8; 32];
        payload_hash.copy_from_slice(&bytes[PAYLOAD_HASH_OFFSET..PAYLOAD_HASH_OFFSET + 32]);
        let mut counts = [0_u32; 10];
        for (index, count) in counts.iter_mut().enumerate() {
            *count = read_u32(bytes, COUNTS_OFFSET + index * 4)?;
        }
        let mut sections = [SectionEntry::default(); SECTION_COUNT];
        for (index, entry) in sections.iter_mut().enumerate() {
            let position = DIRECTORY_OFFSET + index * DIRECTORY_ENTRY_SIZE;
            *entry = SectionEntry {
                offset: read_u64(bytes, position)?,
                length: read_u32(bytes, position + 8)?,
            };
        }
        Ok(Self {
            payload_hash,
            counts,
            sections,
        })
    }

    fn validate_identity(bytes: &[u8]) -> Result<(), BinaryError> {
        if bytes.len() < HEADER_SIZE {
            return Err(invalid("binary file is shorter than the header"));
        }
        if &bytes[0..8] != MAGIC {
            return Err(invalid("invalid binary magic"));
        }
        if read_u32(bytes, 8)? != ENDIAN_MARKER {
            return Err(invalid("unsupported endian marker"));
        }
        if read_u32(bytes, 12)? != BINARY_SCHEMA_VERSION {
            return Err(invalid("unsupported binary schema version"));
        }
        if read_u32(bytes, 16)? as usize != HEADER_SIZE
            || read_u32(bytes, 20)? as usize != SECTION_COUNT
        {
            return Err(invalid("invalid header or section count"));
        }
        if bytes[SOURCE_COMMIT_OFFSET..SOURCE_COMMIT_OFFSET + 20] != decode_hex_20(ZEMBEREK_COMMIT)?
        {
            return Err(invalid(
                "binary source commit does not match the pinned upstream",
            ));
        }
        require_zero(&bytes[44..48], "header source padding")?;
        require_zero(&bytes[248..256], "header reserved bytes")?;
        let file_length = read_u64(bytes, FILE_LENGTH_OFFSET)?;
        if file_length != bytes.len() as u64 {
            return Err(invalid(format!(
                "header file length {file_length} != actual {}",
                bytes.len()
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct SectionEntry {
    offset: u64,
    length: u32,
}

fn collect_condition_strings(condition: &Condition, values: &mut BTreeSet<String>) {
    match condition {
        Condition::And { children } | Condition::Or { children } => {
            for child in children {
                collect_condition_strings(child, values);
            }
        }
        Condition::Not { child } => collect_condition_strings(child, values),
        Condition::RootSurfaceIs { surface } => {
            values.insert(surface.clone());
        }
        Condition::RootSurfaceIsAny { surfaces } => {
            values.extend(surfaces.iter().cloned());
        }
        _ => {}
    }
}

fn compile_template_token(output: &mut Vec<u8>, token: &TemplateToken) -> Result<(), BinaryError> {
    let opcode = match token.token_type {
        TemplateTokenType::IVowel => 1,
        TemplateTokenType::AVowel => 2,
        TemplateTokenType::DevoiceFirst => 3,
        TemplateTokenType::LastVoiced => 4,
        TemplateTokenType::LastNotVoiced => 5,
        TemplateTokenType::Append => 6,
        TemplateTokenType::Letter => 7,
    };
    let scalar = if token.letter.is_empty() {
        0
    } else {
        let mut characters = token.letter.chars();
        let value = characters
            .next()
            .ok_or_else(|| invalid("template token has no scalar"))?;
        if characters.next().is_some() {
            return Err(invalid("template token has more than one scalar"));
        }
        u32::from(value)
    };
    output.push(opcode);
    output.push(u8::from(token.append));
    output.extend_from_slice(&[0; 2]);
    push_u32(output, scalar);
    Ok(())
}

fn reconstruct_binary_template(
    section: &[u8],
    start: usize,
    count: usize,
) -> Result<String, BinaryError> {
    let mut output = String::new();
    for index in start..start + count {
        let record = fixed_record(section, index, TEMPLATE_RECORD_SIZE)?;
        let opcode = record[0];
        let append = record[1];
        require_zero(&record[2..4], "template reserved bytes")?;
        let scalar = read_u32(record, 4)?;
        let letter = if scalar == 0 {
            None
        } else {
            Some(
                char::from_u32(scalar)
                    .ok_or_else(|| invalid(format!("invalid template scalar {scalar}")))?,
            )
        };
        match opcode {
            1 => {
                require_none(letter.as_ref(), "I vowel template letter")?;
                output.push_str(if append == 1 { "+I" } else { "I" });
            }
            2 => {
                require_none(letter.as_ref(), "A vowel template letter")?;
                output.push_str(if append == 1 { "+A" } else { "A" });
            }
            3 => {
                require_zero_u8(append, "devoice append flag")?;
                output.push('>');
                output.push(require_some(letter, "devoice letter")?);
            }
            4 => {
                require_zero_u8(append, "last voiced append flag")?;
                output.push('~');
                output.push(require_some(letter, "last voiced letter")?);
            }
            5 => {
                require_zero_u8(append, "last not voiced append flag")?;
                output.push('!');
                output.push(require_some(letter, "last not voiced letter")?);
            }
            6 => {
                require_zero_u8(append, "append-token flag")?;
                output.push('+');
                output.push(require_some(letter, "append-token letter")?);
            }
            7 => {
                require_zero_u8(append, "letter-token append flag")?;
                output.push(require_some(letter, "literal template letter")?);
            }
            _ => return Err(invalid(format!("unknown template opcode {opcode}"))),
        }
        if append > 1 {
            return Err(invalid("template append flag is not boolean"));
        }
    }
    Ok(output)
}

struct ConditionCompiler<'a, 'bundle> {
    output: &'a mut Vec<u8>,
    morphemes: &'a HashMap<&'bundle str, u32>,
    dictionary: &'a HashMap<&'bundle str, u32>,
    states: &'a HashMap<&'bundle str, u32>,
    strings: &'a StringInterner,
    null_sentinels: &'a mut u32,
}

impl ConditionCompiler<'_, '_> {
    fn compile(&mut self, condition: &Condition) -> Result<(), BinaryError> {
        match condition {
            Condition::And { children } => self.compile_combined(children, 0x01),
            Condition::Or { children } => self.compile_combined(children, 0x02),
            Condition::Not { child } => {
                self.compile(child)?;
                self.output.push(0x03);
                Ok(())
            }
            Condition::HasRootAttribute { .. }
            | Condition::HasAnyRootAttribute { .. }
            | Condition::HasPhoneticAttribute { .. }
            | Condition::DictionaryItemIs { .. }
            | Condition::RootPrimaryPosIs { .. }
            | Condition::SecondaryPosIs { .. }
            | Condition::DictionaryItemIsAny { .. }
            | Condition::DictionaryItemIsNone { .. }
            | Condition::HasAnySuffixSurface
            | Condition::HasTail
            | Condition::HasNoTail => self.compile_feature(condition),
            Condition::HasTailSequence { .. }
            | Condition::ContainsMorphemeSequence { .. }
            | Condition::CurrentMorphemeIs { .. }
            | Condition::PreviousMorphemeIs { .. }
            | Condition::RootSurfaceIs { .. }
            | Condition::RootSurfaceIsAny { .. }
            | Condition::PreviousGroupContainsMorpheme { .. }
            | Condition::NoSurfaceAfterDerivation
            | Condition::ContainsMorpheme { .. }
            | Condition::PreviousMorphemeIsAny { .. }
            | Condition::CurrentMorphemeIsAny { .. } => self.compile_morpheme_or_surface(condition),
            Condition::PreviousStateIs { .. }
            | Condition::PreviousStateIsNot { .. }
            | Condition::CurrentStateIs { .. }
            | Condition::CurrentStateIsNot { .. }
            | Condition::LastDerivationIs { .. }
            | Condition::HasDerivation
            | Condition::LastDerivationIsAny { .. }
            | Condition::CurrentGroupContainsAny { .. }
            | Condition::PreviousGroupContains { .. }
            | Condition::PreviousStateIsAny { .. } => self.compile_state(condition),
        }
    }

    fn compile_combined(&mut self, children: &[Condition], opcode: u8) -> Result<(), BinaryError> {
        for child in children {
            self.compile(child)?;
        }
        self.output.push(opcode);
        push_u16(
            self.output,
            usize_to_u16(children.len(), "combined condition child count")?,
        );
        Ok(())
    }

    fn compile_feature(&mut self, condition: &Condition) -> Result<(), BinaryError> {
        match condition {
            Condition::HasRootAttribute { attribute } => {
                self.output.push(0x10);
                self.output
                    .push(index_u8(ROOT_ATTRIBUTES, attribute, "root attribute")?);
            }
            Condition::HasAnyRootAttribute { attributes } => {
                self.output.push(0x11);
                push_u32(
                    self.output,
                    bitset(ROOT_ATTRIBUTES, attributes, "root attributes")?,
                );
            }
            Condition::HasPhoneticAttribute { attribute } => {
                self.output.push(0x12);
                self.output.push(
                    phonetic_ordinal(attribute)
                        .and_then(|value| u8::try_from(value).ok())
                        .ok_or_else(|| {
                            invalid(format!("unknown phonetic attribute {attribute}"))
                        })?,
                );
            }
            Condition::DictionaryItemIs { item } => {
                self.output.push(0x13);
                push_u32(
                    self.output,
                    required_index(self.dictionary, item, "condition dictionary item")?,
                );
            }
            Condition::RootPrimaryPosIs { pos } => {
                self.output.push(0x14);
                self.output
                    .push(index_u8(PRIMARY_POS_NAMES, pos, "primary POS")?);
            }
            Condition::SecondaryPosIs { pos } => {
                self.output.push(0x15);
                self.output
                    .push(index_u8(SECONDARY_POS_NAMES, pos, "secondary POS")?);
            }
            Condition::DictionaryItemIsAny { items } => {
                self.output.push(0x16);
                compile_nullable_dictionary_set(
                    self.output,
                    items,
                    self.dictionary,
                    self.null_sentinels,
                )?;
            }
            Condition::DictionaryItemIsNone { items } => {
                self.output.push(0x17);
                compile_nullable_dictionary_set(
                    self.output,
                    items,
                    self.dictionary,
                    self.null_sentinels,
                )?;
            }
            Condition::HasAnySuffixSurface => self.output.push(0x18),
            Condition::HasTail => self.output.push(0x19),
            Condition::HasNoTail => self.output.push(0x1a),
            _ => {
                return Err(invalid(
                    "condition was routed to the wrong feature compiler",
                ))
            }
        }
        Ok(())
    }

    fn compile_morpheme_or_surface(&mut self, condition: &Condition) -> Result<(), BinaryError> {
        match condition {
            Condition::HasTailSequence { morphemes } => {
                self.output.push(0x1b);
                compile_reference_list(
                    self.output,
                    morphemes,
                    self.morphemes,
                    "tail morpheme sequence",
                )?;
            }
            Condition::ContainsMorphemeSequence { morphemes } => {
                self.output.push(0x1c);
                compile_reference_list(
                    self.output,
                    morphemes,
                    self.morphemes,
                    "morpheme sequence",
                )?;
            }
            Condition::CurrentMorphemeIs { morpheme } => {
                self.compile_single_reference(0x1d, morpheme, self.morphemes, "current morpheme")?;
            }
            Condition::PreviousMorphemeIs { morpheme } => {
                self.compile_single_reference(0x1e, morpheme, self.morphemes, "previous morpheme")?;
            }
            Condition::RootSurfaceIs { surface } => {
                self.output.push(0x21);
                push_u32(self.output, self.strings.id(surface)?);
            }
            Condition::RootSurfaceIsAny { surfaces } => {
                self.output.push(0x22);
                push_u16(
                    self.output,
                    usize_to_u16(surfaces.len(), "root surface count")?,
                );
                for surface in surfaces {
                    push_u32(self.output, self.strings.id(surface)?);
                }
            }
            Condition::PreviousGroupContainsMorpheme { morphemes } => {
                self.compile_reference_list(
                    0x2a,
                    morphemes,
                    self.morphemes,
                    "previous group morphemes",
                )?;
            }
            Condition::NoSurfaceAfterDerivation => self.output.push(0x2b),
            Condition::ContainsMorpheme { morphemes } => {
                self.compile_reference_list(
                    0x2c,
                    morphemes,
                    self.morphemes,
                    "contained morphemes",
                )?;
            }
            Condition::PreviousMorphemeIsAny { morphemes } => {
                self.compile_reference_list(0x2d, morphemes, self.morphemes, "previous morphemes")?;
            }
            Condition::CurrentMorphemeIsAny { morphemes } => {
                self.compile_reference_list(0x2e, morphemes, self.morphemes, "current morphemes")?;
            }
            _ => {
                return Err(invalid(
                    "condition was routed to the wrong morpheme compiler",
                ))
            }
        }
        Ok(())
    }

    fn compile_state(&mut self, condition: &Condition) -> Result<(), BinaryError> {
        match condition {
            Condition::PreviousStateIs { state } => {
                self.compile_single_reference(0x1f, state, self.states, "previous state")?;
            }
            Condition::PreviousStateIsNot { state } => {
                self.compile_single_reference(0x20, state, self.states, "previous state")?;
            }
            Condition::CurrentStateIs { state } => {
                self.compile_single_reference(0x23, state, self.states, "current state")?;
            }
            Condition::CurrentStateIsNot { state } => {
                self.compile_single_reference(0x24, state, self.states, "current state")?;
            }
            Condition::LastDerivationIs { state } => {
                self.compile_single_reference(0x25, state, self.states, "last derivation state")?;
            }
            Condition::HasDerivation => self.output.push(0x26),
            Condition::LastDerivationIsAny { states } => {
                self.compile_reference_list(0x27, states, self.states, "last derivation states")?;
            }
            Condition::CurrentGroupContainsAny { states } => {
                self.compile_reference_list(0x28, states, self.states, "current group states")?;
            }
            Condition::PreviousGroupContains { states } => {
                self.compile_reference_list(0x29, states, self.states, "previous group states")?;
            }
            Condition::PreviousStateIsAny { states } => {
                self.compile_reference_list(0x2f, states, self.states, "previous states")?;
            }
            _ => return Err(invalid("condition was routed to the wrong state compiler")),
        }
        Ok(())
    }

    fn compile_single_reference(
        &mut self,
        opcode: u8,
        value: &str,
        index: &HashMap<&str, u32>,
        label: &str,
    ) -> Result<(), BinaryError> {
        self.output.push(opcode);
        push_u32(self.output, required_index(index, value, label)?);
        Ok(())
    }

    fn compile_reference_list(
        &mut self,
        opcode: u8,
        values: &[String],
        index: &HashMap<&str, u32>,
        label: &str,
    ) -> Result<(), BinaryError> {
        self.output.push(opcode);
        compile_reference_list(self.output, values, index, label)
    }
}

fn compile_nullable_dictionary_set(
    output: &mut Vec<u8>,
    values: &[Option<String>],
    dictionary: &HashMap<&str, u32>,
    null_sentinels: &mut u32,
) -> Result<(), BinaryError> {
    push_u16(
        output,
        usize_to_u16(values.len(), "dictionary condition set count")?,
    );
    for value in values {
        if let Some(item) = value {
            push_u32(
                output,
                required_index(dictionary, item, "condition dictionary item")?,
            );
        } else {
            push_u32(output, NONE_U32);
            *null_sentinels = null_sentinels
                .checked_add(1)
                .ok_or_else(|| invalid("null sentinel count overflow"))?;
        }
    }
    Ok(())
}

fn compile_reference_list(
    output: &mut Vec<u8>,
    values: &[String],
    index: &HashMap<&str, u32>,
    label: &str,
) -> Result<(), BinaryError> {
    push_u16(output, usize_to_u16(values.len(), label)?);
    for value in values {
        push_u32(output, required_index(index, value, label)?);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default)]
struct ConditionValidation {
    zemberek_count: usize,
    null_dictionary_sentinels: usize,
}

struct ConditionProgramValidator<'a> {
    program: &'a [u8],
    position: usize,
    stack: Vec<usize>,
    null_dictionary_sentinels: usize,
    morpheme_count: usize,
    dictionary_count: usize,
    state_count: usize,
    string_count: usize,
}

impl<'a> ConditionProgramValidator<'a> {
    const fn new(
        program: &'a [u8],
        morpheme_count: usize,
        dictionary_count: usize,
        state_count: usize,
        string_count: usize,
    ) -> Self {
        Self {
            program,
            position: 0,
            stack: Vec::new(),
            null_dictionary_sentinels: 0,
            morpheme_count,
            dictionary_count,
            state_count,
            string_count,
        }
    }

    fn validate(mut self) -> Result<ConditionValidation, BinaryError> {
        if self.program.is_empty() {
            return Ok(ConditionValidation::default());
        }
        while self.position < self.program.len() {
            let opcode = read_byte(self.program, &mut self.position)?;
            let is_leaf = match opcode {
                0x01..=0x03 => {
                    self.validate_structural(opcode)?;
                    false
                }
                0x10..=0x17 => {
                    self.validate_feature(opcode)?;
                    true
                }
                0x18..=0x22 => {
                    self.validate_morpheme_or_surface(opcode)?;
                    true
                }
                0x23..=0x2f => {
                    self.validate_state_or_group(opcode)?;
                    true
                }
                _ => return Err(invalid(format!("unknown condition opcode {opcode:#04x}"))),
            };
            if is_leaf {
                self.stack.push(1);
            }
        }
        if self.stack.len() != 1 {
            return Err(invalid(format!(
                "condition program ended with stack depth {}",
                self.stack.len()
            )));
        }
        Ok(ConditionValidation {
            zemberek_count: self.stack[0],
            null_dictionary_sentinels: self.null_dictionary_sentinels,
        })
    }

    fn validate_structural(&mut self, opcode: u8) -> Result<(), BinaryError> {
        match opcode {
            0x01 | 0x02 => {
                let count = read_program_u16(self.program, &mut self.position)? as usize;
                if count < 2 || self.stack.len() < count {
                    return Err(invalid("invalid AND/OR condition stack"));
                }
                let split = self.stack.len() - count;
                let zemberek_count = self.stack[split..]
                    .iter()
                    .try_fold(0_usize, |total, value| {
                        checked_add(total, *value, "condition count")
                    })?;
                self.stack.truncate(split);
                self.stack.push(zemberek_count);
            }
            0x03 => {
                self.stack
                    .pop()
                    .ok_or_else(|| invalid("NOT condition has no operand"))?;
                self.stack.push(1);
            }
            _ => {
                return Err(invalid(
                    "opcode was routed to the wrong structural validator",
                ))
            }
        }
        Ok(())
    }

    fn validate_feature(&mut self, opcode: u8) -> Result<(), BinaryError> {
        match opcode {
            0x10 => require_small_index(
                read_byte(self.program, &mut self.position)? as usize,
                ROOT_ATTRIBUTES.len(),
                "condition root attribute",
            )?,
            0x11 => {
                let bits = read_program_u32(self.program, &mut self.position)?;
                let valid = (1_u32 << ROOT_ATTRIBUTES.len()) - 1;
                if bits & !valid != 0 {
                    return Err(invalid("condition has unknown root attribute bits"));
                }
            }
            0x12 => require_small_index(
                read_byte(self.program, &mut self.position)? as usize,
                PHONETIC_ATTRIBUTES.len(),
                "condition phonetic attribute",
            )?,
            0x13 => {
                require_index(
                    read_program_u32(self.program, &mut self.position)?,
                    self.dictionary_count,
                    "condition dictionary item",
                )?;
            }
            0x14 => require_small_index(
                read_byte(self.program, &mut self.position)? as usize,
                PRIMARY_POS_NAMES.len(),
                "condition primary POS",
            )?,
            0x15 => require_small_index(
                read_byte(self.program, &mut self.position)? as usize,
                SECONDARY_POS_NAMES.len(),
                "condition secondary POS",
            )?,
            0x16 | 0x17 => {
                let count = validate_nullable_index_list(
                    self.program,
                    &mut self.position,
                    self.dictionary_count,
                    "condition dictionary set",
                )?;
                self.null_dictionary_sentinels =
                    checked_add(self.null_dictionary_sentinels, count, "null sentinel total")?;
            }
            _ => return Err(invalid("opcode was routed to the wrong feature validator")),
        }
        Ok(())
    }

    fn validate_morpheme_or_surface(&mut self, opcode: u8) -> Result<(), BinaryError> {
        match opcode {
            0x18..=0x1a => {}
            0x1b | 0x1c => validate_index_list(
                self.program,
                &mut self.position,
                self.morpheme_count,
                "condition morpheme list",
            )?,
            0x1d | 0x1e => {
                require_index(
                    read_program_u32(self.program, &mut self.position)?,
                    self.morpheme_count,
                    "condition morpheme",
                )?;
            }
            0x1f | 0x20 => {
                require_index(
                    read_program_u32(self.program, &mut self.position)?,
                    self.state_count,
                    "condition state",
                )?;
            }
            0x21 => {
                require_index(
                    read_program_u32(self.program, &mut self.position)?,
                    self.string_count,
                    "condition surface string",
                )?;
            }
            0x22 => validate_index_list(
                self.program,
                &mut self.position,
                self.string_count,
                "condition surface strings",
            )?,
            _ => return Err(invalid("opcode was routed to the wrong path validator")),
        }
        Ok(())
    }

    fn validate_state_or_group(&mut self, opcode: u8) -> Result<(), BinaryError> {
        match opcode {
            0x23..=0x25 => {
                require_index(
                    read_program_u32(self.program, &mut self.position)?,
                    self.state_count,
                    "condition state",
                )?;
            }
            0x26 | 0x2b => {}
            0x27..=0x29 | 0x2f => validate_index_list(
                self.program,
                &mut self.position,
                self.state_count,
                "condition state list",
            )?,
            0x2a | 0x2c..=0x2e => validate_index_list(
                self.program,
                &mut self.position,
                self.morpheme_count,
                "condition morpheme list",
            )?,
            _ => return Err(invalid("opcode was routed to the wrong state validator")),
        }
        Ok(())
    }
}

fn validate_condition_program(
    program: &[u8],
    morpheme_count: usize,
    dictionary_count: usize,
    state_count: usize,
    string_count: usize,
) -> Result<ConditionValidation, BinaryError> {
    ConditionProgramValidator::new(
        program,
        morpheme_count,
        dictionary_count,
        state_count,
        string_count,
    )
    .validate()
}

fn validate_nullable_index_list(
    program: &[u8],
    position: &mut usize,
    limit: usize,
    label: &str,
) -> Result<usize, BinaryError> {
    let count = read_program_u16(program, position)? as usize;
    let mut null_count = 0_usize;
    let mut previous = None;
    for index in 0..count {
        let value = read_program_u32(program, position)?;
        if value == NONE_U32 {
            if index + 1 != count || null_count != 0 {
                return Err(invalid(format!(
                    "{label} has invalid null sentinel placement"
                )));
            }
            null_count = 1;
        } else {
            require_index(value, limit, label)?;
            if previous.is_some_and(|prior| prior >= value) {
                return Err(invalid(format!("{label} is not strictly sorted")));
            }
            previous = Some(value);
        }
    }
    Ok(null_count)
}

fn validate_index_list(
    program: &[u8],
    position: &mut usize,
    limit: usize,
    label: &str,
) -> Result<(), BinaryError> {
    let count = read_program_u16(program, position)? as usize;
    for _ in 0..count {
        require_index(read_program_u32(program, position)?, limit, label)?;
    }
    Ok(())
}

fn read_byte(program: &[u8], position: &mut usize) -> Result<u8, BinaryError> {
    let value = *program
        .get(*position)
        .ok_or_else(|| invalid("truncated condition bytecode"))?;
    *position = checked_add(*position, 1, "condition position")?;
    Ok(value)
}

fn read_program_u16(program: &[u8], position: &mut usize) -> Result<u16, BinaryError> {
    let value = read_u16(program, *position)?;
    *position = checked_add(*position, 2, "condition position")?;
    Ok(value)
}

fn read_program_u32(program: &[u8], position: &mut usize) -> Result<u32, BinaryError> {
    let value = read_u32(program, *position)?;
    *position = checked_add(*position, 4, "condition position")?;
    Ok(value)
}

fn previous_first_alias<'a>(
    aliases: &[u8],
    start: usize,
    count: usize,
    strings: &StringTable<'a>,
) -> Result<Option<&'a str>, BinaryError> {
    if count == 0 {
        Ok(None)
    } else {
        Ok(Some(strings.get(read_u32(aliases, start * 4)?)?))
    }
}

fn fixed_record(section: &[u8], index: usize, record_size: usize) -> Result<&[u8], BinaryError> {
    let start = checked_mul(index, record_size, "record offset")?;
    let end = checked_add(start, record_size, "record end")?;
    section
        .get(start..end)
        .ok_or_else(|| invalid(format!("record {index} is out of bounds")))
}

fn index_by<'a, T, F>(
    values: &'a [T],
    key: F,
    label: &str,
) -> Result<HashMap<&'a str, u32>, BinaryError>
where
    F: Fn(&'a T) -> &'a str,
{
    let mut result = HashMap::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let key = key(value);
        if result.insert(key, usize_to_u32(index, label)?).is_some() {
            return Err(invalid(format!("duplicate {label}: {key}")));
        }
    }
    Ok(result)
}

fn required_index(
    index: &HashMap<&str, u32>,
    value: &str,
    label: &str,
) -> Result<u32, BinaryError> {
    index
        .get(value)
        .copied()
        .ok_or_else(|| invalid(format!("missing {label}: {value}")))
}

fn optional_index(
    value: Option<&str>,
    index: &HashMap<&str, u32>,
    label: &str,
) -> Result<u32, BinaryError> {
    value
        .map(|item| required_index(index, item, label))
        .transpose()
        .map(|item| item.unwrap_or(NONE_U32))
}

fn bitset(known: &[&str], values: &[String], label: &str) -> Result<u32, BinaryError> {
    let mut bits = 0_u32;
    for value in values {
        let index = known
            .iter()
            .position(|known_value| *known_value == value)
            .ok_or_else(|| invalid(format!("unknown {label}: {value}")))?;
        let shift = u32::try_from(index).map_err(|_| invalid(format!("{label} index overflow")))?;
        bits |= 1_u32
            .checked_shl(shift)
            .ok_or_else(|| invalid(format!("{label} bit shift overflow")))?;
    }
    Ok(bits)
}

fn index_u8(known: &[&str], value: &str, label: &str) -> Result<u8, BinaryError> {
    let index = known
        .iter()
        .position(|known_value| *known_value == value)
        .ok_or_else(|| invalid(format!("unknown {label}: {value}")))?;
    u8::try_from(index).map_err(|_| invalid(format!("{label} index does not fit u8")))
}

fn index_u16(known: &[&str], value: &str, label: &str) -> Result<u16, BinaryError> {
    let index = known
        .iter()
        .position(|known_value| *known_value == value)
        .ok_or_else(|| invalid(format!("unknown {label}: {value}")))?;
    u16::try_from(index).map_err(|_| invalid(format!("{label} index does not fit u16")))
}

fn usize_to_u16(value: usize, label: &str) -> Result<u16, BinaryError> {
    u16::try_from(value).map_err(|_| invalid(format!("{label} does not fit u16")))
}

fn usize_to_u32(value: usize, label: &str) -> Result<u32, BinaryError> {
    u32::try_from(value).map_err(|_| invalid(format!("{label} does not fit u32")))
}

fn usize_to_u64(value: usize, label: &str) -> Result<u64, BinaryError> {
    u64::try_from(value).map_err(|_| invalid(format!("{label} does not fit u64")))
}

fn require_index(value: u32, limit: usize, label: &str) -> Result<usize, BinaryError> {
    let index = value as usize;
    if index < limit {
        Ok(index)
    } else {
        Err(invalid(format!("{label} index {value} >= {limit}")))
    }
}

fn require_optional_index(value: u32, limit: usize, label: &str) -> Result<(), BinaryError> {
    if value == NONE_U32 {
        Ok(())
    } else {
        require_index(value, limit, label).map(|_| ())
    }
}

fn require_small_index(value: usize, limit: usize, label: &str) -> Result<(), BinaryError> {
    if value < limit {
        Ok(())
    } else {
        Err(invalid(format!("{label} index {value} >= {limit}")))
    }
}

fn require_strict_order(
    previous: Option<&str>,
    current: &str,
    label: &str,
) -> Result<(), BinaryError> {
    if previous.is_some_and(|value| value >= current) {
        Err(invalid(format!("{label} is not strictly sorted")))
    } else {
        Ok(())
    }
}

fn require_zero(bytes: &[u8], label: &str) -> Result<(), BinaryError> {
    if bytes.iter().all(|byte| *byte == 0) {
        Ok(())
    } else {
        Err(invalid(format!("{label} is not zero")))
    }
}

fn require_zero_u8(value: u8, label: &str) -> Result<(), BinaryError> {
    if value == 0 {
        Ok(())
    } else {
        Err(invalid(format!("{label} is not zero")))
    }
}

fn require_none<T>(value: Option<&T>, label: &str) -> Result<(), BinaryError> {
    if value.is_none() {
        Ok(())
    } else {
        Err(invalid(format!("{label} must be absent")))
    }
}

fn require_some<T>(value: Option<T>, label: &str) -> Result<T, BinaryError> {
    value.ok_or_else(|| invalid(format!("{label} is absent")))
}

fn checked_add(left: usize, right: usize, label: &str) -> Result<usize, BinaryError> {
    left.checked_add(right)
        .ok_or_else(|| invalid(format!("{label} overflow")))
}

fn checked_mul(left: usize, right: usize, label: &str) -> Result<usize, BinaryError> {
    left.checked_mul(right)
        .ok_or_else(|| invalid(format!("{label} overflow")))
}

fn align_eight(output: &mut Vec<u8>) {
    let padding = (8 - output.len() % 8) % 8;
    output.resize(output.len() + padding, 0);
}

fn u32_slice_bytes(values: &[u32]) -> Vec<u8> {
    let mut output = Vec::with_capacity(std::mem::size_of_val(values));
    for &value in values {
        push_u32(&mut output, value);
    }
    output
}

fn push_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_i32(output: &mut Vec<u8>, value: i32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn write_u32(output: &mut [u8], position: usize, value: u32) -> Result<(), BinaryError> {
    let target = output
        .get_mut(position..position + 4)
        .ok_or_else(|| invalid("u32 write is out of bounds"))?;
    target.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn write_u64(output: &mut [u8], position: usize, value: u64) -> Result<(), BinaryError> {
    let target = output
        .get_mut(position..position + 8)
        .ok_or_else(|| invalid("u64 write is out of bounds"))?;
    target.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn read_u16(input: &[u8], position: usize) -> Result<u16, BinaryError> {
    let bytes: [u8; 2] = input
        .get(position..position + 2)
        .ok_or_else(|| invalid("truncated u16"))?
        .try_into()
        .map_err(|_| invalid("invalid u16 slice"))?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32(input: &[u8], position: usize) -> Result<u32, BinaryError> {
    let bytes: [u8; 4] = input
        .get(position..position + 4)
        .ok_or_else(|| invalid("truncated u32"))?
        .try_into()
        .map_err(|_| invalid("invalid u32 slice"))?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i32(input: &[u8], position: usize) -> Result<i32, BinaryError> {
    let bytes: [u8; 4] = input
        .get(position..position + 4)
        .ok_or_else(|| invalid("truncated i32"))?
        .try_into()
        .map_err(|_| invalid("invalid i32 slice"))?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_u64(input: &[u8], position: usize) -> Result<u64, BinaryError> {
    let bytes: [u8; 8] = input
        .get(position..position + 8)
        .ok_or_else(|| invalid("truncated u64"))?
        .try_into()
        .map_err(|_| invalid("invalid u64 slice"))?;
    Ok(u64::from_le_bytes(bytes))
}

fn decode_hex_20(value: &str) -> Result<[u8; 20], BinaryError> {
    if value.len() != 40 {
        return Err(invalid("source commit hex must contain 40 characters"));
    }
    let mut output = [0_u8; 20];
    for (index, byte) in output.iter_mut().enumerate() {
        let start = index * 2;
        *byte = u8::from_str_radix(&value[start..start + 2], 16)
            .map_err(|_| invalid("source commit contains invalid hex"))?;
    }
    Ok(output)
}

/// Native binary compilation or validation error.
#[derive(Debug)]
pub enum BinaryError {
    /// Source JSON bundle validation failed before compilation.
    SourceBundle(BundleError),
    /// Binary schema or data invariant failed.
    Invalid { message: String },
}

impl fmt::Display for BinaryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for BinaryError {}

fn invalid(message: impl Into<String>) -> BinaryError {
    BinaryError::Invalid {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::{compile_binary, BinaryBundleView};
    use crate::{
        BundleManifest, ConditionKindCount, DictionaryRecord, EdgeRecord, MorphBundle,
        MorphemeRecord, StateRecord, StemRecord, TemplateToken, TemplateTokenType,
    };

    fn tiny_bundle() -> MorphBundle {
        MorphBundle {
            manifest: BundleManifest {
                schema_version: 1,
                zemberek_commit: crate::ZEMBEREK_COMMIT.to_owned(),
                zemberek_version: "0.17.2".to_owned(),
                exporter: "nedo-zemberek-graph-bundle-v1".to_owned(),
                morpheme_count: 1,
                dictionary_count: 1,
                stem_count: 1,
                state_count: 1,
                edge_count: 1,
                owner_declared_from_mismatch_count: 0,
                reflected_state_field_count: 1,
                duplicate_zemberek_state_id_count: 0,
                aliasless_state_count: 0,
                condition_kind_counts: vec![ConditionKindCount {
                    kind: "HasTail".to_owned(),
                    count: 1,
                }],
            },
            morphemes: vec![MorphemeRecord {
                id: "Noun".to_owned(),
                name: "Noun".to_owned(),
                pos: Some("Noun".to_owned()),
                derivational: false,
                informal: false,
                mapped_id: None,
            }],
            dictionary: vec![DictionaryRecord {
                id: "ev_Noun".to_owned(),
                lemma: "ev".to_owned(),
                root: "ev".to_owned(),
                pronunciation: "ev".to_owned(),
                primary_pos: "Noun".to_owned(),
                secondary_pos: "None".to_owned(),
                attributes: Vec::new(),
                reference_id: None,
                index: 0,
            }],
            stems: vec![StemRecord {
                surface: "ev".to_owned(),
                source_order: 0,
                dictionary_id: "ev_Noun".to_owned(),
                target_state: "Root.root".to_owned(),
                target_zemberek_id: "root".to_owned(),
                phonetic_bits: 0,
                phonetic_attributes: Vec::new(),
            }],
            states: vec![StateRecord {
                key: "Root.root".to_owned(),
                zemberek_id: "root".to_owned(),
                morpheme_id: "Noun".to_owned(),
                terminal: true,
                derivative: false,
                pos_root: true,
                outgoing_count: 1,
                incoming_count: 1,
                declared_fields: vec!["Root.root".to_owned()],
            }],
            edges: vec![EdgeRecord {
                id: "e000000".to_owned(),
                source_order: 0,
                owner_state: "Root.root".to_owned(),
                owner_zemberek_id: "root".to_owned(),
                declared_from: "Root.root".to_owned(),
                declared_from_zemberek_id: "root".to_owned(),
                owner_matches_declared_from: true,
                to_state: "Root.root".to_owned(),
                to_zemberek_id: "root".to_owned(),
                morpheme_id: "Noun".to_owned(),
                surface_template: "lAr".to_owned(),
                template_tokens: vec![
                    TemplateToken {
                        token_type: TemplateTokenType::Letter,
                        letter: "l".to_owned(),
                        append: false,
                    },
                    TemplateToken {
                        token_type: TemplateTokenType::AVowel,
                        letter: String::new(),
                        append: false,
                    },
                    TemplateToken {
                        token_type: TemplateTokenType::Letter,
                        letter: "r".to_owned(),
                        append: false,
                    },
                ],
                condition_count: 1,
                condition: Some(crate::Condition::HasTail),
            }],
        }
    }

    #[test]
    fn binary_compilation_is_deterministic_and_valid() -> Result<(), super::BinaryError> {
        let bundle = tiny_bundle();
        let first = compile_binary(&bundle)?;
        let second = compile_binary(&bundle)?;
        assert_eq!(first, second);
        let summary = BinaryBundleView::parse(&first)?.summary();
        assert_eq!(summary.dictionary_count, 1);
        assert_eq!(summary.edge_count, 1);
        Ok(())
    }

    #[test]
    fn payload_corruption_is_rejected() -> Result<(), super::BinaryError> {
        let bundle = tiny_bundle();
        let mut bytes = compile_binary(&bundle)?;
        let last = bytes
            .last_mut()
            .ok_or_else(|| super::invalid("compiled fixture is empty"))?;
        *last ^= 1;
        assert!(BinaryBundleView::parse(&bytes).is_err());
        Ok(())
    }

    #[test]
    fn bad_magic_is_rejected() -> Result<(), super::BinaryError> {
        let bundle = tiny_bundle();
        let mut bytes = compile_binary(&bundle)?;
        bytes[0] ^= 1;
        assert!(BinaryBundleView::parse(&bytes).is_err());
        Ok(())
    }
}
