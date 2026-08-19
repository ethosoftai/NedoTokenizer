//! Unified `NedoFormer` generation vocabulary and byte-exact target codec.

use std::collections::{BTreeSet, HashMap};

use sha2::{Digest, Sha256};

use super::{
    LexicalKind, TokenMode, TokenStatus, TokenizedDocument, TokenizedUnit, TokenizerError,
};

const VOCAB_MAGIC: &[u8; 8] = b"NDFVOC01";
const VOCAB_SCHEMA: u32 = 1;
const HEADER_BYTES: usize = 8 + 4 + 4 + 32;

const CONTROL_TOKENS: [&str; 17] = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unit_end>",
    "<word_end>",
    "<char>",
    "</char>",
    "<code>",
    "</code>",
    "<glue>",
    "<space>",
    "<spaces>",
    "</spaces>",
    "<tab>",
    "<lf>",
    "<crlf>",
    "<nbsp>",
];

const PAD_ID: u16 = 0;
const BOS_ID: u16 = 1;
const EOS_ID: u16 = 2;
const UNIT_END_ID: u16 = 3;
const WORD_END_ID: u16 = 4;
const CHAR_START_ID: u16 = 5;
const CHAR_END_ID: u16 = 6;
const CODE_START_ID: u16 = 7;
const CODE_END_ID: u16 = 8;
const GLUE_ID: u16 = 9;
const SPACE_ID: u16 = 10;
const SPACES_START_ID: u16 = 11;
const SPACES_END_ID: u16 = 12;
const TAB_ID: u16 = 13;
const LF_ID: u16 = 14;
const CRLF_ID: u16 = 15;
const NBSP_ID: u16 = 16;

/// Stable class of one unified generation-vocabulary entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum NedoFormerVocabKind {
    /// Zero-byte control symbol.
    Control = 0,
    /// Fixed one-byte fallback symbol.
    Byte = 1,
    /// Learned Unicode scalar.
    Character = 2,
    /// Exact 1-3 digit atom, including leading-zero forms.
    Number = 3,
    /// Surface-aware non-root morpheme.
    Morpheme = 4,
    /// Surface-aware frequent root.
    Root = 5,
    /// Exact code/identifier piece learned from code-mode units.
    Code = 6,
}

impl NedoFormerVocabKind {
    const fn from_u8(value: u8) -> Result<Self, TokenizerError> {
        match value {
            0 => Ok(Self::Control),
            1 => Ok(Self::Byte),
            2 => Ok(Self::Character),
            3 => Ok(Self::Number),
            4 => Ok(Self::Morpheme),
            5 => Ok(Self::Root),
            6 => Ok(Self::Code),
            _ => Err(TokenizerError::InvalidVocabulary(
                "invalid NedoFormer vocabulary entry kind",
            )),
        }
    }
}

/// One unified generation-vocabulary entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NedoFormerVocabEntry {
    /// Entry class.
    pub kind: NedoFormerVocabKind,
    /// Stable human-readable identity.
    pub token: String,
    /// Exact bytes produced by this entry; controls are empty.
    pub surface: Vec<u8>,
}

/// Unified `NedoFormer` output vocabulary.
#[derive(Clone, Debug)]
pub struct NedoFormerVocabulary {
    entries: Vec<NedoFormerVocabEntry>,
    token_to_id: HashMap<String, u16>,
    char_to_id: HashMap<char, u16>,
    byte_ids: [u16; 256],
    number_to_id: HashMap<String, u16>,
}

/// One encoded `NedoFormer` decoder target.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NedoFormerGenerationEncoding {
    /// Unified vocabulary IDs.
    pub ids: Vec<u16>,
}

impl NedoFormerVocabulary {
    /// Trains a deterministic unified vocabulary from selected tokenized documents.
    ///
    /// `max_roots` counts exact `(dictionary ID, realized root surface)` entries.
    /// `max_chars` counts learned Unicode scalars in addition to the fixed 256-byte
    /// fallback. `max_code_pieces` counts exact code-mode lexical pieces.
    ///
    /// # Errors
    ///
    /// Returns an error if the resulting unified ID space exceeds `u16`.
    #[allow(clippy::too_many_lines)] // Deterministic ranking of all unified vocabulary families is one transaction.
    pub fn train(
        documents: &[TokenizedDocument],
        max_roots: usize,
        max_chars: usize,
        max_code_pieces: usize,
    ) -> Result<Self, TokenizerError> {
        let mut root_counts = HashMap::<String, u64>::new();
        let mut roots = HashMap::<String, (String, Vec<u8>)>::new();
        let mut morphemes = HashMap::<String, (String, Vec<u8>)>::new();
        let mut char_counts = HashMap::<char, u64>::new();
        let mut code_counts = HashMap::<Vec<u8>, u64>::new();

        for document in documents {
            count_chars(document.raw(), &mut char_counts);
            for unit in document.units() {
                if unit.mode == TokenMode::Code
                    && !matches!(unit.kind, LexicalKind::Whitespace | LexicalKind::LineBreak)
                {
                    for piece in unit_piece_bytes(document, unit)? {
                        *code_counts.entry(piece.to_vec()).or_insert(0) += 1;
                    }
                }
                let Some(analysis) = unit.analysis.as_ref() else {
                    continue;
                };
                if is_shadow_or_synthetic_analysis(&analysis.dictionary_id) {
                    continue;
                }
                let mut surface_index = 0_usize;
                for morpheme in &analysis.morphemes {
                    if morpheme.span.is_empty() || morpheme.surface.is_empty() {
                        continue;
                    }
                    let surface = unit_slice(document, morpheme.span)?.to_vec();
                    if surface_index == 0 {
                        let token = root_token(&analysis.dictionary_id, &surface);
                        *root_counts.entry(token.clone()).or_insert(0) += 1;
                        roots
                            .entry(token)
                            .or_insert_with(|| (analysis.dictionary_id.clone(), surface));
                    } else {
                        let token = morpheme_token(&morpheme.id, &surface);
                        morphemes
                            .entry(token)
                            .or_insert_with(|| (morpheme.id.clone(), surface));
                    }
                    surface_index += 1;
                }
            }
        }

        let mut entries = CONTROL_TOKENS
            .iter()
            .map(|token| NedoFormerVocabEntry {
                kind: NedoFormerVocabKind::Control,
                token: (*token).to_owned(),
                surface: Vec::new(),
            })
            .collect::<Vec<_>>();
        for value in 0_u8..=255 {
            entries.push(NedoFormerVocabEntry {
                kind: NedoFormerVocabKind::Byte,
                token: format!("<byte:{value:02x}>"),
                surface: vec![value],
            });
        }

        let mut ranked_chars = char_counts.into_iter().collect::<Vec<_>>();
        ranked_chars.sort_unstable_by(|left, right| {
            right
                .1
                .cmp(&left.1)
                .then_with(|| (left.0 as u32).cmp(&(right.0 as u32)))
        });
        ranked_chars.truncate(max_chars);
        let mut chars = ranked_chars
            .into_iter()
            .map(|entry| entry.0)
            .collect::<Vec<_>>();
        chars.sort_unstable();
        for value in chars {
            entries.push(NedoFormerVocabEntry {
                kind: NedoFormerVocabKind::Character,
                token: format!("<char:{:x}>", value as u32),
                surface: value.to_string().into_bytes(),
            });
        }

        for width in 1..=3 {
            let limit = 10_usize.pow(width);
            for value in 0..limit {
                let surface = format!("{value:0width$}", width = width as usize);
                entries.push(NedoFormerVocabEntry {
                    kind: NedoFormerVocabKind::Number,
                    token: format!("<num:{surface}>"),
                    surface: surface.into_bytes(),
                });
            }
        }

        let mut morpheme_rows = morphemes.into_iter().collect::<Vec<_>>();
        morpheme_rows.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        for (token, (_, surface)) in morpheme_rows {
            entries.push(NedoFormerVocabEntry {
                kind: NedoFormerVocabKind::Morpheme,
                token,
                surface,
            });
        }

        let mut root_rows = roots
            .into_iter()
            .map(|(token, (_, surface))| {
                let count = root_counts.get(&token).copied().unwrap_or(0);
                (token, surface, count)
            })
            .collect::<Vec<_>>();
        root_rows.sort_unstable_by(|left, right| {
            right.2.cmp(&left.2).then_with(|| left.0.cmp(&right.0))
        });
        root_rows.truncate(max_roots);
        for (token, surface, _) in root_rows {
            entries.push(NedoFormerVocabEntry {
                kind: NedoFormerVocabKind::Root,
                token,
                surface,
            });
        }

        let mut code_rows = code_counts.into_iter().collect::<Vec<_>>();
        code_rows.sort_unstable_by(|left, right| {
            right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0))
        });
        code_rows.truncate(max_code_pieces);
        for (surface, _) in code_rows {
            entries.push(NedoFormerVocabEntry {
                kind: NedoFormerVocabKind::Code,
                token: code_token(&surface),
                surface,
            });
        }
        Self::from_entries(entries)
    }

    /// Number of unified output IDs.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the vocabulary is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Stable ordered entries.
    #[must_use]
    pub fn entries(&self) -> &[NedoFormerVocabEntry] {
        &self.entries
    }

    /// Looks up a stable token identity.
    #[must_use]
    pub fn id(&self, token: &str) -> Option<u16> {
        self.token_to_id.get(token).copied()
    }

    /// Encodes one selected tokenized document into the unified decoder target.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid spans or missing fixed vocabulary controls.
    pub fn encode_document(
        &self,
        document: &TokenizedDocument,
    ) -> Result<NedoFormerGenerationEncoding, TokenizerError> {
        document.validate()?;
        let mut ids = Vec::with_capacity(document.raw().len().saturating_add(16));
        ids.push(BOS_ID);
        let content = document
            .units()
            .iter()
            .enumerate()
            .filter(|(_, unit)| {
                !matches!(unit.kind, LexicalKind::Whitespace | LexicalKind::LineBreak)
            })
            .map(|entry| entry.0)
            .collect::<Vec<_>>();
        if content.is_empty() {
            self.encode_explicit_gap(document.raw(), &mut ids)?;
            ids.push(EOS_ID);
            return Ok(NedoFormerGenerationEncoding { ids });
        }

        let first = &document.units()[content[0]];
        let leading_end = usize::try_from(first.span.start)
            .map_err(|_| TokenizerError::LengthOverflow("NedoFormer leading gap"))?;
        self.encode_explicit_gap(&document.raw()[..leading_end], &mut ids)?;

        let mut in_code = false;
        for (position, &unit_index) in content.iter().enumerate() {
            let unit = &document.units()[unit_index];
            if unit.mode == TokenMode::Code && !in_code {
                ids.push(CODE_START_ID);
                in_code = true;
            } else if unit.mode != TokenMode::Code && in_code {
                ids.push(CODE_END_ID);
                in_code = false;
            }
            self.encode_unit(document, unit, &mut ids)?;
            ids.push(UNIT_END_ID);

            let next = content
                .get(position + 1)
                .map(|&index| &document.units()[index]);
            if next.is_none_or(|next| {
                unit.group_id.is_none() || next.group_id.is_none() || unit.group_id != next.group_id
            }) {
                ids.push(WORD_END_ID);
            }

            if let Some(next) = next {
                let gap_start = usize::try_from(unit.span.end)
                    .map_err(|_| TokenizerError::LengthOverflow("NedoFormer gap start"))?;
                let gap_end = usize::try_from(next.span.start)
                    .map_err(|_| TokenizerError::LengthOverflow("NedoFormer gap end"))?;
                let gap = document.raw().get(gap_start..gap_end).ok_or(
                    TokenizerError::InvalidTrainingEncoding(
                        "NedoFormer inter-unit gap is outside document",
                    ),
                )?;
                self.encode_internal_gap(gap, &mut ids)?;
            } else {
                let trailing_start = usize::try_from(unit.span.end)
                    .map_err(|_| TokenizerError::LengthOverflow("NedoFormer trailing gap"))?;
                self.encode_explicit_gap(&document.raw()[trailing_start..], &mut ids)?;
            }
        }
        if in_code {
            ids.push(CODE_END_ID);
        }
        ids.push(EOS_ID);
        let encoding = NedoFormerGenerationEncoding { ids };
        let decoded = self.decode(&encoding.ids)?;
        if decoded != document.raw() {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "NedoFormer generation vocabulary failed byte-exact round trip",
            ));
        }
        Ok(encoding)
    }

    /// Decodes one unified generation-ID sequence byte-for-byte.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid ID or malformed whitespace-run control sequence.
    pub fn decode(&self, ids: &[u16]) -> Result<Vec<u8>, TokenizerError> {
        let mut output = Vec::new();
        let mut pending_space = false;
        let mut space_digits: Option<String> = None;
        for &id in ids {
            let entry =
                self.entries
                    .get(usize::from(id))
                    .ok_or(TokenizerError::InvalidVocabulary(
                        "NedoFormer generation ID is outside vocabulary",
                    ))?;
            if let Some(digits) = space_digits.as_mut() {
                if id == SPACES_END_ID {
                    let count = digits.parse::<usize>().map_err(|_| {
                        TokenizerError::InvalidTrainingEncoding(
                            "NedoFormer whitespace-run count is invalid",
                        )
                    })?;
                    output.resize(output.len().saturating_add(count), b' ');
                    space_digits = None;
                    continue;
                }
                if entry.kind != NedoFormerVocabKind::Number {
                    return Err(TokenizerError::InvalidTrainingEncoding(
                        "NedoFormer whitespace-run expects number atoms",
                    ));
                }
                let part = std::str::from_utf8(&entry.surface)
                    .map_err(|_| TokenizerError::InvalidUtf8Unit)?;
                digits.push_str(part);
                continue;
            }
            match id {
                PAD_ID | BOS_ID | EOS_ID | WORD_END_ID | CHAR_START_ID | CHAR_END_ID
                | CODE_START_ID | CODE_END_ID => {}
                UNIT_END_ID => pending_space = true,
                GLUE_ID => pending_space = false,
                SPACE_ID => {
                    pending_space = false;
                    output.push(b' ');
                }
                SPACES_START_ID => {
                    pending_space = false;
                    space_digits = Some(String::new());
                }
                SPACES_END_ID => {
                    return Err(TokenizerError::InvalidTrainingEncoding(
                        "NedoFormer whitespace-run end appeared without a start",
                    ));
                }
                TAB_ID => {
                    pending_space = false;
                    output.push(b'\t');
                }
                LF_ID => {
                    pending_space = false;
                    output.push(b'\n');
                }
                CRLF_ID => {
                    pending_space = false;
                    output.extend_from_slice(b"\r\n");
                }
                NBSP_ID => {
                    pending_space = false;
                    output.extend_from_slice("\u{00a0}".as_bytes());
                }
                _ => {
                    if pending_space {
                        output.push(b' ');
                        pending_space = false;
                    }
                    output.extend_from_slice(&entry.surface);
                }
            }
        }
        if space_digits.is_some() {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "NedoFormer whitespace-run is unterminated",
            ));
        }
        Ok(output)
    }

    /// Checksum-protected deterministic binary vocabulary.
    ///
    /// # Errors
    ///
    /// Returns an error if a field cannot fit the stable schema.
    pub fn to_bytes(&self) -> Result<Vec<u8>, TokenizerError> {
        let mut payload = Vec::new();
        for entry in &self.entries {
            payload.push(entry.kind as u8);
            let token = entry.token.as_bytes();
            payload.extend_from_slice(
                &u32::try_from(token.len())
                    .map_err(|_| TokenizerError::LengthOverflow("NedoFormer vocab token"))?
                    .to_le_bytes(),
            );
            payload.extend_from_slice(
                &u32::try_from(entry.surface.len())
                    .map_err(|_| TokenizerError::LengthOverflow("NedoFormer vocab surface"))?
                    .to_le_bytes(),
            );
            payload.extend_from_slice(token);
            payload.extend_from_slice(&entry.surface);
        }
        let mut output = Vec::with_capacity(HEADER_BYTES + payload.len());
        output.extend_from_slice(VOCAB_MAGIC);
        output.extend_from_slice(&VOCAB_SCHEMA.to_le_bytes());
        output.extend_from_slice(
            &u32::try_from(self.entries.len())
                .map_err(|_| TokenizerError::LengthOverflow("NedoFormer vocab count"))?
                .to_le_bytes(),
        );
        output.extend_from_slice(&Sha256::digest(&payload));
        output.extend_from_slice(&payload);
        Ok(output)
    }

    /// Loads and validates the stable unified vocabulary.
    ///
    /// # Errors
    ///
    /// Returns an error for identity/checksum/UTF-8/length/fixed-entry failures.
    pub fn from_bytes(input: &[u8]) -> Result<Self, TokenizerError> {
        if input.len() < HEADER_BYTES || input.get(..8) != Some(VOCAB_MAGIC) {
            return Err(TokenizerError::InvalidVocabulary(
                "bad NedoFormer vocabulary header",
            ));
        }
        let schema = u32::from_le_bytes(
            input[8..12]
                .try_into()
                .map_err(|_| TokenizerError::InvalidVocabulary("truncated NedoFormer schema"))?,
        );
        if schema != VOCAB_SCHEMA {
            return Err(TokenizerError::InvalidVocabulary(
                "unsupported NedoFormer vocabulary schema",
            ));
        }
        let count = u32::from_le_bytes(
            input[12..16]
                .try_into()
                .map_err(|_| TokenizerError::InvalidVocabulary("truncated NedoFormer count"))?,
        ) as usize;
        let expected: [u8; 32] = input[16..48]
            .try_into()
            .map_err(|_| TokenizerError::InvalidVocabulary("truncated NedoFormer checksum"))?;
        let payload = &input[48..];
        let actual: [u8; 32] = Sha256::digest(payload).into();
        if actual != expected {
            return Err(TokenizerError::InvalidVocabulary(
                "NedoFormer vocabulary checksum mismatch",
            ));
        }
        let mut cursor = 0_usize;
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            let kind = NedoFormerVocabKind::from_u8(
                *take(payload, &mut cursor, 1)?
                    .first()
                    .ok_or(TokenizerError::InvalidVocabulary("missing NedoFormer kind"))?,
            )?;
            let token_len = read_u32(payload, &mut cursor)? as usize;
            let surface_len = read_u32(payload, &mut cursor)? as usize;
            let token = std::str::from_utf8(take(payload, &mut cursor, token_len)?)
                .map_err(|_| TokenizerError::InvalidVocabulary("invalid NedoFormer token UTF-8"))?
                .to_owned();
            let surface = take(payload, &mut cursor, surface_len)?.to_vec();
            entries.push(NedoFormerVocabEntry {
                kind,
                token,
                surface,
            });
        }
        if cursor != payload.len() {
            return Err(TokenizerError::InvalidVocabulary(
                "trailing NedoFormer vocabulary bytes",
            ));
        }
        Self::from_entries(entries)
    }

    /// SHA-256 of the stable serialized vocabulary.
    ///
    /// # Errors
    ///
    /// Returns an error only if serialization fails.
    pub fn fingerprint(&self) -> Result<[u8; 32], TokenizerError> {
        Ok(Sha256::digest(self.to_bytes()?).into())
    }

    fn from_entries(entries: Vec<NedoFormerVocabEntry>) -> Result<Self, TokenizerError> {
        if entries.len() > usize::from(u16::MAX) + 1 {
            return Err(TokenizerError::InvalidVocabulary(
                "NedoFormer unified vocabulary exceeds u16 ID space",
            ));
        }
        if entries.len() < CONTROL_TOKENS.len() + 256 + 1110 {
            return Err(TokenizerError::InvalidVocabulary(
                "NedoFormer vocabulary is missing fixed entries",
            ));
        }
        for (index, expected) in CONTROL_TOKENS.iter().enumerate() {
            let entry = &entries[index];
            if entry.kind != NedoFormerVocabKind::Control
                || entry.token != *expected
                || !entry.surface.is_empty()
            {
                return Err(TokenizerError::InvalidVocabulary(
                    "NedoFormer fixed control IDs differ",
                ));
            }
        }
        let mut token_to_id = HashMap::with_capacity(entries.len());
        let mut char_to_id = HashMap::new();
        let mut byte_ids = [0_u16; 256];
        let mut number_to_id = HashMap::new();
        let mut seen = BTreeSet::new();
        for (index, entry) in entries.iter().enumerate() {
            if !seen.insert(entry.token.clone()) {
                return Err(TokenizerError::InvalidVocabulary(
                    "NedoFormer vocabulary token identities are not unique",
                ));
            }
            let id = u16::try_from(index)
                .map_err(|_| TokenizerError::LengthOverflow("NedoFormer vocabulary ID"))?;
            token_to_id.insert(entry.token.clone(), id);
            match entry.kind {
                NedoFormerVocabKind::Byte => {
                    if entry.surface.len() != 1 {
                        return Err(TokenizerError::InvalidVocabulary(
                            "NedoFormer byte entry is not one byte",
                        ));
                    }
                    byte_ids[usize::from(entry.surface[0])] = id;
                }
                NedoFormerVocabKind::Character => {
                    let text = std::str::from_utf8(&entry.surface).map_err(|_| {
                        TokenizerError::InvalidVocabulary("invalid character entry")
                    })?;
                    let mut values = text.chars();
                    let value = values.next().ok_or(TokenizerError::InvalidVocabulary(
                        "empty NedoFormer character entry",
                    ))?;
                    if values.next().is_some() {
                        return Err(TokenizerError::InvalidVocabulary(
                            "multi-scalar NedoFormer character entry",
                        ));
                    }
                    char_to_id.insert(value, id);
                }
                NedoFormerVocabKind::Number => {
                    let text = std::str::from_utf8(&entry.surface)
                        .map_err(|_| TokenizerError::InvalidVocabulary("invalid number entry"))?;
                    if text.is_empty()
                        || text.len() > 3
                        || !text.bytes().all(|value| value.is_ascii_digit())
                    {
                        return Err(TokenizerError::InvalidVocabulary(
                            "invalid NedoFormer number atom",
                        ));
                    }
                    number_to_id.insert(text.to_owned(), id);
                }
                _ => {}
            }
        }
        for (value, &byte_id) in byte_ids.iter().enumerate() {
            let expected_index = CONTROL_TOKENS.len() + value;
            if usize::from(byte_id) != expected_index {
                return Err(TokenizerError::InvalidVocabulary(
                    "NedoFormer fixed byte IDs differ",
                ));
            }
        }
        Ok(Self {
            entries,
            token_to_id,
            char_to_id,
            byte_ids,
            number_to_id,
        })
    }

    fn encode_unit(
        &self,
        document: &TokenizedDocument,
        unit: &TokenizedUnit,
        ids: &mut Vec<u16>,
    ) -> Result<(), TokenizerError> {
        let bytes = unit_bytes(document, unit)?;
        if unit.mode == TokenMode::Code {
            for piece in unit_piece_bytes(document, unit)? {
                if let Some(id) = self.id(&code_token(piece)) {
                    ids.push(id);
                } else {
                    self.encode_char_escape(piece, ids);
                }
            }
            return Ok(());
        }
        if unit.status == TokenStatus::Morphological {
            if let Some(analysis) = &unit.analysis {
                if !is_shadow_or_synthetic_analysis(&analysis.dictionary_id) {
                    return self.encode_morphological_unit(document, unit, analysis, ids);
                }
            }
        }
        if unit.kind == LexicalKind::Number && bytes.iter().all(u8::is_ascii_digit) {
            return self.encode_digits(bytes, ids);
        }
        if matches!(unit.kind, LexicalKind::Punctuation | LexicalKind::Symbol) {
            self.encode_atoms(bytes, ids);
        } else {
            self.encode_char_escape(bytes, ids);
        }
        Ok(())
    }

    fn encode_morphological_unit(
        &self,
        document: &TokenizedDocument,
        unit: &TokenizedUnit,
        analysis: &super::AnalysisMetadata,
        ids: &mut Vec<u16>,
    ) -> Result<(), TokenizerError> {
        let mut cursor = unit.span.start;
        let mut surface_index = 0_usize;
        for morpheme in &analysis.morphemes {
            if morpheme.span.is_empty() || morpheme.surface.is_empty() {
                continue;
            }
            if morpheme.span.start > cursor {
                self.encode_atoms(
                    unit_slice(
                        document,
                        super::ByteSpan {
                            start: cursor,
                            end: morpheme.span.start,
                        },
                    )?,
                    ids,
                );
            }
            let surface = unit_slice(document, morpheme.span)?;
            if surface_index == 0 {
                if surface.iter().all(u8::is_ascii_digit) {
                    self.encode_digits(surface, ids)?;
                } else if let Some(id) = self.id(&root_token(&analysis.dictionary_id, surface)) {
                    ids.push(id);
                } else {
                    self.encode_char_escape(surface, ids);
                }
            } else if let Some(id) = self.id(&morpheme_token(&morpheme.id, surface)) {
                ids.push(id);
            } else {
                self.encode_char_escape(surface, ids);
            }
            cursor = morpheme.span.end;
            surface_index += 1;
        }
        if cursor < unit.span.end {
            self.encode_atoms(
                unit_slice(
                    document,
                    super::ByteSpan {
                        start: cursor,
                        end: unit.span.end,
                    },
                )?,
                ids,
            );
        }
        if surface_index == 0 {
            self.encode_char_escape(unit_bytes(document, unit)?, ids);
        }
        Ok(())
    }

    fn encode_internal_gap(&self, gap: &[u8], ids: &mut Vec<u16>) -> Result<(), TokenizerError> {
        match gap {
            [] => ids.push(GLUE_ID),
            [b' '] => {}
            _ => self.encode_explicit_gap(gap, ids)?,
        }
        Ok(())
    }

    fn encode_explicit_gap(&self, gap: &[u8], ids: &mut Vec<u16>) -> Result<(), TokenizerError> {
        let mut index = 0_usize;
        while index < gap.len() {
            if gap[index] == b' ' {
                let start = index;
                while index < gap.len() && gap[index] == b' ' {
                    index += 1;
                }
                let count = index - start;
                if count == 1 {
                    ids.push(SPACE_ID);
                } else {
                    ids.push(SPACES_START_ID);
                    self.encode_decimal_count(count, ids)?;
                    ids.push(SPACES_END_ID);
                }
                continue;
            }
            if gap[index..].starts_with(b"\r\n") {
                ids.push(CRLF_ID);
                index += 2;
                continue;
            }
            if gap[index] == b'\n' {
                ids.push(LF_ID);
                index += 1;
                continue;
            }
            if gap[index] == b'\t' {
                ids.push(TAB_ID);
                index += 1;
                continue;
            }
            if gap[index..].starts_with("\u{00a0}".as_bytes()) {
                ids.push(NBSP_ID);
                index += "\u{00a0}".len();
                continue;
            }
            let width = decode_scalar(gap, index).map_or(1, |(_, width)| width);
            self.encode_atoms(&gap[index..index + width], ids);
            index += width;
        }
        Ok(())
    }

    fn encode_char_escape(&self, bytes: &[u8], ids: &mut Vec<u16>) {
        ids.push(CHAR_START_ID);
        self.encode_atoms(bytes, ids);
        ids.push(CHAR_END_ID);
    }

    fn encode_atoms(&self, bytes: &[u8], ids: &mut Vec<u16>) {
        let mut position = 0_usize;
        while position < bytes.len() {
            if let Some((value, width)) = decode_scalar(bytes, position) {
                if let Some(id) = self.char_to_id.get(&value).copied() {
                    ids.push(id);
                } else {
                    for byte in &bytes[position..position + width] {
                        ids.push(self.byte_ids[usize::from(*byte)]);
                    }
                }
                position += width;
            } else {
                ids.push(self.byte_ids[usize::from(bytes[position])]);
                position += 1;
            }
        }
    }

    fn encode_digits(&self, bytes: &[u8], ids: &mut Vec<u16>) -> Result<(), TokenizerError> {
        let text = std::str::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8Unit)?;
        if text.is_empty() || !text.bytes().all(|value| value.is_ascii_digit()) {
            return Err(TokenizerError::InvalidTrainingEncoding(
                "NedoFormer numeric atom input is not decimal digits",
            ));
        }
        let first = match text.len() % 3 {
            0 => 3,
            value => value,
        };
        let mut position = 0_usize;
        let mut width = first;
        while position < text.len() {
            let end = position + width;
            let part = &text[position..end];
            ids.push(
                *self
                    .number_to_id
                    .get(part)
                    .ok_or(TokenizerError::InvalidVocabulary(
                        "missing fixed NedoFormer number atom",
                    ))?,
            );
            position = end;
            width = 3;
        }
        Ok(())
    }

    fn encode_decimal_count(&self, count: usize, ids: &mut Vec<u16>) -> Result<(), TokenizerError> {
        self.encode_digits(count.to_string().as_bytes(), ids)
    }
}

fn is_shadow_or_synthetic_analysis(dictionary_id: &str) -> bool {
    dictionary_id.starts_with("NEDO_") && dictionary_id.ends_with("_Fallback")
}

fn root_token(dictionary_id: &str, surface: &[u8]) -> String {
    format!("<root:{}:{}>", hex(dictionary_id.as_bytes()), hex(surface))
}

fn morpheme_token(morpheme_id: &str, surface: &[u8]) -> String {
    format!("<morph:{}:{}>", hex(morpheme_id.as_bytes()), hex(surface))
}

fn code_token(surface: &[u8]) -> String {
    format!("<codepiece:{}>", hex(surface))
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len().saturating_mul(2));
    for &byte in bytes {
        output.push(DIGITS[usize::from(byte >> 4)] as char);
        output.push(DIGITS[usize::from(byte & 0x0f)] as char);
    }
    output
}

fn count_chars(raw: &[u8], counts: &mut HashMap<char, u64>) {
    let mut position = 0_usize;
    while position < raw.len() {
        if let Some((value, width)) = decode_scalar(raw, position) {
            *counts.entry(value).or_insert(0) += 1;
            position += width;
        } else {
            position += 1;
        }
    }
}

fn decode_scalar(raw: &[u8], position: usize) -> Option<(char, usize)> {
    let tail = raw.get(position..)?;
    let width = match *tail.first()? {
        0x00..=0x7f => 1,
        0xc2..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf4 => 4,
        _ => return None,
    };
    let value = std::str::from_utf8(tail.get(..width)?)
        .ok()?
        .chars()
        .next()?;
    Some((value, width))
}

fn unit_piece_bytes<'a>(
    document: &'a TokenizedDocument,
    unit: &TokenizedUnit,
) -> Result<Vec<&'a [u8]>, TokenizerError> {
    let mut boundaries = Vec::with_capacity(unit.cuts.len().saturating_add(2));
    boundaries.push(unit.span.start);
    boundaries.extend(unit.cuts.iter().copied());
    boundaries.push(unit.span.end);
    let mut pieces = Vec::with_capacity(boundaries.len().saturating_sub(1));
    for window in boundaries.windows(2) {
        pieces.push(unit_slice(
            document,
            super::ByteSpan {
                start: window[0],
                end: window[1],
            },
        )?);
    }
    Ok(pieces)
}

fn unit_bytes<'a>(
    document: &'a TokenizedDocument,
    unit: &TokenizedUnit,
) -> Result<&'a [u8], TokenizerError> {
    unit_slice(document, unit.span)
}

fn unit_slice(
    document: &TokenizedDocument,
    span: super::ByteSpan,
) -> Result<&[u8], TokenizerError> {
    let start = usize::try_from(span.start)
        .map_err(|_| TokenizerError::LengthOverflow("NedoFormer span start"))?;
    let end = usize::try_from(span.end)
        .map_err(|_| TokenizerError::LengthOverflow("NedoFormer span end"))?;
    document
        .raw()
        .get(start..end)
        .ok_or(TokenizerError::UnitOutsideDocument)
}

fn take<'a>(input: &'a [u8], cursor: &mut usize, count: usize) -> Result<&'a [u8], TokenizerError> {
    let end = cursor
        .checked_add(count)
        .ok_or(TokenizerError::LengthOverflow(
            "NedoFormer vocabulary cursor",
        ))?;
    let result = input
        .get(*cursor..end)
        .ok_or(TokenizerError::InvalidVocabulary(
            "truncated NedoFormer vocabulary payload",
        ))?;
    *cursor = end;
    Ok(result)
}

fn read_u32(input: &[u8], cursor: &mut usize) -> Result<u32, TokenizerError> {
    Ok(u32::from_le_bytes(
        take(input, cursor, 4)?
            .try_into()
            .map_err(|_| TokenizerError::InvalidVocabulary("truncated NedoFormer u32"))?,
    ))
}

#[cfg(test)]
mod tests {
    use super::NedoFormerVocabulary;
    use crate::{Tokenizer, TokenizerConfig};

    #[test]
    fn unified_vocabulary_round_trips_mixed_turkish_code_and_whitespace(
    ) -> Result<(), crate::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let samples = [
            "  Ankara'da  23.07.2026!\r\n",
            "geliyor mu?\tçok güzel\u{00a0}evet",
            "```python\ndef parseHttpRequest2XX(x):\n    foo_bar = x\n    return foo_bar+1\n```",
            "cocuklarimizdan X7f9q2 😀",
        ];
        let mut documents = Vec::new();
        for sample in samples {
            documents.push(
                tokenizer
                    .nedoformer_lattice(sample.as_bytes().to_vec())?
                    .selected_document()?,
            );
        }
        let vocab = NedoFormerVocabulary::train(&documents, 16_000, 500, 4_096)?;
        assert!(vocab.len() < usize::from(u16::MAX));
        let code_surfaces = vocab
            .entries()
            .iter()
            .filter(|entry| entry.kind == super::NedoFormerVocabKind::Code)
            .map(|entry| entry.surface.as_slice())
            .collect::<Vec<_>>();
        for expected in [b"parse".as_slice(), b"Http", b"Request", b"2", b"XX", b"_"] {
            assert!(
                code_surfaces.contains(&expected),
                "missing code piece {expected:?}"
            );
        }
        for document in &documents {
            let encoded = vocab.encode_document(document)?;
            assert_eq!(vocab.decode(&encoded.ids)?, document.raw());
        }
        let bytes = vocab.to_bytes()?;
        let loaded = NedoFormerVocabulary::from_bytes(&bytes)?;
        assert_eq!(loaded.to_bytes()?, bytes);
        assert_eq!(loaded.fingerprint()?, vocab.fingerprint()?);
        Ok(())
    }
}
