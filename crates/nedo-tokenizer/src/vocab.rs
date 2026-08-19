//! Deterministic character/byte-fallback and generation vocabularies.

use std::collections::{BTreeSet, HashMap};

use sha2::{Digest, Sha256};

use crate::{TokenMode, TokenizedDocument, TokenizerError};

/// Padding ID.
pub const PAD_ID: u32 = 0;
/// Beginning-of-document ID.
pub const BOS_ID: u32 = 1;
/// End-of-document ID.
pub const EOS_ID: u32 = 2;
/// Lexical-unit boundary ID.
pub const UNIT_BOUNDARY_ID: u32 = 3;
/// Morpheme boundary ID.
pub const MORPHEME_BOUNDARY_ID: u32 = 4;
/// Code-mode entry ID.
pub const CODE_START_ID: u32 = 5;
/// Code-mode exit ID.
pub const CODE_END_ID: u32 = 6;
/// First byte fallback ID; byte `b` is `BYTE_BASE_ID + b`.
pub const BYTE_BASE_ID: u32 = 7;
/// First learned Unicode-scalar ID.
pub const CHAR_BASE_ID: u32 = BYTE_BASE_ID + 256;

const CHAR_MAGIC: &[u8; 8] = b"NDCHR001";
const GEN_MAGIC: &[u8; 8] = b"NDGEN001";

/// Deterministic Unicode-scalar vocabulary with complete byte fallback.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CharacterVocabulary {
    chars: Vec<char>,
}

impl CharacterVocabulary {
    /// Trains a deterministic vocabulary from exact document bytes.
    ///
    /// Characters are ranked by descending frequency and scalar value as the
    /// deterministic tie-break. Invalid UTF-8 remains representable by bytes.
    #[must_use]
    pub fn train(documents: &[TokenizedDocument], max_chars: usize) -> Self {
        let mut counts = HashMap::<char, u64>::new();
        for document in documents {
            let mut position = 0_usize;
            while position < document.raw().len() {
                if let Some((value, width)) = decode_scalar(document.raw(), position) {
                    *counts.entry(value).or_insert(0) += 1;
                    position += width;
                } else {
                    position += 1;
                }
            }
        }
        let mut ranked: Vec<(char, u64)> = counts.into_iter().collect();
        ranked.sort_unstable_by(|left, right| {
            right
                .1
                .cmp(&left.1)
                .then_with(|| (left.0 as u32).cmp(&(right.0 as u32)))
        });
        ranked.truncate(max_chars);
        let mut chars: Vec<char> = ranked.into_iter().map(|entry| entry.0).collect();
        chars.sort_unstable();
        Self { chars }
    }

    /// Builds a vocabulary from an already sorted unique character list.
    ///
    /// # Errors
    ///
    /// Returns an error if the list is not strictly sorted.
    pub fn from_sorted(chars: Vec<char>) -> Result<Self, TokenizerError> {
        if chars.windows(2).any(|window| window[0] >= window[1]) {
            return Err(TokenizerError::InvalidVocabulary(
                "characters are not strictly sorted",
            ));
        }
        Ok(Self { chars })
    }

    /// Learned character count, excluding fixed specials and bytes.
    #[must_use]
    pub const fn char_count(&self) -> usize {
        self.chars.len()
    }

    /// Total embedding vocabulary size.
    #[must_use]
    pub fn len(&self) -> usize {
        usize::try_from(CHAR_BASE_ID).unwrap_or(263) + self.chars.len()
    }

    /// Whether no learned character is present.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    /// Returns the ID for a learned character.
    #[must_use]
    pub fn id_for_char(&self, value: char) -> Option<u32> {
        self.chars
            .binary_search(&value)
            .ok()
            .and_then(|index| u32::try_from(index).ok())
            .and_then(|index| CHAR_BASE_ID.checked_add(index))
    }

    /// Encodes a tokenized document with explicit unit, morpheme, and code boundaries.
    ///
    /// The sequence is deterministic. Any unlisted scalar and invalid UTF-8 byte
    /// is represented by fixed byte IDs, so coverage is complete.
    #[must_use]
    pub fn encode(&self, document: &TokenizedDocument) -> Vec<u32> {
        let mut ids = vec![BOS_ID];
        let mut in_code = false;
        for unit in document.units() {
            if unit.mode == TokenMode::Code && !in_code {
                ids.push(CODE_START_ID);
                in_code = true;
            } else if unit.mode != TokenMode::Code && in_code {
                ids.push(CODE_END_ID);
                in_code = false;
            }
            ids.push(UNIT_BOUNDARY_ID);
            let mut cuts = unit.cuts.iter().copied().peekable();
            let start = usize::try_from(unit.span.start).unwrap_or(0);
            let end = usize::try_from(unit.span.end).unwrap_or(start);
            let mut position = start;
            while position < end {
                let absolute = u64::try_from(position).unwrap_or(u64::MAX);
                while cuts.peek().is_some_and(|cut| *cut == absolute) {
                    ids.push(MORPHEME_BOUNDARY_ID);
                    cuts.next();
                }
                if let Some((value, width)) = decode_scalar(document.raw(), position) {
                    if position + width <= end {
                        if let Some(id) = self.id_for_char(value) {
                            ids.push(id);
                        } else {
                            for byte in &document.raw()[position..position + width] {
                                ids.push(BYTE_BASE_ID + u32::from(*byte));
                            }
                        }
                        position += width;
                        continue;
                    }
                }
                ids.push(BYTE_BASE_ID + u32::from(document.raw()[position]));
                position += 1;
            }
        }
        if in_code {
            ids.push(CODE_END_ID);
        }
        ids.push(EOS_ID);
        ids
    }

    /// Stable checksum-protected encoding.
    ///
    /// # Errors
    ///
    /// Returns an error only if the learned scalar count cannot fit the stable format.
    pub fn to_bytes(&self) -> Result<Vec<u8>, TokenizerError> {
        let mut payload = Vec::with_capacity(self.chars.len() * 4);
        for value in &self.chars {
            payload.extend_from_slice(&(*value as u32).to_le_bytes());
        }
        let mut output = Vec::with_capacity(8 + 4 + 32 + payload.len());
        output.extend_from_slice(CHAR_MAGIC);
        output.extend_from_slice(
            &u32::try_from(self.chars.len())
                .map_err(|_| TokenizerError::LengthOverflow("character vocabulary count"))?
                .to_le_bytes(),
        );
        output.extend_from_slice(&Sha256::digest(&payload));
        output.extend_from_slice(&payload);
        Ok(output)
    }

    /// Decodes a stable character vocabulary.
    ///
    /// # Errors
    ///
    /// Returns an error for identity, checksum, length, scalar, or ordering failures.
    pub fn from_bytes(input: &[u8]) -> Result<Self, TokenizerError> {
        if input.len() < 44 || input.get(..8) != Some(CHAR_MAGIC) {
            return Err(TokenizerError::InvalidVocabulary(
                "bad character vocabulary header",
            ));
        }
        let count = u32::from_le_bytes(
            input[8..12]
                .try_into()
                .map_err(|_| TokenizerError::InvalidVocabulary("truncated character count"))?,
        ) as usize;
        let expected: [u8; 32] = input[12..44]
            .try_into()
            .map_err(|_| TokenizerError::InvalidVocabulary("truncated character hash"))?;
        let payload = &input[44..];
        if payload.len() != count.saturating_mul(4) {
            return Err(TokenizerError::InvalidVocabulary(
                "character payload length mismatch",
            ));
        }
        let actual: [u8; 32] = Sha256::digest(payload).into();
        if actual != expected {
            return Err(TokenizerError::InvalidVocabulary(
                "character payload checksum mismatch",
            ));
        }
        let mut chars = Vec::with_capacity(count);
        for chunk in payload.chunks_exact(4) {
            let scalar = u32::from_le_bytes(
                chunk
                    .try_into()
                    .map_err(|_| TokenizerError::InvalidVocabulary("truncated scalar"))?,
            );
            chars.push(
                char::from_u32(scalar)
                    .ok_or(TokenizerError::InvalidVocabulary("invalid Unicode scalar"))?,
            );
        }
        Self::from_sorted(chars)
    }
}

/// Root/morpheme generation vocabulary used by the inner decoder.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationVocabulary {
    entries: Vec<String>,
}

impl GenerationVocabulary {
    /// Trains a deterministic root+morpheme vocabulary.
    ///
    /// All observed morpheme IDs are retained. Roots are ranked by descending
    /// selected-analysis frequency, then lexical ID; numeric atoms `0..999` and
    /// required control symbols are always present.
    #[must_use]
    pub fn train(documents: &[TokenizedDocument], max_roots: usize) -> Self {
        let mut root_counts = HashMap::<String, u64>::new();
        let mut morphemes = BTreeSet::<String>::new();
        for document in documents {
            for unit in document.units() {
                let Some(analysis) = &unit.analysis else {
                    continue;
                };
                *root_counts
                    .entry(analysis.dictionary_id.clone())
                    .or_insert(0) += 1;
                for morpheme in &analysis.morphemes {
                    morphemes.insert(morpheme.id.clone());
                }
            }
        }
        let mut roots: Vec<(String, u64)> = root_counts.into_iter().collect();
        roots.sort_unstable_by(|left, right| {
            right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0))
        });
        roots.truncate(max_roots);
        let mut entries = [
            "<pad>",
            "<bos>",
            "<eos>",
            "<word_end>",
            "<char>",
            "<code>",
            "<text>",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
        entries.extend((0..=999).map(|value| format!("<num:{value}>")));
        entries.extend(morphemes.into_iter().map(|id| format!("<m:{id}>")));
        entries.extend(roots.into_iter().map(|entry| format!("<r:{}>", entry.0)));
        Self { entries }
    }

    /// Number of output symbols.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the vocabulary is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns ordered entries.
    #[must_use]
    pub fn entries(&self) -> &[String] {
        &self.entries
    }

    /// Looks up an exact symbol ID.
    #[must_use]
    pub fn id(&self, value: &str) -> Option<u32> {
        self.entries
            .iter()
            .position(|entry| entry == value)
            .and_then(|index| u32::try_from(index).ok())
    }

    /// Stable checksum-protected encoding.
    ///
    /// # Errors
    ///
    /// Returns an error when an entry or collection length exceeds the format.
    pub fn to_bytes(&self) -> Result<Vec<u8>, TokenizerError> {
        let mut payload = Vec::new();
        for entry in &self.entries {
            let length = u32::try_from(entry.len())
                .map_err(|_| TokenizerError::LengthOverflow("generation entry"))?;
            payload.extend_from_slice(&length.to_le_bytes());
            payload.extend_from_slice(entry.as_bytes());
        }
        let mut output = Vec::new();
        output.extend_from_slice(GEN_MAGIC);
        output.extend_from_slice(
            &u32::try_from(self.entries.len())
                .map_err(|_| TokenizerError::LengthOverflow("generation count"))?
                .to_le_bytes(),
        );
        output.extend_from_slice(&Sha256::digest(&payload));
        output.extend_from_slice(&payload);
        Ok(output)
    }

    /// Decodes the stable generation vocabulary.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed identity, checksum, lengths, UTF-8, or trailing bytes.
    pub fn from_bytes(input: &[u8]) -> Result<Self, TokenizerError> {
        if input.len() < 44 || input.get(..8) != Some(GEN_MAGIC) {
            return Err(TokenizerError::InvalidVocabulary(
                "bad generation vocabulary header",
            ));
        }
        let count = u32::from_le_bytes(
            input[8..12]
                .try_into()
                .map_err(|_| TokenizerError::InvalidVocabulary("truncated generation count"))?,
        ) as usize;
        let expected: [u8; 32] = input[12..44]
            .try_into()
            .map_err(|_| TokenizerError::InvalidVocabulary("truncated generation hash"))?;
        let payload = &input[44..];
        let actual: [u8; 32] = Sha256::digest(payload).into();
        if actual != expected {
            return Err(TokenizerError::InvalidVocabulary(
                "generation checksum mismatch",
            ));
        }
        let mut position = 0_usize;
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            let end_length = position
                .checked_add(4)
                .ok_or(TokenizerError::LengthOverflow("generation position"))?;
            let length_bytes =
                payload
                    .get(position..end_length)
                    .ok_or(TokenizerError::InvalidVocabulary(
                        "truncated generation length",
                    ))?;
            let length = u32::from_le_bytes(
                length_bytes
                    .try_into()
                    .map_err(|_| TokenizerError::InvalidVocabulary("bad generation length"))?,
            ) as usize;
            position = end_length;
            let end = position
                .checked_add(length)
                .ok_or(TokenizerError::LengthOverflow("generation string"))?;
            let value = payload
                .get(position..end)
                .ok_or(TokenizerError::InvalidVocabulary(
                    "truncated generation entry",
                ))?;
            entries.push(
                std::str::from_utf8(value)
                    .map_err(|_| TokenizerError::InvalidVocabulary("invalid generation UTF-8"))?
                    .to_owned(),
            );
            position = end;
        }
        if position != payload.len() {
            return Err(TokenizerError::InvalidVocabulary(
                "trailing generation bytes",
            ));
        }
        validate_generation_entries(&entries)?;
        Ok(Self { entries })
    }
}

fn validate_generation_entries(entries: &[String]) -> Result<(), TokenizerError> {
    const CONTROLS: [&str; 7] = [
        "<pad>",
        "<bos>",
        "<eos>",
        "<word_end>",
        "<char>",
        "<code>",
        "<text>",
    ];
    if entries.len() < CONTROLS.len() + 1000 {
        return Err(TokenizerError::InvalidVocabulary(
            "generation vocabulary is missing fixed entries",
        ));
    }
    if entries[..CONTROLS.len()]
        .iter()
        .map(String::as_str)
        .ne(CONTROLS)
    {
        return Err(TokenizerError::InvalidVocabulary(
            "generation control IDs differ",
        ));
    }
    for value in 0..=999 {
        if entries[CONTROLS.len() + value] != format!("<num:{value}>") {
            return Err(TokenizerError::InvalidVocabulary(
                "generation numeric IDs differ",
            ));
        }
    }
    let unique = entries.iter().collect::<BTreeSet<_>>();
    if unique.len() != entries.len() {
        return Err(TokenizerError::InvalidVocabulary(
            "generation entries are not unique",
        ));
    }
    Ok(())
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
    let bytes = tail.get(..width)?;
    let value = std::str::from_utf8(bytes).ok()?.chars().next()?;
    Some((value, width))
}

#[cfg(test)]
mod tests {
    use super::{CharacterVocabulary, BYTE_BASE_ID};
    use crate::{Tokenizer, TokenizerConfig};

    #[test]
    fn character_vocabulary_is_deterministic_and_byte_complete() -> Result<(), crate::TokenizerError>
    {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let documents = vec![tokenizer.tokenize("İstanbul 😀".as_bytes().to_vec())?];
        let first = CharacterVocabulary::train(&documents, 4);
        let second = CharacterVocabulary::train(&documents, 4);
        assert_eq!(first, second);
        assert_eq!(CharacterVocabulary::from_bytes(&first.to_bytes()?)?, first);
        let ids = first.encode(&documents[0]);
        assert!(ids.iter().any(|id| *id >= BYTE_BASE_ID));
        Ok(())
    }
}
