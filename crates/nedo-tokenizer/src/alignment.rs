//! Exact alignment of native morphological surfaces to original UTF-8 bytes.

use nedo_format::ByteSpan;
use nedo_morph_bundle::NativeAnalysis;

use crate::{AlignedMorpheme, TokenizerError};

pub struct Alignment {
    pub cuts: Vec<u64>,
    pub morphemes: Vec<AlignedMorpheme>,
}

struct NormalizedInput {
    chars: Vec<(char, usize, usize)>,
    connectors: Vec<(usize, usize)>,
}

pub fn align_analysis(
    token: &str,
    absolute_start: u64,
    analysis: &NativeAnalysis,
) -> Result<Alignment, TokenizerError> {
    let normalized_input = normalize_input(token, analysis)?;
    let (target, morpheme_lengths) = normalized_target(analysis)?;
    let normalized = normalized_input.chars;
    let connectors = normalized_input.connectors;
    let actual: String = normalized.iter().map(|entry| entry.0).collect();
    let expected: String = target.iter().collect();
    if !normalization_matches(&actual, &expected) {
        return Err(TokenizerError::AlignmentMismatch {
            token: token.to_owned(),
            expected,
            actual,
            canonical: analysis.canonical.clone(),
        });
    }

    let mut cuts = Vec::new();
    let mut aligned = Vec::with_capacity(analysis.morphemes.len());
    let mut cursor = 0_usize;
    let mut last_raw = 0_usize;
    for (morpheme, length) in analysis.morphemes.iter().zip(morpheme_lengths) {
        let start_raw = if length == 0 {
            last_raw
        } else {
            normalized.get(cursor).map_or(last_raw, |entry| entry.1)
        };
        cursor = cursor
            .checked_add(length)
            .ok_or(TokenizerError::LengthOverflow("morpheme alignment"))?;
        let end_raw = if length == 0 {
            start_raw
        } else {
            normalized
                .get(cursor - 1)
                .map_or(start_raw, |entry| entry.2)
        };
        last_raw = end_raw;
        aligned.push(AlignedMorpheme {
            id: morpheme.id.clone(),
            surface: morpheme.surface.clone(),
            span: ByteSpan {
                start: absolute_start
                    + u64::try_from(start_raw)
                        .map_err(|_| TokenizerError::LengthOverflow("aligned morpheme start"))?,
                end: absolute_start
                    + u64::try_from(end_raw)
                        .map_err(|_| TokenizerError::LengthOverflow("aligned morpheme end"))?,
            },
            derivational: morpheme.derivational,
        });
        if cursor < normalized.len() && length > 0 {
            cuts.push(
                absolute_start
                    + u64::try_from(end_raw)
                        .map_err(|_| TokenizerError::LengthOverflow("morpheme cut"))?,
            );
        }
    }
    for (start, end) in connectors {
        cuts.push(
            absolute_start
                + u64::try_from(start)
                    .map_err(|_| TokenizerError::LengthOverflow("connector start"))?,
        );
        cuts.push(
            absolute_start
                + u64::try_from(end)
                    .map_err(|_| TokenizerError::LengthOverflow("connector end"))?,
        );
    }
    cuts.sort_unstable();
    cuts.dedup();
    let token_end = absolute_start
        + u64::try_from(token.len()).map_err(|_| TokenizerError::LengthOverflow("token end"))?;
    cuts.retain(|cut| *cut > absolute_start && *cut < token_end);
    Ok(Alignment {
        cuts,
        morphemes: aligned,
    })
}

fn normalize_input(
    token: &str,
    analysis: &NativeAnalysis,
) -> Result<NormalizedInput, TokenizerError> {
    let ignore_dots = analysis.secondary_pos == "RegAbbrv" || is_dotted_abbreviation(token);
    let mut chars = Vec::new();
    let mut connectors = Vec::new();
    for (start, value) in token.char_indices() {
        let end = start + value.len_utf8();
        if is_apostrophe(value) {
            connectors.push((start, end));
        } else if !(ignore_dots && value == '.') {
            chars.push((normalize_char(value)?, start, end));
        }
    }
    Ok(NormalizedInput { chars, connectors })
}

fn normalized_target(analysis: &NativeAnalysis) -> Result<(Vec<char>, Vec<usize>), TokenizerError> {
    let mut target = Vec::new();
    let mut lengths = Vec::with_capacity(analysis.morphemes.len());
    for morpheme in &analysis.morphemes {
        let before = target.len();
        for value in morpheme.surface.chars() {
            target.push(normalize_char(value)?);
        }
        lengths.push(target.len() - before);
    }
    Ok((target, lengths))
}

fn normalize_char(value: char) -> Result<char, TokenizerError> {
    let first = if value == 'I' {
        'ı'
    } else if value == 'İ' {
        'i'
    } else {
        let mut lower = value.to_lowercase();
        let first = lower.next().unwrap_or(value);
        if lower.next().is_some() {
            return Err(TokenizerError::UnsupportedNormalization { value });
        }
        first
    };
    Ok(normalize_circumflex(first))
}

fn normalization_matches(actual: &str, expected: &str) -> bool {
    if actual == expected {
        return true;
    }
    let foreign_mapped: String = actual.chars().map(foreign_diacritic_to_turkish).collect();
    if foreign_mapped == expected {
        return true;
    }
    foreign_mapped
        .chars()
        .map(zemberek_runtime_normalized_char)
        .eq(expected.chars())
}

const fn normalize_circumflex(value: char) -> char {
    match value {
        'â' => 'a',
        'î' => 'i',
        'û' => 'u',
        _ => value,
    }
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

const fn zemberek_runtime_normalized_char(value: char) -> char {
    if is_zemberek_normalized_character(value) {
        value
    } else {
        '?'
    }
}

const fn is_zemberek_normalized_character(value: char) -> bool {
    matches!(
        value,
        'a'..='z' | 'ç' | 'ğ' | 'ı' | 'ö' | 'ş' | 'ü' | 'â' | 'î' | 'û' | '.' | '-'
    )
}

const fn is_apostrophe(value: char) -> bool {
    matches!(
        value,
        '\'' | '\u{2018}' | '\u{2019}' | '\u{02bc}' | '\u{ff07}'
    )
}

fn is_dotted_abbreviation(token: &str) -> bool {
    let stem = token
        .split_once(['\'', '\u{2019}'])
        .map_or(token, |value| value.0);
    let mut letters = stem.chars().filter(|value| *value != '.');
    stem.contains('.')
        && letters.next().is_some_and(char::is_alphabetic)
        && letters.all(char::is_alphabetic)
}

#[cfg(test)]
mod tests {
    use super::is_dotted_abbreviation;

    #[test]
    fn punctuation_is_not_a_dotted_abbreviation() {
        assert!(!is_dotted_abbreviation("."));
        assert!(!is_dotted_abbreviation("..."));
        assert!(is_dotted_abbreviation("T.B.M.M."));
        assert!(is_dotted_abbreviation("A.B.D.'nin"));
    }

    #[test]
    fn zemberek_runtime_unknown_character_set_is_explicit() {
        assert!(super::is_zemberek_normalized_character('ü'));
        assert!(super::is_zemberek_normalized_character('q'));
        assert!(!super::is_zemberek_normalized_character('ŕ'));
        assert!(!super::is_zemberek_normalized_character('7'));
        assert_eq!(super::normalize_circumflex('â'), 'a');
        assert_eq!(super::foreign_diacritic_to_turkish('á'), 'a');
        assert!(super::normalization_matches("m1", "m1"));
        assert!(super::normalization_matches("uáña", "uana"));
        assert!(super::normalization_matches("uŕka", "u?ka"));
        assert!(super::normalization_matches("m1", "m?"));
        assert!(super::normalization_matches("\u{0340}u55a", "?u??a"));
    }
}
