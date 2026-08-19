//! Stable, checksum-protected tokenized-document codec.

use sha2::{Digest, Sha256};

use crate::{
    AlignedMorpheme, AnalysisMetadata, TokenMode, TokenStatus, TokenizedDocument, TokenizedUnit,
    TokenizerError, MODEL_SHA256, MORPHOLOGY_SHA256, TOKENIZER_SCHEMA_VERSION,
};
use nedo_core::LexicalKind;
use nedo_format::ByteSpan;

const MAGIC: &[u8; 8] = b"NDTOK001";
const HEADER_LEN: usize = 8 + 4 + 8 + 8 + 64 + 64 + 32;
const NONE_GROUP: u32 = u32::MAX;

///
/// # Errors
///
/// Returns an error if document invariants or serialized lengths are invalid.
pub fn encode(document: &TokenizedDocument) -> Result<Vec<u8>, TokenizerError> {
    document.validate()?;
    let mut payload = Vec::new();
    payload.extend_from_slice(document.raw());
    for unit in document.units() {
        write_u64(&mut payload, unit.span.start);
        write_u64(&mut payload, unit.span.end);
        payload.push(unit.kind as u8);
        payload.push(unit.mode as u8);
        payload.push(unit.status as u8);
        payload.push(u8::from(unit.analysis.is_some()));
        write_u32(&mut payload, unit.group_id.unwrap_or(NONE_GROUP));
        write_u32(&mut payload, checked_u32(unit.cuts.len(), "cut count")?);
        for cut in &unit.cuts {
            write_u64(&mut payload, *cut);
        }
        if let Some(analysis) = &unit.analysis {
            write_string(&mut payload, &analysis.canonical)?;
            write_string(&mut payload, &analysis.dictionary_id)?;
            write_string(&mut payload, &analysis.lemma)?;
            write_string(&mut payload, &analysis.primary_pos)?;
            write_string(&mut payload, &analysis.secondary_pos)?;
            write_u32(
                &mut payload,
                checked_u32(analysis.morphemes.len(), "morpheme count")?,
            );
            for morpheme in &analysis.morphemes {
                write_string(&mut payload, &morpheme.id)?;
                write_string(&mut payload, &morpheme.surface)?;
                write_u64(&mut payload, morpheme.span.start);
                write_u64(&mut payload, morpheme.span.end);
                payload.push(u8::from(morpheme.derivational));
            }
        }
    }
    let mut output = Vec::with_capacity(HEADER_LEN + payload.len());
    output.extend_from_slice(MAGIC);
    output.extend_from_slice(&TOKENIZER_SCHEMA_VERSION.to_le_bytes());
    write_u64(
        &mut output,
        checked_u64(document.raw().len(), "raw length")?,
    );
    write_u64(
        &mut output,
        checked_u64(document.units().len(), "unit count")?,
    );
    output.extend_from_slice(MORPHOLOGY_SHA256.as_bytes());
    output.extend_from_slice(MODEL_SHA256.as_bytes());
    output.extend_from_slice(&Sha256::digest(&payload));
    output.extend_from_slice(&payload);
    Ok(output)
}

///
/// # Errors
///
/// Returns an error for identity, checksum, truncation, count, UTF-8, or invariant failures.
pub fn decode(input: &[u8]) -> Result<TokenizedDocument, TokenizerError> {
    if input.len() < HEADER_LEN {
        return Err(TokenizerError::TruncatedCodec);
    }
    if input.get(..8) != Some(MAGIC) {
        return Err(TokenizerError::BadCodecMagic);
    }
    let mut reader = Reader::new(&input[8..]);
    let version = reader.u32()?;
    if version != TOKENIZER_SCHEMA_VERSION {
        return Err(TokenizerError::UnsupportedCodecVersion(version));
    }
    let raw_len = reader.usize("raw length")?;
    let unit_count = reader.usize("unit count")?;
    let morph_sha = reader.bytes(64)?;
    let model_sha = reader.bytes(64)?;
    if morph_sha != MORPHOLOGY_SHA256.as_bytes() || model_sha != MODEL_SHA256.as_bytes() {
        return Err(TokenizerError::AssetIdentityMismatch);
    }
    let expected_hash: [u8; 32] = reader
        .bytes(32)?
        .try_into()
        .map_err(|_| TokenizerError::TruncatedCodec)?;
    let payload = reader.remaining_bytes();
    let actual_hash: [u8; 32] = Sha256::digest(payload).into();
    if expected_hash != actual_hash {
        return Err(TokenizerError::CodecChecksumMismatch);
    }
    let mut payload_reader = Reader::new(payload);
    let raw = payload_reader.bytes(raw_len)?.to_vec();
    if unit_count > payload_reader.remaining() / 24 {
        return Err(TokenizerError::ImpossibleCodecCount("unit count"));
    }
    let mut units = Vec::with_capacity(unit_count);
    for _ in 0..unit_count {
        let span = ByteSpan {
            start: payload_reader.u64()?,
            end: payload_reader.u64()?,
        };
        let kind = lexical_kind(payload_reader.u8()?)?;
        let mode = TokenMode::try_from(payload_reader.u8()?)?;
        let status = TokenStatus::try_from(payload_reader.u8()?)?;
        let has_analysis = payload_reader.boolean("has_analysis")?;
        let group = payload_reader.u32()?;
        let cut_count = payload_reader.usize32("cut count")?;
        if cut_count > payload_reader.remaining() / 8 {
            return Err(TokenizerError::ImpossibleCodecCount("cut count"));
        }
        let mut cuts = Vec::with_capacity(cut_count);
        for _ in 0..cut_count {
            cuts.push(payload_reader.u64()?);
        }
        let analysis = if has_analysis {
            let canonical = payload_reader.string()?;
            let dictionary_id = payload_reader.string()?;
            let lemma = payload_reader.string()?;
            let primary_pos = payload_reader.string()?;
            let secondary_pos = payload_reader.string()?;
            let count = payload_reader.usize32("morpheme count")?;
            if count > payload_reader.remaining() / 25 {
                return Err(TokenizerError::ImpossibleCodecCount("morpheme count"));
            }
            let mut morphemes = Vec::with_capacity(count);
            for _ in 0..count {
                morphemes.push(AlignedMorpheme {
                    id: payload_reader.string()?,
                    surface: payload_reader.string()?,
                    span: ByteSpan {
                        start: payload_reader.u64()?,
                        end: payload_reader.u64()?,
                    },
                    derivational: payload_reader.boolean("derivational")?,
                });
            }
            Some(AnalysisMetadata {
                canonical,
                dictionary_id,
                lemma,
                primary_pos,
                secondary_pos,
                morphemes,
            })
        } else {
            None
        };
        units.push(TokenizedUnit {
            span,
            kind,
            mode,
            status,
            group_id: (group != NONE_GROUP).then_some(group),
            cuts,
            analysis,
        });
    }
    if payload_reader.remaining() != 0 {
        return Err(TokenizerError::TrailingCodecBytes(
            payload_reader.remaining(),
        ));
    }
    TokenizedDocument::new(raw, units)
}

fn write_string(output: &mut Vec<u8>, value: &str) -> Result<(), TokenizerError> {
    write_u32(output, checked_u32(value.len(), "string length")?);
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn checked_u32(value: usize, field: &'static str) -> Result<u32, TokenizerError> {
    u32::try_from(value).map_err(|_| TokenizerError::LengthOverflow(field))
}

fn checked_u64(value: usize, field: &'static str) -> Result<u64, TokenizerError> {
    u64::try_from(value).map_err(|_| TokenizerError::LengthOverflow(field))
}

fn write_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn write_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

const fn lexical_kind(value: u8) -> Result<LexicalKind, TokenizerError> {
    match value {
        1 => Ok(LexicalKind::Whitespace),
        2 => Ok(LexicalKind::LineBreak),
        3 => Ok(LexicalKind::Word),
        4 => Ok(LexicalKind::Number),
        5 => Ok(LexicalKind::Punctuation),
        6 => Ok(LexicalKind::Symbol),
        7 => Ok(LexicalKind::Control),
        8 => Ok(LexicalKind::Opaque),
        _ => Err(TokenizerError::InvalidCodecEnum("lexical kind", value)),
    }
}

struct Reader<'a> {
    input: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    const fn new(input: &'a [u8]) -> Self {
        Self { input, position: 0 }
    }

    const fn remaining(&self) -> usize {
        self.input.len() - self.position
    }

    fn remaining_bytes(&self) -> &'a [u8] {
        &self.input[self.position..]
    }

    fn bytes(&mut self, count: usize) -> Result<&'a [u8], TokenizerError> {
        let end = self
            .position
            .checked_add(count)
            .ok_or(TokenizerError::LengthOverflow("codec position"))?;
        let value = self
            .input
            .get(self.position..end)
            .ok_or(TokenizerError::TruncatedCodec)?;
        self.position = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, TokenizerError> {
        self.bytes(1)?
            .first()
            .copied()
            .ok_or(TokenizerError::TruncatedCodec)
    }

    fn boolean(&mut self, field: &'static str) -> Result<bool, TokenizerError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(TokenizerError::InvalidCodecBoolean { field, value }),
        }
    }

    fn u32(&mut self) -> Result<u32, TokenizerError> {
        let bytes: [u8; 4] = self
            .bytes(4)?
            .try_into()
            .map_err(|_| TokenizerError::TruncatedCodec)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, TokenizerError> {
        let bytes: [u8; 8] = self
            .bytes(8)?
            .try_into()
            .map_err(|_| TokenizerError::TruncatedCodec)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn usize(&mut self, field: &'static str) -> Result<usize, TokenizerError> {
        usize::try_from(self.u64()?).map_err(|_| TokenizerError::LengthOverflow(field))
    }

    fn usize32(&mut self, field: &'static str) -> Result<usize, TokenizerError> {
        usize::try_from(self.u32()?).map_err(|_| TokenizerError::LengthOverflow(field))
    }

    fn string(&mut self) -> Result<String, TokenizerError> {
        let length = self.usize32("string length")?;
        let value = self.bytes(length)?;
        std::str::from_utf8(value)
            .map(str::to_owned)
            .map_err(|_| TokenizerError::InvalidCodecUtf8)
    }
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::{decode, encode, HEADER_LEN};
    use crate::{Tokenizer, TokenizerConfig, TokenizerError};

    const PAYLOAD_HASH_OFFSET: usize = 8 + 4 + 8 + 8 + 64 + 64;

    fn resign(encoded: &mut [u8]) {
        let digest: [u8; 32] = Sha256::digest(&encoded[HEADER_LEN..]).into();
        encoded[PAYLOAD_HASH_OFFSET..PAYLOAD_HASH_OFFSET + 32].copy_from_slice(&digest);
    }

    #[test]
    fn rejects_noncanonical_boolean_even_with_valid_checksum() -> Result<(), TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let raw = b"evler".to_vec();
        let document = tokenizer.tokenize(raw.clone())?;
        let mut encoded = encode(&document)?;
        let has_analysis = HEADER_LEN + raw.len() + 16 + 3;
        encoded[has_analysis] = 2;
        resign(&mut encoded);
        assert!(matches!(
            decode(&encoded),
            Err(TokenizerError::InvalidCodecBoolean {
                field: "has_analysis",
                value: 2
            })
        ));
        Ok(())
    }

    #[test]
    fn rejects_mode_status_mismatch_even_with_valid_checksum() -> Result<(), TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let raw = b"evler".to_vec();
        let document = tokenizer.tokenize(raw.clone())?;
        let mut encoded = encode(&document)?;
        let mode = HEADER_LEN + raw.len() + 16 + 1;
        encoded[mode] = 2;
        resign(&mut encoded);
        assert!(matches!(
            decode(&encoded),
            Err(TokenizerError::InvalidUnitMetadata { .. })
        ));
        Ok(())
    }
}
