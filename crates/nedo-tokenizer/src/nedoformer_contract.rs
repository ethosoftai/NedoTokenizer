//! Unified semantic fingerprint for the `NedoFormer` tokenizer/segmenter contract.

use sha2::{Digest, Sha256};

use super::{
    CharacterVocabulary, NedoFormerVocabulary, Tokenizer, TokenizerError, MODEL_SHA256,
    MORPHOLOGY_SHA256, TOKENIZER_SCHEMA_VERSION,
};

/// Current `NedoFormer` tokenizer contract schema.
pub const NEDOFORMER_TOKENIZER_CONTRACT_VERSION: u32 = 1;
/// Byte-mapped Turkish lowercase/deasciify shadow behavior revision.
pub const NEDOFORMER_SHADOW_NORMALIZATION_VERSION: u32 = 1;
/// Code-span and identifier-splitting behavior revision.
pub const NEDOFORMER_CODE_SEGMENTATION_VERSION: u32 = 1;
/// Number/date/apostrophe micro-segmentation behavior revision.
pub const NEDOFORMER_NUMERIC_SEGMENTATION_VERSION: u32 = 1;
/// Implicit-space, glue and explicit whitespace-run grammar revision.
pub const NEDOFORMER_WHITESPACE_SCHEMA_VERSION: u32 = 1;
/// Multi-candidate segmentation lattice and sampling revision.
pub const NEDOFORMER_LATTICE_SCHEMA_VERSION: u32 = 1;
/// Inner-character stream, recurrent reset and pooling-metadata revision.
pub const NEDOFORMER_INPUT_ENCODING_VERSION: u32 = 1;

/// SHA-256 identity of the tokenizer-side `NedoFormer` contract.
///
/// A complete model checkpoint must additionally bind decoder-side morphology
/// constraints such as the FSM/allomorph/pronunciation assets described by the
/// architecture. Those assets are intentionally not implemented by this tokenizer crate.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NedoFormerContractFingerprint([u8; 32]);

impl NedoFormerContractFingerprint {
    /// Raw SHA-256 bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Lowercase hexadecimal digest.
    #[must_use]
    pub fn hex(&self) -> String {
        let mut output = String::with_capacity(64);
        for byte in self.0 {
            use std::fmt::Write as _;
            let _ = write!(&mut output, "{byte:02x}");
        }
        output
    }

    /// Builds a fingerprint from exact bytes, primarily for checkpoint loading.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl Tokenizer<'_> {
    /// Computes the complete tokenizer-side `NedoFormer` identity.
    ///
    /// The final model checkpoint contract must extend this digest with decoder-side
    /// FSM/allomorph/pronunciation identities; this method does not pretend to hash
    /// assets that are outside the tokenizer/segmenter package.
    ///
    /// Exact acceleration tables/caches are deliberately excluded because their
    /// validators guarantee semantic equivalence.  Everything that may change IDs,
    /// boundaries, normalization, grouping, code-mode placement, or whitespace
    /// reconstruction is included.
    ///
    /// # Errors
    ///
    /// Returns an error only if one of the stable vocabularies cannot serialize.
    pub fn nedoformer_contract_fingerprint(
        &self,
        input_characters: &CharacterVocabulary,
        output_vocabulary: &NedoFormerVocabulary,
    ) -> Result<NedoFormerContractFingerprint, TokenizerError> {
        let input_bytes = input_characters.to_bytes()?;
        let output_bytes = output_vocabulary.to_bytes()?;
        let input_sha: [u8; 32] = Sha256::digest(&input_bytes).into();
        let output_sha: [u8; 32] = Sha256::digest(&output_bytes).into();

        let mut hash = Sha256::new();
        hash.update(b"NEDOFORMER-TOKENIZER-CONTRACT\0");
        add_u32(&mut hash, NEDOFORMER_TOKENIZER_CONTRACT_VERSION);
        add_u32(&mut hash, TOKENIZER_SCHEMA_VERSION);
        add_u32(&mut hash, nedo_format::FORMAT_SCHEMA_VERSION);
        add_bytes(&mut hash, MORPHOLOGY_SHA256.as_bytes())?;
        add_bytes(&mut hash, MODEL_SHA256.as_bytes())?;
        add_bytes(&mut hash, &input_sha)?;
        add_bytes(&mut hash, &output_sha)?;

        hash.update([self.config.mode as u8]);
        add_u64(
            &mut hash,
            u64::try_from(self.config.max_sentence_tokens)
                .map_err(|_| TokenizerError::LengthOverflow("max sentence tokens fingerprint"))?,
        );
        add_u64(
            &mut hash,
            u64::try_from(self.config.max_fallback_chars)
                .map_err(|_| TokenizerError::LengthOverflow("max fallback chars fingerprint"))?,
        );
        hash.update([
            u8::from(self.config.contextual_disambiguation),
            u8::from(self.config.detect_unmarked_code),
        ]);
        add_u32(&mut hash, NEDOFORMER_SHADOW_NORMALIZATION_VERSION);
        add_u32(&mut hash, NEDOFORMER_CODE_SEGMENTATION_VERSION);
        add_u32(&mut hash, NEDOFORMER_NUMERIC_SEGMENTATION_VERSION);
        add_u32(&mut hash, NEDOFORMER_WHITESPACE_SCHEMA_VERSION);
        add_u32(&mut hash, NEDOFORMER_LATTICE_SCHEMA_VERSION);
        add_u32(&mut hash, NEDOFORMER_INPUT_ENCODING_VERSION);
        Ok(NedoFormerContractFingerprint(hash.finalize().into()))
    }

    /// Rejects a tokenizer-side checkpoint contract mismatch loudly.
    ///
    /// # Errors
    ///
    /// Returns an error when the active tokenizer-side fingerprint differs.
    pub fn verify_nedoformer_contract_fingerprint(
        &self,
        input_characters: &CharacterVocabulary,
        output_vocabulary: &NedoFormerVocabulary,
        expected: NedoFormerContractFingerprint,
    ) -> Result<(), TokenizerError> {
        if self.nedoformer_contract_fingerprint(input_characters, output_vocabulary)? != expected {
            return Err(TokenizerError::AssetIdentityMismatch);
        }
        Ok(())
    }
}

fn add_u32(hash: &mut Sha256, value: u32) {
    hash.update(value.to_le_bytes());
}

fn add_u64(hash: &mut Sha256, value: u64) {
    hash.update(value.to_le_bytes());
}

fn add_bytes(hash: &mut Sha256, bytes: &[u8]) -> Result<(), TokenizerError> {
    hash.update(
        u64::try_from(bytes.len())
            .map_err(|_| TokenizerError::LengthOverflow("contract fingerprint field"))?
            .to_le_bytes(),
    );
    hash.update(bytes);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::NedoFormerContractFingerprint;
    use crate::{CharacterVocabulary, NedoFormerVocabulary, Tokenizer, TokenizerConfig};

    #[test]
    #[allow(clippy::too_many_lines)] // Cross-feature contract test intentionally spans all public layers.
    fn nedoformer_tokenizer_contract_acceptance() -> Result<(), crate::TokenizerError> {
        use crate::{NedoFormerLatticeSidecar, NedoFormerSamplingPolicy, TokenStatus};

        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        assert_eq!(TokenizerConfig::default().max_fallback_chars, 48);

        let long_raw = "q".repeat(65).into_bytes();
        let long = tokenizer.nedoformer_lattice(long_raw.clone())?;
        let selected_long = long.selected_document()?;
        for unit in selected_long.units().iter().filter(|unit| {
            matches!(
                unit.status,
                TokenStatus::Unknown | TokenStatus::Code | TokenStatus::Opaque
            ) || unit.mode == crate::TokenMode::Opaque
        }) {
            let start = usize::try_from(unit.span.start)
                .map_err(|_| crate::TokenizerError::LengthOverflow("acceptance fallback start"))?;
            let end = usize::try_from(unit.span.end)
                .map_err(|_| crate::TokenizerError::LengthOverflow("acceptance fallback end"))?;
            let text = std::str::from_utf8(&long_raw[start..end])
                .map_err(|_| crate::TokenizerError::InvalidUtf8Unit)?;
            assert!(text.chars().count() <= 48);
        }

        let mixed_raw = "koyun cocuklarimizdan geliyor mu?  23.07.2026\r\n```python\nparseHttpRequest2XX(foo_bar)\n```"
            .as_bytes()
            .to_vec();
        let lattice = tokenizer.nedoformer_lattice(mixed_raw.clone())?;
        assert!(lattice.units().iter().any(|unit| unit.candidates.len() > 1));
        assert_eq!(lattice.selected_document()?.decode(), mixed_raw.as_slice());
        let rich_blob = lattice.to_bytes()?;
        let reloaded = crate::NedoFormerLatticeDocument::from_bytes(&rich_blob)?;
        let sampled = reloaded.sample(
            NedoFormerSamplingPolicy::ContextWeighted { temperature: 1.0 },
            2026,
        )?;
        assert_eq!(sampled.decode(), mixed_raw.as_slice());

        let sidecar_blob = lattice.to_sidecar_bytes()?;
        assert!(!sidecar_blob
            .windows(mixed_raw.len())
            .any(|window| window == mixed_raw));
        let sidecar = NedoFormerLatticeSidecar::from_bytes(mixed_raw.clone(), &sidecar_blob)?;
        let sampled_sidecar = sidecar.sample_lossless(
            NedoFormerSamplingPolicy::ContextWeighted { temperature: 1.0 },
            2026,
        )?;
        assert_eq!(sampled_sidecar.decode(), mixed_raw.as_slice());

        let selected = lattice.selected_document()?;
        let input = CharacterVocabulary::train(std::slice::from_ref(&selected), 500);
        let output =
            NedoFormerVocabulary::train(std::slice::from_ref(&selected), 16_000, 500, 4_096)?;
        let generation = output.encode_document(&selected)?;
        assert_eq!(output.decode(&generation.ids)?, mixed_raw);
        let rich_input = lattice.sample_input_encoding(
            &input,
            NedoFormerSamplingPolicy::ContextWeighted { temperature: 1.0 },
            2026,
        )?;
        let sidecar_input = sidecar.sample_input_encoding(
            &input,
            NedoFormerSamplingPolicy::ContextWeighted { temperature: 1.0 },
            2026,
        )?;
        assert_eq!(rich_input, sidecar_input);
        assert_eq!(rich_input.segment_offsets.first(), Some(&0));
        assert_eq!(
            rich_input.segment_offsets.last().copied(),
            u32::try_from(rich_input.ids.len()).ok()
        );
        assert!(!rich_input.pooled_segments.is_empty());
        let fingerprint = tokenizer.nedoformer_contract_fingerprint(&input, &output)?;
        tokenizer.verify_nedoformer_contract_fingerprint(&input, &output, fingerprint)?;
        assert_eq!(fingerprint.hex().len(), 64);
        Ok(())
    }

    #[test]
    fn contract_fingerprint_is_deterministic_and_config_sensitive(
    ) -> Result<(), crate::TokenizerError> {
        let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
        let document = tokenizer
            .nedoformer_lattice("cocuklarimizdan 2026'da".as_bytes().to_vec())?
            .selected_document()?;
        let input = CharacterVocabulary::train(std::slice::from_ref(&document), 500);
        let output =
            NedoFormerVocabulary::train(std::slice::from_ref(&document), 16_000, 500, 4_096)?;
        let first = tokenizer.nedoformer_contract_fingerprint(&input, &output)?;
        let second = tokenizer.nedoformer_contract_fingerprint(&input, &output)?;
        assert_eq!(first, second);
        tokenizer.verify_nedoformer_contract_fingerprint(&input, &output, first)?;

        let changed = Tokenizer::embedded(TokenizerConfig {
            max_fallback_chars: 32,
            ..TokenizerConfig::default()
        })?;
        let changed_fingerprint = changed.nedoformer_contract_fingerprint(&input, &output)?;
        assert_ne!(first, changed_fingerprint);
        assert!(changed
            .verify_nedoformer_contract_fingerprint(
                &input,
                &output,
                NedoFormerContractFingerprint::from_bytes(*first.as_bytes()),
            )
            .is_err());
        Ok(())
    }
}
