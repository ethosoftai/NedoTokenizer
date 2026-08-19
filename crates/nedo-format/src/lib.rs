//! Versioned, lossless byte-span document format.
//!
//! Raw bytes are the only source of truth. Analyses and boundaries are metadata
//! over half-open byte ranges and never rewrite the input.

#![forbid(unsafe_code)]

use core::fmt;

/// Current in-memory/on-disk schema generation.
pub const FORMAT_SCHEMA_VERSION: u32 = 1;

const FORMAT_MAGIC: &[u8; 8] = b"NEDOFMT\0";
const HEADER_LEN: usize = 8 + 4 + 8 + 8;

/// A half-open byte interval `[start, end)` into a document.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ByteSpan {
    /// Inclusive byte offset.
    pub start: u64,
    /// Exclusive byte offset.
    pub end: u64,
}

impl ByteSpan {
    /// Creates a span after validating ordering.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ReversedSpan`] when `start > end`.
    pub const fn new(start: u64, end: u64) -> Result<Self, FormatError> {
        if start > end {
            return Err(FormatError::ReversedSpan { start, end });
        }
        Ok(Self { start, end })
    }

    /// Returns the byte length of this span.
    #[must_use]
    pub const fn len(self) -> u64 {
        self.end - self.start
    }

    /// Returns true for an empty span.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }
}

/// A surface unit and optional cuts, all expressed in original byte offsets.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SurfaceUnit {
    /// Full unit span.
    pub span: ByteSpan,
    /// Strictly increasing cuts inside `span`.
    pub cuts: Vec<u64>,
}

impl SurfaceUnit {
    /// Constructs and validates a unit.
    ///
    /// # Errors
    ///
    /// Returns an error when a cut is outside the span or cuts are not strictly increasing.
    pub fn new(span: ByteSpan, cuts: Vec<u64>) -> Result<Self, FormatError> {
        let unit = Self { span, cuts };
        unit.validate()?;
        Ok(unit)
    }

    /// Validates cut ordering and containment.
    ///
    /// # Errors
    ///
    /// Returns an error when a cut is outside the span or cuts are not strictly increasing.
    pub fn validate(&self) -> Result<(), FormatError> {
        if self.span.is_empty() {
            return Err(FormatError::EmptyUnit {
                offset: self.span.start,
            });
        }
        let mut previous = self.span.start;
        for &cut in &self.cuts {
            if cut <= self.span.start || cut >= self.span.end {
                return Err(FormatError::CutOutsideSpan {
                    cut,
                    start: self.span.start,
                    end: self.span.end,
                });
            }
            if cut <= previous {
                return Err(FormatError::CutsNotStrictlyIncreasing {
                    previous,
                    current: cut,
                });
            }
            previous = cut;
        }
        Ok(())
    }

    /// Iterates over the contiguous morpheme surface spans implied by cuts.
    pub fn pieces(&self) -> impl Iterator<Item = ByteSpan> + '_ {
        let starts = core::iter::once(self.span.start).chain(self.cuts.iter().copied());
        let ends = self
            .cuts
            .iter()
            .copied()
            .chain(core::iter::once(self.span.end));
        starts.zip(ends).map(|(start, end)| ByteSpan { start, end })
    }
}

/// A document carrying immutable raw bytes and metadata spans.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LosslessDocument {
    raw: Vec<u8>,
    units: Vec<SurfaceUnit>,
}

impl LosslessDocument {
    /// Creates a document and validates every span against raw byte length.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid units, overlapping units, or spans past the document.
    pub fn new(raw: Vec<u8>, units: Vec<SurfaceUnit>) -> Result<Self, FormatError> {
        let document = Self { raw, units };
        document.validate()?;
        Ok(document)
    }

    /// Creates a document with no analysis metadata.
    #[must_use]
    pub const fn unanalyzed(raw: Vec<u8>) -> Self {
        Self {
            raw,
            units: Vec::new(),
        }
    }

    /// Returns exact original bytes.
    #[must_use]
    pub fn decode(&self) -> &[u8] {
        &self.raw
    }

    /// Returns surface units.
    #[must_use]
    pub fn units(&self) -> &[SurfaceUnit] {
        &self.units
    }

    /// Consumes the document and returns its exact raw bytes and surface units.
    #[must_use]
    pub fn into_parts(self) -> (Vec<u8>, Vec<SurfaceUnit>) {
        (self.raw, self.units)
    }

    /// Returns a checked slice for a span.
    ///
    /// # Errors
    ///
    /// Returns an error when offsets cannot be represented or the span exceeds the document.
    pub fn slice(&self, span: ByteSpan) -> Result<&[u8], FormatError> {
        let start = usize::try_from(span.start)
            .map_err(|_| FormatError::OffsetTooLarge { offset: span.start })?;
        let end = usize::try_from(span.end)
            .map_err(|_| FormatError::OffsetTooLarge { offset: span.end })?;
        self.raw
            .get(start..end)
            .ok_or(FormatError::SpanPastDocument {
                start: span.start,
                end: span.end,
                document_len: self.raw.len() as u64,
            })
    }

    /// Validates all format invariants.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid cuts, overlapping units, or spans past the document.
    pub fn validate(&self) -> Result<(), FormatError> {
        let document_len = self.raw.len() as u64;
        let mut previous_end = 0_u64;
        for (index, unit) in self.units.iter().enumerate() {
            unit.validate()?;
            if unit.span.end > document_len {
                return Err(FormatError::SpanPastDocument {
                    start: unit.span.start,
                    end: unit.span.end,
                    document_len,
                });
            }
            if index > 0 && unit.span.start < previous_end {
                return Err(FormatError::OverlappingUnits {
                    previous_end,
                    current_start: unit.span.start,
                });
            }
            previous_end = unit.span.end;
        }
        Ok(())
    }
}

/// Serializes a document to the stable little-endian format.
///
/// The encoding is deterministic for an identical [`LosslessDocument`].
///
/// # Errors
///
/// Returns an error when document metadata violates format invariants or a
/// collection length cannot be represented by the format.
pub fn encode_binary(document: &LosslessDocument) -> Result<Vec<u8>, FormatError> {
    document.validate()?;
    let unit_count =
        u64::try_from(document.units.len()).map_err(|_| FormatError::LengthOverflow {
            field: "unit_count",
        })?;
    let mut capacity = HEADER_LEN
        .checked_add(document.raw.len())
        .ok_or(FormatError::LengthOverflow { field: "capacity" })?;
    for unit in &document.units {
        capacity = capacity
            .checked_add(8 + 8 + 4)
            .and_then(|value| value.checked_add(unit.cuts.len().saturating_mul(8)))
            .ok_or(FormatError::LengthOverflow { field: "capacity" })?;
    }
    let mut output = Vec::with_capacity(capacity);
    output.extend_from_slice(FORMAT_MAGIC);
    output.extend_from_slice(&FORMAT_SCHEMA_VERSION.to_le_bytes());
    output.extend_from_slice(&(document.raw.len() as u64).to_le_bytes());
    output.extend_from_slice(&unit_count.to_le_bytes());
    output.extend_from_slice(&document.raw);
    for unit in &document.units {
        output.extend_from_slice(&unit.span.start.to_le_bytes());
        output.extend_from_slice(&unit.span.end.to_le_bytes());
        let cut_count = u32::try_from(unit.cuts.len())
            .map_err(|_| FormatError::LengthOverflow { field: "cut_count" })?;
        output.extend_from_slice(&cut_count.to_le_bytes());
        for cut in &unit.cuts {
            output.extend_from_slice(&cut.to_le_bytes());
        }
    }
    Ok(output)
}

/// Decodes and fully validates the stable little-endian document format.
///
/// # Errors
///
/// Returns an error for bad magic, unsupported versions, truncated or trailing
/// data, unsafe lengths, or invalid document metadata.
pub fn decode_binary(input: &[u8]) -> Result<LosslessDocument, FormatError> {
    let mut reader = BinaryReader::new(input);
    let magic = reader.take(8)?;
    if magic != FORMAT_MAGIC {
        return Err(FormatError::BadMagic);
    }
    let version = reader.read_u32()?;
    if version != FORMAT_SCHEMA_VERSION {
        return Err(FormatError::UnsupportedVersion { version });
    }
    let raw_len = reader.read_u64()?;
    let unit_count = reader.read_u64()?;
    let raw_len_usize =
        usize::try_from(raw_len).map_err(|_| FormatError::LengthOverflow { field: "raw_len" })?;
    let raw = reader.take(raw_len_usize)?.to_vec();

    // Every unit requires at least 20 bytes. This bound prevents an attacker
    // from forcing a large allocation using only a forged count.
    let max_units_from_bytes = reader.remaining() / 20;
    let unit_count_usize =
        usize::try_from(unit_count).map_err(|_| FormatError::LengthOverflow {
            field: "unit_count",
        })?;
    if unit_count_usize > max_units_from_bytes {
        return Err(FormatError::ImpossibleCount {
            field: "unit_count",
            count: unit_count,
            remaining: reader.remaining(),
        });
    }
    let mut units = Vec::with_capacity(unit_count_usize);
    for _ in 0..unit_count_usize {
        let span = ByteSpan::new(reader.read_u64()?, reader.read_u64()?)?;
        let cut_count = reader.read_u32()?;
        let cut_count_usize = usize::try_from(cut_count)
            .map_err(|_| FormatError::LengthOverflow { field: "cut_count" })?;
        if cut_count_usize > reader.remaining() / 8 {
            return Err(FormatError::ImpossibleCount {
                field: "cut_count",
                count: u64::from(cut_count),
                remaining: reader.remaining(),
            });
        }
        let mut cuts = Vec::with_capacity(cut_count_usize);
        for _ in 0..cut_count_usize {
            cuts.push(reader.read_u64()?);
        }
        units.push(SurfaceUnit::new(span, cuts)?);
    }
    if reader.remaining() != 0 {
        return Err(FormatError::TrailingBytes {
            count: reader.remaining(),
        });
    }
    LosslessDocument::new(raw, units)
}

struct BinaryReader<'a> {
    input: &'a [u8],
    position: usize,
}

impl<'a> BinaryReader<'a> {
    const fn new(input: &'a [u8]) -> Self {
        Self { input, position: 0 }
    }

    const fn remaining(&self) -> usize {
        self.input.len() - self.position
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], FormatError> {
        let end = self
            .position
            .checked_add(count)
            .ok_or(FormatError::LengthOverflow { field: "position" })?;
        let result = self
            .input
            .get(self.position..end)
            .ok_or_else(|| FormatError::Truncated {
                position: self.position,
                needed: count,
                remaining: self.remaining(),
            })?;
        self.position = end;
        Ok(result)
    }

    fn read_u32(&mut self) -> Result<u32, FormatError> {
        let bytes: [u8; 4] = self
            .take(4)?
            .try_into()
            .map_err(|_| FormatError::Truncated {
                position: self.position,
                needed: 4,
                remaining: self.remaining(),
            })?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> Result<u64, FormatError> {
        let bytes: [u8; 8] = self
            .take(8)?
            .try_into()
            .map_err(|_| FormatError::Truncated {
                position: self.position,
                needed: 8,
                remaining: self.remaining(),
            })?;
        Ok(u64::from_le_bytes(bytes))
    }
}

/// Format validation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FormatError {
    /// A surface unit has zero length.
    EmptyUnit { offset: u64 },
    /// Binary data does not start with the required magic bytes.
    BadMagic,
    /// Binary data uses a schema this implementation cannot read.
    UnsupportedVersion { version: u32 },
    /// Binary data ended before a required field was complete.
    Truncated {
        position: usize,
        needed: usize,
        remaining: usize,
    },
    /// A serialized or in-memory length cannot be represented safely.
    LengthOverflow { field: &'static str },
    /// A count cannot fit in the bytes left in the input.
    ImpossibleCount {
        field: &'static str,
        count: u64,
        remaining: usize,
    },
    /// Valid data was followed by unrecognized bytes.
    TrailingBytes { count: usize },
    /// A span begins after it ends.
    ReversedSpan { start: u64, end: u64 },
    /// A cut does not lie strictly within its unit span.
    CutOutsideSpan { cut: u64, start: u64, end: u64 },
    /// Cuts must be strictly increasing.
    CutsNotStrictlyIncreasing { previous: u64, current: u64 },
    /// A span exceeds the raw document length.
    SpanPastDocument {
        start: u64,
        end: u64,
        document_len: u64,
    },
    /// Surface units overlap.
    OverlappingUnits {
        previous_end: u64,
        current_start: u64,
    },
    /// Offset cannot be represented on this platform.
    OffsetTooLarge { offset: u64 },
}

impl fmt::Display for FormatError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for FormatError {}

#[cfg(test)]
mod tests {
    use super::{
        decode_binary, encode_binary, ByteSpan, FormatError, LosslessDocument, SurfaceUnit,
    };

    #[test]
    fn arbitrary_bytes_round_trip_exactly() {
        let raw = vec![0, 0xff, b'\r', b'\n', 0xc4, 0xb0, 0];
        let document = LosslessDocument::unanalyzed(raw.clone());
        assert_eq!(document.decode(), raw);
    }

    #[test]
    fn turkish_utf8_offsets_are_byte_offsets() -> Result<(), FormatError> {
        let raw = "İstanbul'da".as_bytes().to_vec();
        let apostrophe = raw
            .iter()
            .position(|byte| *byte == b'\'')
            .ok_or(FormatError::OffsetTooLarge { offset: 0 })?;
        let unit = SurfaceUnit::new(
            ByteSpan::new(0, raw.len() as u64)?,
            vec![apostrophe as u64, (apostrophe + 1) as u64],
        )?;
        let document = LosslessDocument::new(raw.clone(), vec![unit])?;
        assert_eq!(document.decode(), raw);
        let pieces: Vec<&[u8]> = document.units()[0]
            .pieces()
            .map(|span| document.slice(span))
            .collect::<Result<_, _>>()?;
        assert_eq!(pieces, ["İstanbul".as_bytes(), b"'", b"da"]);
        Ok(())
    }

    #[test]
    fn rejects_duplicate_cut() -> Result<(), FormatError> {
        let error = SurfaceUnit::new(ByteSpan::new(0, 4)?, vec![2, 2]);
        assert!(matches!(
            error,
            Err(FormatError::CutsNotStrictlyIncreasing { .. })
        ));
        Ok(())
    }

    #[test]
    fn rejects_overlapping_units() -> Result<(), FormatError> {
        let first = SurfaceUnit::new(ByteSpan::new(0, 3)?, Vec::new())?;
        let second = SurfaceUnit::new(ByteSpan::new(2, 4)?, Vec::new())?;
        let error = LosslessDocument::new(b"abcd".to_vec(), vec![first, second]);
        assert!(matches!(error, Err(FormatError::OverlappingUnits { .. })));
        Ok(())
    }

    #[test]
    fn binary_encoding_is_deterministic_and_reversible() -> Result<(), FormatError> {
        let raw = b"evler\x00\xff\r\n".to_vec();
        let unit = SurfaceUnit::new(ByteSpan::new(0, 5)?, vec![2])?;
        let document = LosslessDocument::new(raw, vec![unit])?;
        let first = encode_binary(&document)?;
        let second = encode_binary(&document)?;
        assert_eq!(first, second);
        assert_eq!(decode_binary(&first)?, document);
        Ok(())
    }

    #[test]
    fn binary_decoder_rejects_corruption() -> Result<(), FormatError> {
        let document = LosslessDocument::new(
            b"evler".to_vec(),
            vec![SurfaceUnit::new(ByteSpan::new(0, 5)?, vec![2])?],
        )?;
        let encoded = encode_binary(&document)?;

        let mut bad_magic = encoded.clone();
        bad_magic[0] ^= 0xff;
        assert_eq!(decode_binary(&bad_magic), Err(FormatError::BadMagic));

        let mut unsupported = encoded.clone();
        unsupported[8..12].copy_from_slice(&99_u32.to_le_bytes());
        assert_eq!(
            decode_binary(&unsupported),
            Err(FormatError::UnsupportedVersion { version: 99 })
        );

        assert!(matches!(
            decode_binary(&encoded[..encoded.len() - 1]),
            Err(FormatError::Truncated { .. } | FormatError::ImpossibleCount { .. })
        ));

        let mut trailing = encoded;
        trailing.push(0);
        assert_eq!(
            decode_binary(&trailing),
            Err(FormatError::TrailingBytes { count: 1 })
        );
        Ok(())
    }

    #[test]
    fn rejects_empty_units() -> Result<(), FormatError> {
        let error = SurfaceUnit::new(ByteSpan::new(2, 2)?, Vec::new());
        assert_eq!(error, Err(FormatError::EmptyUnit { offset: 2 }));
        Ok(())
    }
}
