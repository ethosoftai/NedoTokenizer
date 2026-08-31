use std::env;
use std::fs::File;
use std::io::{BufReader, Read};

use nedo_tokenizer::{SurfaceVocabulary, SurfaceVocabularyKind, Tokenizer, TokenizerConfig};

const INPUT_MAGIC: &[u8; 8] = b"MVOCBIN1";
const MAX_BATCH_RECORDS: usize = 2048;
const MAX_BATCH_BYTES: usize = 16 * 1024 * 1024;

struct Record {
    raw: Vec<u8>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() < 5 || (args.len() - 3) % 2 != 0 {
        return Err(format!(
            "usage: {} EVAL_MVOC THREADS LABEL VOCAB [LABEL VOCAB ...]",
            args.first().map(String::as_str).unwrap_or("nedo-surface-parity")
        )
        .into());
    }
    let eval_path = &args[1];
    let threads = args[2].parse::<usize>()?;
    if threads == 0 {
        return Err("threads must be positive".into());
    }
    let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
    for pair in args[3..].chunks_exact(2) {
        let vocabulary = SurfaceVocabulary::from_bytes(&std::fs::read(&pair[1])?)?;
        verify(eval_path, threads, &tokenizer, &pair[0], &vocabulary)?;
    }
    Ok(())
}

fn verify(
    path: &str,
    threads: usize,
    tokenizer: &Tokenizer<'_>,
    label: &str,
    vocabulary: &SurfaceVocabulary,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, File::open(path)?);
    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != INPUT_MAGIC {
        return Err("parity input magic mismatch".into());
    }
    let mut records = 0_u64;
    let mut bytes = 0_u64;
    let mut tokens = 0_u64;
    let mut batch = Vec::<Record>::with_capacity(MAX_BATCH_RECORDS);
    let mut batch_bytes = 0_usize;
    loop {
        let Some(record) = read_record(&mut reader)? else {
            break;
        };
        batch_bytes = batch_bytes.saturating_add(record.raw.len());
        batch.push(record);
        if batch.len() >= MAX_BATCH_RECORDS || batch_bytes >= MAX_BATCH_BYTES {
            consume(tokenizer, vocabulary, threads, &batch, &mut records, &mut bytes, &mut tokens)?;
            batch.clear();
            batch_bytes = 0;
        }
    }
    if !batch.is_empty() {
        consume(tokenizer, vocabulary, threads, &batch, &mut records, &mut bytes, &mut tokens)?;
    }
    println!(
        "PARITY\t{}\t{:?}\trecords={}\tbytes={}\ttokens={}\tmismatches=0\troundtrip_failures=0\tboundary_violations=0",
        label,
        vocabulary.kind(),
        records,
        bytes,
        tokens,
    );
    Ok(())
}

fn consume(
    tokenizer: &Tokenizer<'_>,
    vocabulary: &SurfaceVocabulary,
    threads: usize,
    batch: &[Record],
    records: &mut u64,
    bytes: &mut u64,
    tokens: &mut u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let inputs = batch.iter().map(|record| record.raw.clone()).collect::<Vec<_>>();
    let documents = tokenizer.tokenize_batch(&inputs, threads)?;
    let newlines = vec![false; inputs.len()];
    let flat = tokenizer.encode_surface_batch(&inputs, &newlines, vocabulary, threads, true)?;
    if documents.len() != batch.len() || flat.document_offsets.len() != batch.len() + 1 {
        return Err("parity batch cardinality mismatch".into());
    }
    for (index, (record, document)) in batch.iter().zip(&documents).enumerate() {
        let rich = vocabulary.encode_document(document, false)?;
        let start = usize::try_from(flat.document_offsets[index])?;
        let end = usize::try_from(flat.document_offsets[index + 1])?;
        let flat_ids = flat.ids.get(start..end).ok_or("flat ID range invalid")?;
        let flat_lengths = flat.lengths.get(start..end).ok_or("flat length range invalid")?;
        if flat_ids != rich.ids.as_slice() || flat_lengths != rich.lengths.as_slice() {
            return Err(format!("rich/flat surface parity mismatch at record {}", *records + index as u64).into());
        }
        if vocabulary.decode_ids(flat_ids)? != record.raw {
            return Err(format!("surface roundtrip mismatch at record {}", *records + index as u64).into());
        }
        let mut token_boundaries = Vec::with_capacity(flat_lengths.len());
        let mut byte_cursor = 0_u64;
        for &length in flat_lengths {
            if length == 0 {
                continue;
            }
            byte_cursor = byte_cursor.saturating_add(u64::from(length));
            token_boundaries.push(byte_cursor);
        }
        for unit in document.units() {
            match vocabulary.kind() {
                SurfaceVocabularyKind::ByteBpe | SurfaceVocabularyKind::GreedyLongest => {
                    for &cut in &unit.cuts {
                        if token_boundaries.binary_search(&cut).is_err() {
                            return Err(format!(
                                "morphology boundary violation at record {} byte {}",
                                *records + index as u64,
                                cut
                            )
                            .into());
                        }
                    }
                }
                SurfaceVocabularyKind::RootSuffixByteBpe => {
                    if let Some(&root_end) = unit.cuts.first() {
                        if token_boundaries.binary_search(&root_end).is_err() {
                            return Err(format!(
                                "root boundary violation at record {} byte {}",
                                *records + index as u64,
                                root_end
                            )
                            .into());
                        }
                    }
                }
                SurfaceVocabularyKind::LexicalByteBpe => {}
            }
        }
        *bytes = bytes.saturating_add(u64::try_from(record.raw.len())?);
        *tokens = tokens.saturating_add(
            u64::try_from(flat_lengths.iter().filter(|&&length| length > 0).count())?,
        );
    }
    *records = records.saturating_add(u64::try_from(batch.len())?);
    Ok(())
}

fn read_record(reader: &mut BufReader<File>) -> Result<Option<Record>, Box<dyn std::error::Error>> {
    let mut source = [0_u8; 1];
    match reader.read_exact(&mut source) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(error) => return Err(error.into()),
    }
    let mut key = [0_u8; 8];
    let mut length = [0_u8; 4];
    reader.read_exact(&mut key)?;
    reader.read_exact(&mut length)?;
    let length = usize::try_from(u32::from_le_bytes(length))?;
    let mut raw = vec![0_u8; length];
    reader.read_exact(&mut raw)?;
    Ok(Some(Record { raw }))
}
