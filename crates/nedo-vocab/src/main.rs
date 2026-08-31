use std::env;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use nedo_tokenizer::{
    surface_bpe_root_suffix_segments, surface_bpe_segments, Tokenizer, TokenizerConfig,
};

const INPUT_MAGIC: &[u8; 8] = b"MVOCBIN1";
const SEGMENT_MAGIC: &[u8; 8] = b"NSEG0001";
const SENTINEL: u8 = 0x1f;
const MAX_BATCH_RECORDS: usize = 8192;
const MAX_BATCH_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug)]
struct Record {
    source_id: u8,
    doc_key: u64,
    raw: Vec<u8>,
}

#[derive(Default)]
struct Stats {
    input_records: u64,
    input_bytes: u64,
    train_records: u64,
    train_bytes: u64,
    eval_records: u64,
    eval_bytes: u64,
    skipped_invalid_utf8: u64,
    skipped_sentinel: u64,
    segment_count: u64,
    segmented_bytes: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 8 {
        return Err(format!(
            "usage: {} INPUT_MVOC TRAIN_NSEG EVAL_MVOC EVAL_MODULUS SPLIT_SEED THREADS BOUNDARY_POLICY",
            args.first().map(String::as_str).unwrap_or("nedo-surface-segment")
        )
        .into());
    }
    let input_path = &args[1];
    let train_path = &args[2];
    let eval_path = &args[3];
    let eval_modulus = args[4].parse::<u64>()?;
    let split_seed = parse_u64(&args[5])?;
    let threads = args[6].parse::<usize>()?;
    let boundary_policy = args[7].as_str();
    if !matches!(boundary_policy, "morphology" | "root-suffix" | "lexical") {
        return Err("boundary policy must be morphology, root-suffix, or lexical".into());
    }
    if eval_modulus < 2 {
        return Err("eval modulus must be at least 2".into());
    }
    if threads == 0 {
        return Err("threads must be positive".into());
    }

    let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
    let mut input = BufReader::with_capacity(8 * 1024 * 1024, File::open(input_path)?);
    require_magic(&mut input, INPUT_MAGIC)?;
    let mut train = BufWriter::with_capacity(8 * 1024 * 1024, create_new(train_path)?);
    let mut eval = BufWriter::with_capacity(8 * 1024 * 1024, create_new(eval_path)?);
    train.write_all(SEGMENT_MAGIC)?;
    eval.write_all(INPUT_MAGIC)?;

    let mut stats = Stats::default();
    let mut batch = Vec::<Record>::with_capacity(MAX_BATCH_RECORDS);
    let mut batch_bytes = 0_usize;

    loop {
        let Some(record) = read_record(&mut input)? else {
            break;
        };
        stats.input_records = stats.input_records.saturating_add(1);
        stats.input_bytes = stats
            .input_bytes
            .saturating_add(u64::try_from(record.raw.len())?);
        if splitmix64(record.doc_key ^ split_seed) % eval_modulus == 0 {
            write_mvoc_record(&mut eval, &record)?;
            stats.eval_records = stats.eval_records.saturating_add(1);
            stats.eval_bytes = stats
                .eval_bytes
                .saturating_add(u64::try_from(record.raw.len())?);
            continue;
        }
        if std::str::from_utf8(&record.raw).is_err() {
            stats.skipped_invalid_utf8 = stats.skipped_invalid_utf8.saturating_add(1);
            continue;
        }
        if record.raw.contains(&SENTINEL) {
            stats.skipped_sentinel = stats.skipped_sentinel.saturating_add(1);
            continue;
        }
        batch_bytes = batch_bytes.saturating_add(record.raw.len());
        batch.push(record);
        if batch.len() >= MAX_BATCH_RECORDS || batch_bytes >= MAX_BATCH_BYTES {
            flush_train_batch(
                &tokenizer,
                &mut train,
                &mut batch,
                threads,
                boundary_policy,
                &mut stats,
            )?;
            batch_bytes = 0;
        }
    }
    if !batch.is_empty() {
        flush_train_batch(
            &tokenizer,
            &mut train,
            &mut batch,
            threads,
            boundary_policy,
            &mut stats,
        )?;
    }
    train.flush()?;
    eval.flush()?;

    println!("status=PASS");
    println!("input_records={}", stats.input_records);
    println!("input_bytes={}", stats.input_bytes);
    println!("train_records={}", stats.train_records);
    println!("train_bytes={}", stats.train_bytes);
    println!("eval_records={}", stats.eval_records);
    println!("eval_bytes={}", stats.eval_bytes);
    println!("skipped_invalid_utf8={}", stats.skipped_invalid_utf8);
    println!("skipped_sentinel={}", stats.skipped_sentinel);
    println!("segment_count={}", stats.segment_count);
    println!("segmented_bytes={}", stats.segmented_bytes);
    println!("sentinel_hex=1f");
    println!("eval_modulus={eval_modulus}");
    println!("split_seed={split_seed}");
    println!("boundary_policy={boundary_policy}");
    Ok(())
}

fn create_new(path: &str) -> Result<File, std::io::Error> {
    if Path::new(path).exists() {
        std::fs::remove_file(path)?;
    }
    File::create(path)
}

fn flush_train_batch(
    tokenizer: &Tokenizer<'_>,
    writer: &mut BufWriter<File>,
    batch: &mut Vec<Record>,
    threads: usize,
    boundary_policy: &str,
    stats: &mut Stats,
) -> Result<(), Box<dyn std::error::Error>> {
    let inputs = batch.iter().map(|record| record.raw.clone()).collect::<Vec<_>>();
    let documents = tokenizer.tokenize_batch(&inputs, threads)?;
    if documents.len() != batch.len() {
        return Err("tokenizer batch cardinality mismatch".into());
    }
    for (record, document) in batch.iter().zip(&documents) {
        let spans = match boundary_policy {
            "morphology" => surface_bpe_segments(document, true)?,
            "root-suffix" => surface_bpe_root_suffix_segments(document)?,
            "lexical" => surface_bpe_segments(document, false)?,
            _ => return Err("invalid boundary policy after validation".into()),
        };
        let mut segmented = Vec::with_capacity(
            record
                .raw
                .len()
                .saturating_add(spans.len().saturating_sub(1)),
        );
        for (index, span) in spans.iter().enumerate() {
            let start = usize::try_from(span.start)?;
            let end = usize::try_from(span.end)?;
            let bytes = record
                .raw
                .get(start..end)
                .ok_or("surface segment outside input")?;
            segmented.extend_from_slice(bytes);
            if index + 1 < spans.len() {
                segmented.push(SENTINEL);
            }
        }
        let reconstructed = segmented
            .iter()
            .copied()
            .filter(|byte| *byte != SENTINEL)
            .collect::<Vec<_>>();
        if reconstructed != record.raw {
            return Err("segmented document does not reconstruct exact source bytes".into());
        }
        let length = u32::try_from(segmented.len())?;
        writer.write_all(&length.to_le_bytes())?;
        writer.write_all(&segmented)?;
        stats.train_records = stats.train_records.saturating_add(1);
        stats.train_bytes = stats
            .train_bytes
            .saturating_add(u64::try_from(record.raw.len())?);
        stats.segment_count = stats
            .segment_count
            .saturating_add(u64::try_from(spans.len())?);
        stats.segmented_bytes = stats
            .segmented_bytes
            .saturating_add(u64::try_from(segmented.len())?);
    }
    batch.clear();
    Ok(())
}

fn require_magic(reader: &mut BufReader<File>, expected: &[u8; 8]) -> Result<(), Box<dyn std::error::Error>> {
    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != expected {
        return Err("input sample magic mismatch".into());
    }
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
    Ok(Some(Record {
        source_id: source[0],
        doc_key: u64::from_le_bytes(key),
        raw,
    }))
}

fn write_mvoc_record(writer: &mut BufWriter<File>, record: &Record) -> Result<(), Box<dyn std::error::Error>> {
    writer.write_all(&[record.source_id])?;
    writer.write_all(&record.doc_key.to_le_bytes())?;
    writer.write_all(&u32::try_from(record.raw.len())?.to_le_bytes())?;
    writer.write_all(&record.raw)?;
    Ok(())
}

fn parse_u64(value: &str) -> Result<u64, Box<dyn std::error::Error>> {
    if let Some(hex) = value.strip_prefix("0x") {
        Ok(u64::from_str_radix(hex, 16)?)
    } else {
        Ok(value.parse::<u64>()?)
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
