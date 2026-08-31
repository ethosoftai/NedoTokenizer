use std::env;
use std::fs::File;
use std::io::{BufReader, Read};

use nedo_tokenizer::{
    SurfaceVocabulary, Tokenizer, TokenizerConfig, SURFACE_BYTE_BASE_ID, SURFACE_ENTRY_BASE_ID,
};

const INPUT_MAGIC: &[u8; 8] = b"MVOCBIN1";
const MAX_BATCH_RECORDS: usize = 8192;
const MAX_BATCH_BYTES: usize = 32 * 1024 * 1024;

#[derive(Clone)]
struct Record {
    source_id: u8,
    raw: Vec<u8>,
}

#[derive(Clone, Copy, Default)]
struct Counts {
    records: u64,
    bytes: u64,
    tokens: u64,
    fallback_tokens: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() < 5 || (args.len() - 3) % 2 != 0 {
        return Err(format!(
            "usage: {} EVAL_MVOC THREADS LABEL VOCAB [LABEL VOCAB ...]",
            args.first().map(String::as_str).unwrap_or("nedo-surface-benchmark")
        )
        .into());
    }
    let eval_path = &args[1];
    let threads = args[2].parse::<usize>()?;
    if threads == 0 {
        return Err("threads must be positive".into());
    }
    let tokenizer = Tokenizer::embedded(TokenizerConfig::default())?;
    let pairs = args[3..]
        .chunks_exact(2)
        .map(|pair| Ok((pair[0].clone(), SurfaceVocabulary::from_bytes(&std::fs::read(&pair[1])?)?)))
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;

    for (label, vocabulary) in pairs {
        benchmark(eval_path, threads, &tokenizer, label.as_str(), &vocabulary)?;
    }
    Ok(())
}

fn benchmark(
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
        return Err("benchmark input magic mismatch".into());
    }
    let mut source = [Counts::default(); 256];
    let mut total = Counts::default();
    let mut batch = Vec::<Record>::with_capacity(MAX_BATCH_RECORDS);
    let mut batch_bytes = 0_usize;
    loop {
        let Some(record) = read_record(&mut reader)? else {
            break;
        };
        batch_bytes = batch_bytes.saturating_add(record.raw.len());
        batch.push(record);
        if batch.len() >= MAX_BATCH_RECORDS || batch_bytes >= MAX_BATCH_BYTES {
            consume_batch(tokenizer, vocabulary, threads, &batch, &mut source, &mut total)?;
            batch.clear();
            batch_bytes = 0;
        }
    }
    if !batch.is_empty() {
        consume_batch(tokenizer, vocabulary, threads, &batch, &mut source, &mut total)?;
    }
    let bytes_per_token = total.bytes as f64 / total.tokens.max(1) as f64;
    let fallback_pct = 100.0 * total.fallback_tokens as f64 / total.tokens.max(1) as f64;
    println!(
        "OVERALL\t{}\t{:?}\trecords={}\tbytes={}\ttokens={}\tbytes_per_token={:.6}\tfallback_tokens={}\tfallback_pct={:.6}",
        label,
        vocabulary.kind(),
        total.records,
        total.bytes,
        total.tokens,
        bytes_per_token,
        total.fallback_tokens,
        fallback_pct
    );
    for (source_id, counts) in source.iter().enumerate() {
        if counts.records == 0 {
            continue;
        }
        println!(
            "SOURCE\t{}\t{}\trecords={}\tbytes={}\ttokens={}\tbytes_per_token={:.6}\tfallback_tokens={}\tfallback_pct={:.6}",
            label,
            source_id,
            counts.records,
            counts.bytes,
            counts.tokens,
            counts.bytes as f64 / counts.tokens.max(1) as f64,
            counts.fallback_tokens,
            100.0 * counts.fallback_tokens as f64 / counts.tokens.max(1) as f64,
        );
    }
    Ok(())
}

fn consume_batch(
    tokenizer: &Tokenizer<'_>,
    vocabulary: &SurfaceVocabulary,
    threads: usize,
    records: &[Record],
    source: &mut [Counts; 256],
    total: &mut Counts,
) -> Result<(), Box<dyn std::error::Error>> {
    let inputs = records.iter().map(|record| record.raw.clone()).collect::<Vec<_>>();
    let newlines = vec![false; inputs.len()];
    let encoded = tokenizer.encode_surface_batch(&inputs, &newlines, vocabulary, threads, true)?;
    if encoded.document_offsets.len() != records.len().saturating_add(1) {
        return Err("benchmark document offset cardinality mismatch".into());
    }
    for (index, record) in records.iter().enumerate() {
        let start = usize::try_from(encoded.document_offsets[index])?;
        let end = usize::try_from(encoded.document_offsets[index + 1])?;
        let ids = encoded.ids.get(start..end).ok_or("benchmark ID range invalid")?;
        let lengths = encoded.lengths.get(start..end).ok_or("benchmark length range invalid")?;
        let mut tokens = 0_u64;
        let mut fallback = 0_u64;
        let mut bytes = 0_u64;
        for (&id, &length) in ids.iter().zip(lengths) {
            if length == 0 {
                continue;
            }
            tokens = tokens.saturating_add(1);
            bytes = bytes.saturating_add(u64::from(length));
            let id = u32::from(id);
            if (SURFACE_BYTE_BASE_ID..SURFACE_ENTRY_BASE_ID).contains(&id) {
                fallback = fallback.saturating_add(1);
            }
        }
        if bytes != u64::try_from(record.raw.len())? {
            return Err("benchmark byte accounting mismatch".into());
        }
        let counts = Counts {
            records: 1,
            bytes,
            tokens,
            fallback_tokens: fallback,
        };
        add(&mut source[usize::from(record.source_id)], counts);
        add(total, counts);
    }
    Ok(())
}

fn add(target: &mut Counts, value: Counts) {
    target.records = target.records.saturating_add(value.records);
    target.bytes = target.bytes.saturating_add(value.bytes);
    target.tokens = target.tokens.saturating_add(value.tokens);
    target.fallback_tokens = target.fallback_tokens.saturating_add(value.fallback_tokens);
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
        raw,
    }))
}
