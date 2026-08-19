use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::ExitCode;

use nedo_morph_bundle::{
    ambiguity_word_data, NativeAnalysis, NativeDisambiguator, NativeMorphology,
};
use serde::Deserialize;

const MAX_REPORTED_MISMATCHES: usize = 200;

#[derive(Deserialize)]
struct OracleRecord {
    sentence: String,
    tokens: Vec<String>,
    best: Vec<String>,
    candidates: Vec<Vec<String>>,
    candidate_hashes: Vec<Vec<i32>>,
    candidate_lemmas: Vec<Vec<String>>,
    candidate_igs: Vec<Vec<Vec<String>>>,
}

#[derive(Default)]
struct Summary {
    sentences: usize,
    exact_sentences: usize,
    mismatched_sentences: usize,
    tokens: usize,
    exact_tokens: usize,
    candidate_tokens: usize,
    exact_candidate_order_tokens: usize,
    candidates: usize,
    exact_candidate_hashes: usize,
    exact_candidate_lemmas: usize,
    exact_candidate_igs: usize,
}

fn main() -> ExitCode {
    match run() {
        Ok(summary) => {
            println!(
                "NEDO_NATIVE_DISAMBIGUATION_PARITY sentences={} exact_sentences={} mismatched_sentences={} tokens={} exact_tokens={} candidate_tokens={} exact_candidate_order_tokens={} candidates={} exact_candidate_hashes={} exact_candidate_lemmas={} exact_candidate_igs={}",
                summary.sentences,
                summary.exact_sentences,
                summary.mismatched_sentences,
                summary.tokens,
                summary.exact_tokens,
                summary.candidate_tokens,
                summary.exact_candidate_order_tokens,
                summary.candidates,
                summary.exact_candidate_hashes,
                summary.exact_candidate_lemmas,
                summary.exact_candidate_igs,
            );
            if summary.mismatched_sentences == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(error) => {
            eprintln!("native disambiguation parity failed: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<Summary, Box<dyn std::error::Error>> {
    let mut arguments = env::args_os();
    let program = arguments
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "nedo-native-disambiguation-parity".to_owned());
    let binary = arguments.next().ok_or_else(|| usage(&program))?;
    let model = arguments.next().ok_or_else(|| usage(&program))?;
    let oracle = arguments.next().ok_or_else(|| usage(&program))?;
    if arguments.next().is_some() {
        return Err(usage(&program).into());
    }
    let native_bytes = fs::read(binary)?;
    let model_bytes = fs::read(model)?;
    let morphology = NativeMorphology::parse(&native_bytes)?;
    let disambiguator = NativeDisambiguator::from_bytes(&model_bytes)?;
    compare(&morphology, &disambiguator, Path::new(&oracle))
}

fn compare(
    morphology: &NativeMorphology<'_>,
    disambiguator: &NativeDisambiguator,
    oracle: &Path,
) -> Result<Summary, Box<dyn std::error::Error>> {
    let reader = BufReader::new(File::open(oracle)?);
    let mut summary = Summary::default();
    for (line_index, line) in reader.lines().enumerate() {
        let record: OracleRecord = serde_json::from_str(&line?)?;
        let token_refs: Vec<&str> = record.tokens.iter().map(String::as_str).collect();
        let candidates = token_refs
            .iter()
            .map(|token| morphology.analyze_token(token))
            .collect::<Result<Vec<_>, _>>()?;
        audit_candidates(line_index, &record, &candidates, &mut summary)?;
        let result = disambiguator.disambiguate(&token_refs, &candidates)?;
        let actual: Vec<&str> = result
            .best
            .iter()
            .map(|analysis| analysis.canonical.as_str())
            .collect();
        let expected: Vec<&str> = record.best.iter().map(String::as_str).collect();
        summary.sentences += 1;
        summary.tokens += expected.len();
        summary.exact_tokens += expected
            .iter()
            .zip(&actual)
            .filter(|(left, right)| left == right)
            .count();
        if expected == actual {
            summary.exact_sentences += 1;
        } else {
            summary.mismatched_sentences += 1;
            if summary.mismatched_sentences <= MAX_REPORTED_MISMATCHES {
                report_mismatch(line_index, &record, &actual, &candidates);
            }
        }
    }
    Ok(summary)
}

fn audit_candidates(
    line_index: usize,
    record: &OracleRecord,
    candidates: &[Vec<NativeAnalysis>],
    summary: &mut Summary,
) -> Result<(), Box<dyn std::error::Error>> {
    for (token_index, native_candidates) in candidates.iter().enumerate() {
        let expected_candidates = record
            .candidates
            .get(token_index)
            .ok_or("oracle candidate token index is missing")?;
        let native_keys: Vec<&str> = native_candidates
            .iter()
            .map(|candidate| candidate.canonical.as_str())
            .collect();
        let expected_keys: Vec<&str> = expected_candidates.iter().map(String::as_str).collect();
        summary.candidate_tokens += 1;
        if native_keys == expected_keys {
            summary.exact_candidate_order_tokens += 1;
        } else {
            eprintln!(
                "CANDIDATE_ORDER_MISMATCH line={} token_index={} token={:?} java={:?} native={:?}",
                line_index + 1,
                token_index,
                record.tokens[token_index],
                expected_keys,
                native_keys,
            );
        }
        for candidate in native_candidates {
            summary.candidates += 1;
            let Some(expected_index) = expected_candidates
                .iter()
                .position(|expected| expected == &candidate.canonical)
            else {
                continue;
            };
            let metadata = ambiguity_word_data(candidate);
            let hash_exact =
                record.candidate_hashes[token_index][expected_index] == metadata.java_hash;
            let lemma_exact =
                record.candidate_lemmas[token_index][expected_index] == metadata.lemma;
            let igs_exact = record.candidate_igs[token_index][expected_index] == metadata.igs;
            if hash_exact {
                summary.exact_candidate_hashes += 1;
            }
            if lemma_exact {
                summary.exact_candidate_lemmas += 1;
            }
            if igs_exact {
                summary.exact_candidate_igs += 1;
            }
            if !hash_exact || !lemma_exact || !igs_exact {
                eprintln!(
                    "CANDIDATE_METADATA_MISMATCH line={} token_index={} token={:?} candidate={:?} java_hash={} native_hash={} java_lemma={:?} native_lemma={:?} java_igs={:?} native_igs={:?}",
                    line_index + 1,
                    token_index,
                    record.tokens[token_index],
                    candidate.canonical,
                    record.candidate_hashes[token_index][expected_index],
                    metadata.java_hash,
                    record.candidate_lemmas[token_index][expected_index],
                    metadata.lemma,
                    record.candidate_igs[token_index][expected_index],
                    metadata.igs,
                );
            }
        }
    }
    Ok(())
}

fn report_mismatch(
    line_index: usize,
    record: &OracleRecord,
    actual: &[&str],
    candidates: &[Vec<NativeAnalysis>],
) {
    eprintln!(
        "DISAMBIGUATION_MISMATCH line={} sentence={:?}",
        line_index + 1,
        record.sentence,
    );
    for (token_index, (((token, expected_key), actual_key), token_candidates)) in record
        .tokens
        .iter()
        .zip(&record.best)
        .zip(actual)
        .zip(candidates)
        .enumerate()
    {
        if expected_key.as_str() == *actual_key {
            continue;
        }
        let candidate_keys: Vec<&str> = token_candidates
            .iter()
            .map(|candidate| candidate.canonical.as_str())
            .collect();
        let expected_index = candidate_keys
            .iter()
            .position(|candidate| *candidate == expected_key.as_str());
        let actual_index = candidate_keys
            .iter()
            .position(|candidate| *candidate == *actual_key);
        let candidate_metadata: Vec<_> = token_candidates.iter().map(ambiguity_word_data).collect();
        eprintln!(
            "TOKEN_MISMATCH line={} token_index={} token={:?} expected={:?} actual={:?} expected_candidate_index={:?} actual_candidate_index={:?} candidate_count={} candidates={:?} native_candidate_metadata={:?} java_candidate_hashes={:?} java_candidate_lemmas={:?} java_candidate_igs={:?}",
            line_index + 1,
            token_index,
            token,
            expected_key,
            actual_key,
            expected_index,
            actual_index,
            candidate_keys.len(),
            candidate_keys,
            candidate_metadata,
            record.candidate_hashes[token_index],
            record.candidate_lemmas[token_index],
            record.candidate_igs[token_index],
        );
    }
}

fn usage(program: &str) -> String {
    format!("usage: {program} <native-binary> <model-compressed> <oracle-jsonl>")
}
