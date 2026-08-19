use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::ExitCode;

use nedo_morph_bundle::{NativeGenerator, NativeMorphology};
use serde::Deserialize;

const MAX_REPORTED_MISMATCHES: usize = 20;

#[derive(Deserialize)]
struct GenerationRecord {
    dictionary_id: String,
    morphemes: Vec<String>,
    results: Vec<GenerationResult>,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd)]
struct GenerationResult {
    surface: String,
    canonical: String,
}

#[derive(Default)]
struct ParitySummary {
    requests: usize,
    exact_requests: usize,
    mismatched_requests: usize,
    expected_results: usize,
    native_results: usize,
}

fn main() -> ExitCode {
    match run() {
        Ok(summary) => {
            println!(
                "NEDO_NATIVE_GENERATION_PARITY requests={} exact_requests={} mismatched_requests={} expected_results={} native_results={}",
                summary.requests,
                summary.exact_requests,
                summary.mismatched_requests,
                summary.expected_results,
                summary.native_results,
            );
            if summary.mismatched_requests == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(error) => {
            eprintln!("native generation parity failed: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<ParitySummary, Box<dyn std::error::Error>> {
    let mut arguments = env::args_os();
    let program = arguments
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "nedo-native-generation-parity".to_owned());
    let binary = arguments.next().ok_or_else(|| usage(&program))?;
    let fixture = arguments.next().ok_or_else(|| usage(&program))?;
    if arguments.next().is_some() {
        return Err(usage(&program).into());
    }
    let bytes = fs::read(binary)?;
    let morphology = NativeMorphology::parse(&bytes)?;
    let generator = morphology.generator()?;
    compare_fixture(&generator, Path::new(&fixture))
}

fn compare_fixture(
    generator: &NativeGenerator<'_, '_>,
    fixture: &Path,
) -> Result<ParitySummary, Box<dyn std::error::Error>> {
    let reader = BufReader::new(File::open(fixture)?);
    let mut summary = ParitySummary::default();
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        let record: GenerationRecord = serde_json::from_str(&line)?;
        compare_record(generator, &record, line_index + 1, &mut summary)?;
    }
    Ok(summary)
}

fn compare_record(
    generator: &NativeGenerator<'_, '_>,
    record: &GenerationRecord,
    line_number: usize,
    summary: &mut ParitySummary,
) -> Result<(), Box<dyn std::error::Error>> {
    let morphemes: Vec<&str> = record.morphemes.iter().map(String::as_str).collect();
    let generated = generator.generate(&record.dictionary_id, &morphemes)?;
    let mut actual: Vec<GenerationResult> = generated
        .into_iter()
        .map(|analysis| GenerationResult {
            surface: analysis.surface_form,
            canonical: analysis.canonical,
        })
        .collect();
    actual.sort_unstable();
    let mut expected = record.results.clone();
    expected.sort_unstable();

    summary.requests += 1;
    summary.expected_results += expected.len();
    summary.native_results += actual.len();
    if expected == actual {
        summary.exact_requests += 1;
    } else {
        summary.mismatched_requests += 1;
        if summary.mismatched_requests <= MAX_REPORTED_MISMATCHES {
            eprintln!(
                "GENERATION_MISMATCH line={} dictionary={:?} morphemes={:?} expected={:?} actual={:?}",
                line_number, record.dictionary_id, record.morphemes, expected, actual
            );
        }
    }
    Ok(())
}

fn usage(program: &str) -> String {
    format!("usage: {program} <native-binary> <generation-oracle-jsonl>")
}
