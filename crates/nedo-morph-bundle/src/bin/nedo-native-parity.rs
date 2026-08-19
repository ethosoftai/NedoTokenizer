use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::ExitCode;

use nedo_morph_bundle::NativeMorphology;
use serde::Deserialize;

const MAX_REPORTED_MISMATCHES: usize = 20;

#[derive(Deserialize)]
struct FixtureRecord {
    input: String,
    normalized_input: String,
    analyses: Vec<FixtureAnalysis>,
}

#[derive(Deserialize)]
struct FixtureAnalysis {
    canonical: String,
}

#[derive(Default)]
struct ParitySummary {
    records: usize,
    expected_analyses: usize,
    native_analyses: usize,
    exact_records: usize,
    mismatched_records: usize,
}

fn main() -> ExitCode {
    match run() {
        Ok(summary) => {
            println!(
                "NEDO_NATIVE_PARITY records={} exact_records={} mismatched_records={} expected_analyses={} native_analyses={}",
                summary.records,
                summary.exact_records,
                summary.mismatched_records,
                summary.expected_analyses,
                summary.native_analyses,
            );
            if summary.mismatched_records == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(error) => {
            eprintln!("native parity failed: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<ParitySummary, Box<dyn std::error::Error>> {
    let mut arguments = env::args_os();
    let program = arguments
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "nedo-native-parity".to_owned());
    let binary = arguments.next().ok_or_else(|| usage(&program))?;
    let fixture = arguments.next().ok_or_else(|| usage(&program))?;
    if arguments.next().is_some() {
        return Err(usage(&program).into());
    }
    let bytes = fs::read(binary)?;
    let morphology = NativeMorphology::parse(&bytes)?;
    compare_fixture(&morphology, Path::new(&fixture))
}

fn compare_fixture(
    morphology: &NativeMorphology<'_>,
    fixture: &Path,
) -> Result<ParitySummary, Box<dyn std::error::Error>> {
    let reader = BufReader::new(File::open(fixture)?);
    let mut summary = ParitySummary::default();
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        let record: FixtureRecord = serde_json::from_str(&line)?;
        compare_record(morphology, &record, line_index + 1, &mut summary)?;
    }
    Ok(summary)
}

fn compare_record(
    morphology: &NativeMorphology<'_>,
    record: &FixtureRecord,
    line_number: usize,
    summary: &mut ParitySummary,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut expected: Vec<&str> = record
        .analyses
        .iter()
        .map(|analysis| analysis.canonical.as_str())
        .collect();
    let native = morphology.analyze_token(&record.input)?;
    let mut actual: Vec<&str> = native
        .iter()
        .map(|analysis| analysis.canonical.as_str())
        .collect();
    expected.sort_unstable();
    actual.sort_unstable();
    summary.records += 1;
    summary.expected_analyses += expected.len();
    summary.native_analyses += actual.len();
    if expected == actual {
        summary.exact_records += 1;
    } else {
        summary.mismatched_records += 1;
        if summary.mismatched_records <= MAX_REPORTED_MISMATCHES {
            eprintln!(
                "PARITY_MISMATCH line={} input={:?} expected={:?} actual={:?}",
                line_number, record.normalized_input, expected, actual
            );
        }
    }
    Ok(())
}

fn usage(program: &str) -> String {
    format!("usage: {program} <native-binary> <zemberek-fixture-jsonl>")
}
