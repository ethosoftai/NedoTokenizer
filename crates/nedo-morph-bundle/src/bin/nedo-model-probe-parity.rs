use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::process::ExitCode;

use nedo_morph_bundle::PerceptronModel;
use serde::Deserialize;

#[derive(Deserialize)]
struct Probe {
    key: String,
    bits: i32,
}

fn main() -> ExitCode {
    match run() {
        Ok((keys, exact)) => {
            println!("NEDO_MODEL_PROBE_PARITY keys={keys} exact={exact}");
            if keys == exact {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(error) => {
            eprintln!("model probe parity failed: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os();
    let program = arguments
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "nedo-model-probe-parity".to_owned());
    let model_path = arguments.next().ok_or_else(|| usage(&program))?;
    let probe_path = arguments.next().ok_or_else(|| usage(&program))?;
    if arguments.next().is_some() {
        return Err(usage(&program).into());
    }
    let model = PerceptronModel::parse(&fs::read(model_path)?)?;
    let reader = BufReader::new(File::open(probe_path)?);
    let mut keys = 0_usize;
    let mut exact = 0_usize;
    for line in reader.lines() {
        let probe: Probe = serde_json::from_str(&line?)?;
        let actual = model.get(&probe.key).to_bits().cast_signed();
        keys += 1;
        if actual == probe.bits {
            exact += 1;
        } else if keys - exact <= 20 {
            eprintln!(
                "MODEL_PROBE_MISMATCH key={:?} expected_bits={} actual_bits={}",
                probe.key, probe.bits, actual
            );
        }
    }
    Ok((keys, exact))
}

fn usage(program: &str) -> String {
    format!("usage: {program} <model-compressed> <probe-jsonl>")
}
