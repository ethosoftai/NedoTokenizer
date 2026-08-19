use std::env;
use std::fs;
use std::process::ExitCode;

use nedo_morph_bundle::PerceptronModel;

fn main() -> ExitCode {
    let Some(path) = env::args_os().nth(1) else {
        eprintln!("usage: nedo-model-validate <model-compressed>");
        return ExitCode::from(2);
    };
    match fs::read(path)
        .map_err(|error| error.to_string())
        .and_then(|bytes| PerceptronModel::parse(&bytes).map_err(|error| error.to_string()))
    {
        Ok(model) => {
            println!("NEDO_AMBIGUITY_MODEL_OK entries={}", model.len());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("ambiguity model validation failed: {error}");
            ExitCode::FAILURE
        }
    }
}
