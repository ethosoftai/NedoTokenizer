use std::env;
use std::path::Path;
use std::process::ExitCode;

use nedo_morph_bundle::MorphBundle;

fn main() -> ExitCode {
    let mut arguments = env::args_os();
    let program = arguments
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "nedo-bundle-validate".to_owned());
    let Some(path) = arguments.next() else {
        eprintln!("usage: {program} <bundle-directory>");
        return ExitCode::from(2);
    };
    if arguments.next().is_some() {
        eprintln!("usage: {program} <bundle-directory>");
        return ExitCode::from(2);
    }

    match MorphBundle::load_directory(Path::new(&path)) {
        Ok(bundle) => {
            let summary = bundle.summary();
            println!(
                "NEDO_MORPH_BUNDLE_OK morphemes={} dictionary={} stems={} states={} edges={} owner_mismatches={} duplicate_state_ids={} condition_kinds={}",
                summary.morphemes,
                summary.dictionary,
                summary.stems,
                summary.states,
                summary.edges,
                summary.owner_mismatches,
                summary.duplicate_state_ids,
                summary.condition_kinds,
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("bundle validation failed: {error}");
            ExitCode::FAILURE
        }
    }
}
