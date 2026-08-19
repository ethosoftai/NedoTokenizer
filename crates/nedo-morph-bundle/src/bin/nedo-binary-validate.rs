use std::env;
use std::fs;
use std::process::ExitCode;

use nedo_morph_bundle::BinaryBundleView;

fn main() -> ExitCode {
    let mut arguments = env::args_os();
    let program = arguments
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "nedo-binary-validate".to_owned());
    let Some(path) = arguments.next() else {
        eprintln!("usage: {program} <native-bundle-file>");
        return ExitCode::from(2);
    };
    if arguments.next().is_some() {
        eprintln!("usage: {program} <native-bundle-file>");
        return ExitCode::from(2);
    }
    match fs::read(&path)
        .map_err(|error| error.to_string())
        .and_then(|bytes| {
            BinaryBundleView::parse(&bytes)
                .map(BinaryBundleView::summary)
                .map_err(|error| error.to_string())
        }) {
        Ok(summary) => {
            println!(
                "NEDO_MORPH_BINARY_VALID strings={} morphemes={} dictionary={} stems={} states={} edges={} aliases={} templates={} condition_bytes={} null_sentinels={} file_bytes={}",
                summary.string_count,
                summary.morpheme_count,
                summary.dictionary_count,
                summary.stem_count,
                summary.state_count,
                summary.edge_count,
                summary.alias_count,
                summary.template_token_count,
                summary.condition_byte_count,
                summary.null_dictionary_sentinel_count,
                summary.file_byte_count,
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("native bundle validation failed: {error}");
            ExitCode::FAILURE
        }
    }
}
