use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use nedo_morph_bundle::{compile_binary, BinaryBundleView, MorphBundle};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("native bundle compilation failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os();
    let program = arguments
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "nedo-bundle-compile".to_owned());
    let input = arguments.next().ok_or_else(|| usage(&program))?;
    let output = arguments.next().ok_or_else(|| usage(&program))?;
    if arguments.next().is_some() {
        return Err(usage(&program).into());
    }
    let bundle = MorphBundle::load_directory(Path::new(&input))?;
    let bytes = compile_binary(&bundle)?;
    let view = BinaryBundleView::parse(&bytes)?;
    atomic_write(Path::new(&output), &bytes)?;
    let summary = view.summary();
    println!(
        "NEDO_MORPH_BINARY_OK strings={} morphemes={} dictionary={} stems={} states={} edges={} aliases={} templates={} condition_bytes={} null_sentinels={} file_bytes={}",
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
    Ok(())
}

fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("nedo-morph.bin");
    let temporary = parent.join(format!(".{file_name}.tmp.{}", std::process::id()));
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, path)
}

fn usage(program: &str) -> String {
    format!("usage: {program} <reference-bundle-directory> <output-file>")
}
