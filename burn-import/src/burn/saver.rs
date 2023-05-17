use proc_macro2::TokenStream;
use rust_format::{Config, Edition, Formatter, PostProcess, RustFmt};
use std::path::PathBuf;

pub fn save(tokens: TokenStream, mut out: PathBuf) {
    out.set_extension("rs");

    std::fs::write(out, format_tokens(tokens)).unwrap();
}

pub fn format_tokens(tokens: TokenStream) -> String {
    code_formatter().format_tokens(tokens).unwrap()
}

fn code_formatter() -> RustFmt {
    let config = Config::new_str()
        .post_proc(PostProcess::ReplaceMarkersAndDocBlocks)
        .edition(Edition::Rust2021);

    RustFmt::from_config(config)
}
