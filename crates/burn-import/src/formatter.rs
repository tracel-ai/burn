use proc_macro2::TokenStream;
use rust_format::{Config, Edition, Formatter, PostProcess, RustFmt};

/// Formats a token stream into a string.
pub fn format_tokens(tokens: TokenStream) -> String {
    let fmt = code_formatter();

    fmt.format_tokens(tokens).expect("Valid token tree")
}

fn code_formatter() -> RustFmt {
    let config = Config::new_str()
        .post_proc(PostProcess::ReplaceMarkersAndDocBlocks)
        .edition(Edition::Rust2021);

    RustFmt::from_config(config)
}
