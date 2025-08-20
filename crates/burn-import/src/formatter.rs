use proc_macro2::TokenStream;
use rust_format::{Config, Formatter, PostProcess, PrettyPlease};

/// Formats a token stream into a string.
pub fn format_tokens(tokens: TokenStream) -> String {
    let fmt = code_formatter();

    fmt.format_tokens(tokens).expect("Valid token tree")
}

fn code_formatter() -> PrettyPlease {
    let config = Config::new_str().post_proc(PostProcess::ReplaceMarkersAndDocBlocks);

    PrettyPlease::from_config(config)
}
