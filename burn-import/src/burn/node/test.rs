use proc_macro2::TokenStream;
use rust_format::{Config, Edition, Formatter, PostProcess, RustFmt};

pub fn assert_tokens(tokens1: TokenStream, tokens2: TokenStream) {
    let fmt = code_formatter();

    let tokens1 = fmt.format_tokens(tokens1).unwrap();
    let tokens2 = fmt.format_tokens(tokens2).unwrap();

    pretty_assertions::assert_eq!(tokens1, tokens2);
}

fn code_formatter() -> RustFmt {
    let config = Config::new_str()
        .post_proc(PostProcess::ReplaceMarkersAndDocBlocks)
        .edition(Edition::Rust2021);

    RustFmt::from_config(config)
}
