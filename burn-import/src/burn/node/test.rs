use crate::format_tokens;
use proc_macro2::TokenStream;

#[track_caller]
pub fn assert_tokens(tokens1: TokenStream, tokens2: TokenStream) {
    let tokens1 = format_tokens(tokens1);
    let tokens2 = format_tokens(tokens2);

    pretty_assertions::assert_eq!(tokens1, tokens2);
}
