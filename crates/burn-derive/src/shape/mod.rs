//! Shape assertion macros: `unpack_shape!`, `assert_shape!` and `debug_assert_shape!`.
//!
//! All three take a tensor expression and a bracketed pattern with one slot per axis:
//!
//! - a bare identifier binds the axis size (only in `unpack_shape!`),
//! - `=expr` checks the axis against an in-scope `usize` expression,
//! - an integer literal checks the axis against that value,
//! - `_` skips the axis.
//!
//! The pattern length is the expected rank. Since `Tensor::dims()` returns `[usize; D]`,
//! binding it to `[usize; N]` turns a rank mismatch into a compile error. Axis checks are
//! runtime assertions.
//!
//! Inspired by the `burn-contracts` crate by Crutcher Dunnavant.

use proc_macro2::{Literal, Span, TokenStream};
use quote::{ToTokens, quote, quote_spanned};
use syn::{
    Expr, ExprLit, Ident, Lit, LitInt, Token, bracketed,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    spanned::Spanned,
};

/// One position in a shape pattern.
enum Slot {
    /// Bare identifier: bound and returned by `unpack_shape!`; rejected by the assert macros.
    Fresh(Ident),
    /// `=expr` or an integer literal: the axis must equal this `usize` expression.
    Check(Expr),
    /// `_`: neither bound nor checked.
    Wildcard,
}

impl Parse for Slot {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.peek(Token![_]) {
            input.parse::<Token![_]>()?;
            Ok(Slot::Wildcard)
        } else if input.peek(Token![=]) {
            input.parse::<Token![=]>()?;
            Ok(Slot::Check(input.parse()?))
        } else if input.peek(LitInt) {
            let lit: LitInt = input.parse()?;
            Ok(Slot::Check(Expr::Lit(ExprLit {
                attrs: Vec::new(),
                lit: Lit::Int(lit),
            })))
        } else if input.peek(Ident) {
            Ok(Slot::Fresh(input.parse()?))
        } else {
            Err(input.error(
                "expected a shape slot: a name to bind, `=expr`, an integer literal, or `_`",
            ))
        }
    }
}

/// Parsed `tensor, [slot, slot, ...]` input.
pub(crate) struct ShapeInput {
    tensor: Expr,
    slots: Vec<Slot>,
}

impl Parse for ShapeInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let tensor: Expr = input.parse()?;
        input.parse::<Token![,]>()?;
        let content;
        bracketed!(content in input);
        let slots = Punctuated::<Slot, Token![,]>::parse_terminated(&content)?
            .into_iter()
            .collect();
        if !input.is_empty() {
            return Err(input.error("unexpected tokens after the shape pattern"));
        }
        Ok(ShapeInput { tensor, slots })
    }
}

/// Which macro is expanding, which decides validation and the emitted assertion.
#[derive(Clone, Copy)]
pub(crate) enum Mode {
    Unpack,
    Assert,
    DebugAssert,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::Unpack => "unpack_shape!",
            Mode::Assert => "assert_shape!",
            Mode::DebugAssert => "debug_assert_shape!",
        }
    }
}

pub(crate) fn expand(input: ShapeInput, mode: Mode) -> TokenStream {
    let ShapeInput { tensor, slots } = input;
    let name = mode.name();

    match mode {
        Mode::Unpack => {
            if !slots.iter().any(|s| matches!(s, Slot::Fresh(_))) {
                return syn::Error::new(
                    Span::call_site(),
                    "`unpack_shape!` needs at least one bare identifier to bind; \
                     use `assert_shape!` when every axis is a check",
                )
                .to_compile_error();
            }
        }
        Mode::Assert | Mode::DebugAssert => {
            for slot in &slots {
                if let Slot::Fresh(id) = slot {
                    return syn::Error::new(
                        id.span(),
                        format!(
                            "`{name}` does not bind new names; bare identifier `{id}` would bind \
                             a new value. Use `unpack_shape!` to bind, or `=` to check against \
                             an in-scope value"
                        ),
                    )
                    .to_compile_error();
                }
            }
        }
    }

    // Mixed-site hygiene: these names are invisible to the caller, so a user `=expr` that
    // mentions `__dims` still resolves to the user's own binding.
    let t = Ident::new("__tensor", Span::mixed_site());
    let dims = Ident::new("__dims", Span::mixed_site());
    let expected = Ident::new("__expected", Span::mixed_site());

    let rank = slots.len();
    let call_text = format!("{name}({}, {})", source_like(&tensor), pattern_text(&slots));

    let mut checks = Vec::new();
    let mut fresh = Vec::new();
    for (axis, slot) in slots.iter().enumerate() {
        let index = Literal::usize_unsuffixed(axis);
        match slot {
            Slot::Check(expr) => {
                let msg = format!("{call_text}: axis {axis} expected {{}}, got {{}} (dims {{:?}})");
                let check = quote_spanned! { expr.span() =>
                    {
                        let #expected: usize = #expr;
                        ::core::assert!(
                            #dims[#index] == #expected,
                            #msg, #expected, #dims[#index], #dims
                        );
                    }
                };
                checks.push(check);
            }
            Slot::Fresh(_) => fresh.push(quote! { #dims[#index] }),
            Slot::Wildcard => {}
        }
    }

    // The type annotation is the rank check: `Tensor<D>::dims()` returns `[usize; D]`.
    let rank_check = if checks.is_empty() && fresh.is_empty() {
        quote_spanned! { tensor.span() => let _: [usize; #rank] = #t.dims(); }
    } else {
        quote_spanned! { tensor.span() => let #dims: [usize; #rank] = #t.dims(); }
    };

    let tuple = match fresh.len() {
        0 => TokenStream::new(),
        1 => quote! { (#(#fresh)*,) },
        _ => quote! { (#(#fresh),*) },
    };

    let body = quote! {
        let #t = &#tensor;
        #rank_check
        #(#checks)*
        #tuple
    };

    match mode {
        Mode::DebugAssert => quote! { if ::core::cfg!(debug_assertions) { #body } },
        Mode::Unpack | Mode::Assert => quote! { { #body } },
    }
}

/// Render the pattern back as the user wrote it, for panic messages.
fn pattern_text(slots: &[Slot]) -> String {
    let parts: Vec<String> = slots
        .iter()
        .map(|slot| match slot {
            Slot::Fresh(id) => id.to_string(),
            Slot::Check(Expr::Lit(lit)) => source_like(lit),
            Slot::Check(expr) => format!("={}", source_like(expr)),
            Slot::Wildcard => "_".to_string(),
        })
        .collect();
    format!("[{}]", parts.join(", "))
}

/// Undo the spacing `ToTokens` inserts between every token, then escape braces so the
/// result can be embedded in a format string.
fn source_like(tokens: &impl ToTokens) -> String {
    tokens
        .to_token_stream()
        .to_string()
        .replace(" . ", ".")
        .replace(" :: ", "::")
        .replace("& ", "&")
        .replace("( ", "(")
        .replace(" )", ")")
        .replace(" ,", ",")
        .replace('{', "{{")
        .replace('}', "}}")
}
