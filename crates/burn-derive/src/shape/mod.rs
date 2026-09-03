//! Implementation of `unpack_shape!`, `assert_shape!` and `debug_assert_shape!`. The public
//! documentation lives on the macros in `lib.rs`.

use proc_macro2::{Literal, Span, TokenStream};
use quote::{quote, quote_spanned};
use syn::{
    Expr, Ident, LitInt, Token, bracketed,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    spanned::Spanned,
};

/// One position in a shape pattern.
enum Slot {
    /// Bare identifier: bound and returned by `unpack_shape!`; rejected by the assert macros.
    Fresh(Ident),
    /// `=expr` or an integer literal: the axis must equal this `usize` expression.
    Check(Box<Expr>),
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
            Ok(Slot::Check(Box::new(input.parse()?)))
        } else if input.peek(LitInt) {
            Ok(Slot::Check(Box::new(Expr::Lit(input.parse()?))))
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

/// `call` is the macro input as the caller wrote it, echoed in panic messages.
pub(crate) fn expand(input: ShapeInput, mode: Mode, call: &str) -> TokenStream {
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
    let call = format!("{name}({call})");

    let mut checks = Vec::new();
    let mut fresh = Vec::new();
    for (axis, slot) in slots.iter().enumerate() {
        let index = Literal::usize_unsuffixed(axis);
        match slot {
            Slot::Check(expr) => checks.push(quote_spanned! { expr.span() =>
                {
                    let #expected: usize = #expr;
                    ::core::assert!(
                        #dims[#index] == #expected,
                        "{}: axis {} expected {}, got {} (dims {:?})",
                        #call, #index, #expected, #dims[#index], #dims
                    );
                }
            }),
            Slot::Fresh(_) => fresh.push(quote! { #dims[#index] }),
            Slot::Wildcard => {}
        }
    }

    // The type annotation is the rank check: `Tensor<D>::dims()` returns `[usize; D]`. Going
    // through a reference keeps the caller's expression unmoved and hides the borrow from lints.
    let rank_check = quote_spanned! { tensor.span() =>
        let #t = &#tensor;
        let #dims: [usize; #rank] = #t.dims();
    };

    // With nothing to bind the block ends in a statement and evaluates to `()` implicitly; an
    // explicit `()` would trip clippy's `unused_unit` at every `assert_shape!` call site.
    let tuple = if fresh.is_empty() {
        TokenStream::new()
    } else {
        quote! { (#(#fresh,)*) }
    };

    let body = quote! {
        #rank_check
        #(#checks)*
        #tuple
    };

    match mode {
        Mode::DebugAssert => quote! { if ::core::cfg!(debug_assertions) { #body } },
        Mode::Unpack | Mode::Assert => quote! { { #body } },
    }
}
