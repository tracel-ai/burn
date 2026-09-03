//! Implementation of `assert_shape!` and `debug_assert_shape!`. The public macros and their
//! documentation live in burn-tensor, whose `macro_rules!` wrappers forward `$crate` so the
//! expansion can name `Tensor::dims` by path.

use proc_macro2::{Literal, Span, TokenStream};
use quote::{quote, quote_spanned};
use syn::{
    Expr, Ident, LitInt, Path, Token, bracketed,
    parse::{Parse, ParseStream},
    spanned::Spanned,
};

/// One position in a shape pattern.
enum Slot {
    /// `=expr` or an integer literal: the axis must equal this `usize` expression.
    Check(Box<Expr>),
    /// `_`: the axis is not checked.
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
            let id: Ident = input.parse()?;
            Err(syn::Error::new(
                id.span(),
                format!(
                    "bare identifier `{id}` is not a check; use `={id}` to compare the axis \
                     against an in-scope value, `_` to skip it, or `let [..] = tensor.dims()` \
                     to name axes"
                ),
            ))
        } else {
            Err(input.error("expected a shape slot: `=expr`, an integer literal, or `_`"))
        }
    }
}

/// Parsed `$crate, tensor, [slot, slot, ...]` input. The leading path is the crate that owns
/// the public macro, forwarded by its `macro_rules!` wrapper.
pub(crate) struct ShapeInput {
    krate: Path,
    tensor: Expr,
    slots: Vec<Slot>,
}

impl Parse for ShapeInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let krate: Path = input.parse()?;
        input.parse::<Token![,]>()?;
        let tensor: Expr = input.parse()?;
        input.parse::<Token![,]>()?;
        let content;
        bracketed!(content in input);
        let mut slots = Vec::new();
        while !content.is_empty() {
            slots.push(content.parse()?);
            if content.is_empty() {
                break;
            }
            if !content.peek(Token![,]) {
                return Err(content.error(
                    "expected `,`; to check an axis against an expression, prefix it with `=`",
                ));
            }
            content.parse::<Token![,]>()?;
        }
        Ok(ShapeInput {
            krate,
            tensor,
            slots,
        })
    }
}

/// Which macro is expanding. It decides the name in messages and the debug gating.
#[derive(Clone, Copy)]
pub(crate) enum Mode {
    Assert,
    DebugAssert,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::Assert => "assert_shape!",
            Mode::DebugAssert => "debug_assert_shape!",
        }
    }
}

/// `call` is the macro input after the crate path, as rendered by the compiler's token
/// printer: whitespace is normalized and comments are dropped. It is echoed in panic messages.
pub(crate) fn expand(input: ShapeInput, mode: Mode, call: &str) -> TokenStream {
    let ShapeInput {
        krate,
        tensor,
        slots,
    } = input;

    // Mixed-site hygiene: these names are invisible to the caller, so a user `=expr` that
    // mentions `__dims` still resolves to the user's own binding.
    let t = Ident::new("__tensor", Span::mixed_site());
    let dims = Ident::new("__dims", Span::mixed_site());
    let expected = Ident::new("__expected", Span::mixed_site());

    let rank = slots.len();
    let call = format!("{}({call})", mode.name());

    let checks = slots
        .iter()
        .enumerate()
        .filter_map(|(axis, slot)| match slot {
            Slot::Check(expr) => {
                let index = Literal::usize_unsuffixed(axis);
                Some(quote_spanned! { expr.span() =>
                    {
                        let #expected: usize = #expr;
                        ::core::assert!(
                            #dims[#index] == #expected,
                            "{}: axis {} expected {}, got {} (dims {:?})",
                            #call, #index, #expected, #dims[#index], #dims
                        );
                    }
                })
            }
            Slot::Wildcard => None,
        });

    // The type annotation is the rank check: `Tensor<D>::dims()` returns `[usize; D]`. Calling
    // `Tensor::dims` by path rather than as a method makes any other receiver a type error;
    // `Shape::dims::<N>` would infer `N` from the annotation and truncate silently. Going through
    // a reference keeps a `&x` argument free of `clippy::needless_borrow`, which a
    // `(#tensor).dims()` receiver would trigger.
    let body = quote_spanned! { tensor.span() =>
        let #t = &#tensor;
        let #dims: [usize; #rank] = #krate::Tensor::dims(#t);
        #(#checks)*
    };

    match mode {
        Mode::DebugAssert => quote! { if ::core::cfg!(debug_assertions) { #body } },
        Mode::Assert => quote! { { #body } },
    }
}
