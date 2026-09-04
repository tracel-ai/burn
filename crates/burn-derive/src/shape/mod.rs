//! Implementation of `assert_shape!` and `debug_assert_shape!`. The public macros and their
//! documentation live in burn-tensor, whose `macro_rules!` wrappers forward `$crate` so the
//! expansion can name `Tensor::dims` by path.

use proc_macro2::{Literal, Span, TokenStream};
use quote::{quote, quote_spanned};
use syn::{
    Expr, Ident, Path, Token, bracketed,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    spanned::Spanned,
};

/// One position in a shape pattern.
enum Slot {
    /// Any expression: the axis must equal this `usize` value.
    Check(Box<Expr>),
    /// `_`: the axis is not checked.
    Wildcard,
    /// `..`: any number of axes, none of them checked. At most one per pattern.
    Rest(Token![..]),
}

impl Parse for Slot {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.peek(Token![..]) {
            Ok(Slot::Rest(input.parse()?))
        } else if input.peek(Token![_]) {
            input.parse::<Token![_]>()?;
            Ok(Slot::Wildcard)
        } else {
            Ok(Slot::Check(Box::new(input.parse()?)))
        }
    }
}

/// Parsed `$crate, tensor, [slot, slot, ...]` input. The leading path is the crate that owns
/// the public macro, forwarded by its `macro_rules!` wrapper.
pub(crate) struct ShapeInput {
    krate: Path,
    /// Everything after the crate path, as rendered by the compiler's token printer: whitespace
    /// is normalized and comments are dropped. Echoed in panic messages.
    call: String,
    tensor: Expr,
    slots: Vec<Slot>,
}

impl Parse for ShapeInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let krate: Path = input.parse()?;
        input.parse::<Token![,]>()?;
        let call = input.fork().parse::<TokenStream>()?.to_string();
        let tensor: Expr = input.parse()?;
        input.parse::<Token![,]>()?;
        let content;
        bracketed!(content in input);
        let slots: Vec<Slot> = Punctuated::<Slot, Token![,]>::parse_terminated(&content)?
            .into_iter()
            .collect();
        let mut rests = slots.iter().filter_map(|slot| match slot {
            Slot::Rest(token) => Some(token),
            _ => None,
        });
        rests.next();
        if let Some(second) = rests.next() {
            return Err(syn::Error::new(
                second.span(),
                "at most one `..` per pattern",
            ));
        }
        Ok(ShapeInput {
            krate,
            call,
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

pub(crate) fn expand(input: ShapeInput, mode: Mode) -> TokenStream {
    let ShapeInput {
        krate,
        call,
        tensor,
        slots,
    } = input;

    // Mixed-site hygiene: these names are invisible to the caller, so a slot expression that
    // mentions `__dims` still resolves to the user's own binding.
    let t = Ident::new("__tensor", Span::mixed_site());
    let dims = Ident::new("__dims", Span::mixed_site());
    let rank = Ident::new("__rank", Span::mixed_site());
    let axis = Ident::new("__axis", Span::mixed_site());
    let expected = Ident::new("__expected", Span::mixed_site());

    let call = format!("{}({call})", mode.name());

    // A pattern with `..` splits into the axes before it and the axes after it. The latter are
    // addressed from the end, since the macro cannot know how many axes `..` stands for.
    let rest = slots.iter().position(|slot| matches!(slot, Slot::Rest(_)));
    let (prefix, suffix): (&[Slot], &[Slot]) = match rest {
        Some(position) => (&slots[..position], &slots[position + 1..]),
        None => (&slots[..], &[]),
    };

    let check = |index: TokenStream, expr: &Expr| {
        quote_spanned! { expr.span() =>
            {
                let #axis: usize = #index;
                let #expected: usize = #expr;
                ::core::assert!(
                    #dims[#axis] == #expected,
                    "{}: axis {} expected {}, got {} (dims {:?})",
                    #call, #axis, #expected, #dims[#axis], #dims
                );
            }
        }
    };
    let prefix_checks = prefix
        .iter()
        .enumerate()
        .filter_map(|(i, slot)| match slot {
            Slot::Check(expr) => {
                let index = Literal::usize_unsuffixed(i);
                Some(check(quote! { #index }, expr))
            }
            _ => None,
        });
    let suffix_checks = suffix
        .iter()
        .enumerate()
        .filter_map(|(j, slot)| match slot {
            Slot::Check(expr) => {
                let from_end = Literal::usize_unsuffixed(suffix.len() - j);
                Some(check(quote! { #rank - #from_end }, expr))
            }
            _ => None,
        });

    // Without `..`, the type annotation is the rank check: `Tensor<D>::dims()` returns
    // `[usize; D]`. With `..`, only a minimum rank is known, so it is checked at runtime. Calling
    // `Tensor::dims` by path rather than as a method makes any other receiver a type error;
    // `Shape::dims::<N>` would infer `N` from the annotation and truncate silently. Going through
    // a reference keeps a `&x` argument free of `clippy::needless_borrow`, which a
    // `(#tensor).dims()` receiver would trigger.
    let rank_check = match rest {
        None => {
            let exact = slots.len();
            quote_spanned! { tensor.span() =>
                let #t = &#tensor;
                let #dims: [usize; #exact] = #krate::Tensor::dims(#t);
            }
        }
        Some(_) => {
            let min = prefix.len() + suffix.len();
            let min_check = (min > 0).then(|| {
                quote! {
                    ::core::assert!(
                        #rank >= #min,
                        "{}: expected rank at least {}, got {} (dims {:?})",
                        #call, #min, #rank, #dims
                    );
                }
            });
            quote_spanned! { tensor.span() =>
                let #t = &#tensor;
                let #dims = #krate::Tensor::dims(#t);
                let #rank = #dims.len();
                #min_check
            }
        }
    };

    let body = quote! {
        #rank_check
        #(#prefix_checks)*
        #(#suffix_checks)*
    };

    match mode {
        Mode::DebugAssert => quote! { if ::core::cfg!(debug_assertions) { #body } },
        Mode::Assert => quote! { { #body } },
    }
}
