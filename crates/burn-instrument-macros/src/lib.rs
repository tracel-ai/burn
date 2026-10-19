#![warn(missing_docs)]

//! The attribute a component declares itself with, re-exported as `burn::instrument::compute`.
//!
//! The declaration has to sit between the method an author writes and the call every caller
//! makes. A trait cannot put it there: a provided method wrapping an inner one renames what the
//! author writes, and a composite that forgets to call a `measured(..)` helper silently drops a
//! level out of the tree. Generating the tag into the method leaves both names alone, so an
//! annotated method cannot be called without its declaration, and declared bounds cannot drift
//! from the signature they price.
//!
//! The expansion is checked in `burn-instrument`, where the paths it names resolve.

mod compute;

use proc_macro::TokenStream;

/// Declare a method a unit of compute, leaving how it is written and how it is called as they
/// were.
///
/// ```ignore
/// #[compute]                                                    // a row, timed
/// #[compute(bounds = Self::bounds)]                             // and a floor
/// #[compute(bounds = self.bounds(&x))]                          // priced by an expression
/// #[compute(label = self.role, bounds = Self::bounds)]          // a role named by the parent
/// #[compute(label = "attention core", variant = self.kernel().name())]
/// ```
///
/// * `label`: an expression evaluated in the method's scope naming this component's row.
///   Defaults to the type's name as words (`RmsNorm` is `rms norm`), which is right except for
///   a type serving several roles under one parent, such as the projections of an attention
///   block: rows merge on their label among siblings.
/// * `variant`: an expression naming which execution served this entry, where that is chosen at
///   call time, yielding `&str` or `Option<&str>`. Rows still merge on their label; the
///   breakdown keeps a histogram of variants per row.
/// * `bounds`: this entry's `burn::instrument::Bounds`, the floor for running it alone, as the
///   call that computes it, evaluated with every parameter in scope. A bare path (`Self::bounds`)
///   is called with `self` and every parameter by reference. It may yield `Bounds` or
///   `Option<Bounds>`. Omitted, the component is timed and not scored, which is right for one
///   that will be fused into its neighbour.
///
/// The method must take `self` (the type names the row), and with a bare-path `bounds` bind each
/// parameter to a plain name (the path is handed them by name). It may be an inherent method or a
/// trait method. A parameter only the declaration reads does not warn as unused.
///
/// With no observer installed, the method runs its body after one thread-local read.
#[proc_macro_attribute]
pub fn compute(args: TokenStream, item: TokenStream) -> TokenStream {
    let function = syn::parse_macro_input!(item as syn::ItemFn);
    compute::expand(args.into(), function)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
