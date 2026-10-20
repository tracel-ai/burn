//! What `#[compute]` generates.
//!
//! The method opens a tag before its body runs, and the tag closes when the method returns, so
//! the author still writes `forward` and every caller still calls `forward`. The body stays where
//! it was written rather than moving into another item, so the method keeps its signature as
//! written (`mut` parameters, `impl Trait` returns, lifetimes), and a trait method can be
//! annotated as well as an inherent one.

use proc_macro2::TokenStream;
use quote::quote;
use syn::spanned::Spanned;
use syn::{Expr, FnArg, ItemFn, Pat, Token};

/// The parsed `#[compute(..)]` arguments.
#[derive(Default)]
struct Args {
    /// Expression naming this component, evaluated in the function's scope.
    label: Option<Expr>,
    /// Expression naming which execution served this entry, evaluated in the function's scope.
    variant: Option<Expr>,
    /// This entry's floor as an expression, or, as a bare path, the function that computes it
    /// from every parameter.
    bounds: Option<Expr>,
}

impl syn::parse::Parse for Args {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut args = Args::default();
        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            match key.to_string().as_str() {
                "label" => args.label = Some(input.parse()?),
                "variant" => args.variant = Some(input.parse()?),
                // Tokens rather than a string, so go-to-definition and rename keep working.
                "bounds" => args.bounds = Some(input.parse()?),
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown `compute` argument `{other}`, expected `label`, `variant` \
                             or `bounds`"
                        ),
                    ));
                }
            }
            if input.is_empty() {
                break;
            }
            // Separated by commas, with a trailing one allowed.
            input.parse::<Token![,]>()?;
        }
        Ok(args)
    }
}

/// Open a tag around the body of the method `item`.
pub fn expand(args: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    let args: Args = syn::parse2(args)?;
    let mut function = annotated(item)?;

    let receiver = function.sig.inputs.first().and_then(|arg| match arg {
        FnArg::Receiver(receiver) => Some(receiver),
        FnArg::Typed(_) => None,
    });
    if receiver.is_none() {
        return Err(syn::Error::new(
            function.sig.span(),
            "`compute` annotates a method: it needs a `self` to name the component and to \
             reach the component's own bounds",
        ));
    }

    let label = match &args.label {
        Some(label) => quote!(#label),
        None => quote!(burn::instrument::role_label::<Self>()),
    };
    // `Into`, so a variant may be `&'static str` or `Option<&'static str>`.
    let variant = match &args.variant {
        Some(variant) => quote!(::core::convert::Into::into(#variant)),
        None => quote!(None),
    };
    // `Into` in both arms, so the bounds may be `Bounds` or `Option<Bounds>`: a component that
    // has no floor for what it was handed declines rather than declaring a zero.
    let bounds = match &args.bounds {
        None => quote!(None),
        // A function's path is called with `self` and every parameter by reference.
        Some(Expr::Path(path)) if names_a_function(path, &function) => {
            let priced = parameters(&function)?;
            let priced = priced.iter().map(Parameter::priced);
            quote!(::core::convert::Into::into(#path(&self #(, #priced)*)))
        }
        // Anything else is the value or the call the author wrote, evaluated with every
        // parameter in scope.
        Some(bounds) => quote!(::core::convert::Into::into(#bounds)),
    };

    let entry = function.sig.ident.to_string();
    let site = quote!(::core::concat!(::core::module_path!(), "::", #entry));

    // The body's statements rather than its block, so a one-line body is not a block in tail
    // position that rustc would call unnecessary.
    let body = &function.block.stmts;
    // Bound first, so the tag outlives every local of the body and closes on every return. The
    // declaration is evaluated before the body runs, so a parameter it reads has not been moved.
    // The names are the macro's own, so they cannot hide a parameter the declaration reads.
    function.block = syn::parse_quote!({
        let __burn_instrument_tag = match burn::instrument::observing() {
            None => None,
            Some(__burn_instrument_observing) => Some(__burn_instrument_observing.enter(
                burn::instrument::Declaration {
                label: #label,
                variant: #variant,
                site: #site,
                bounds: #bounds,
                },
            )),
        };
        #(#body)*
    });

    Ok(quote!(#function))
}

/// The method `item` annotates, refused where a tag cannot be held around its body.
fn annotated(item: TokenStream) -> syn::Result<ItemFn> {
    let function: ItemFn = match syn::parse2(item.clone()) {
        Ok(function) => function,
        // Told apart from any other parse error so a required trait method is named for what it
        // is rather than reported as missing braces.
        Err(error) => match syn::parse2::<syn::TraitItemFn>(item) {
            Ok(required) => {
                return Err(syn::Error::new(
                    required.sig.span(),
                    "`compute` wraps a body, and a required trait method has none: annotate each \
                     implementation instead",
                ));
            }
            Err(_) => return Err(error),
        },
    };
    if let Some(asyncness) = function.sig.asyncness {
        return Err(syn::Error::new(
            asyncness.span(),
            "`compute` holds a tag on the calling thread while the body runs, and an `async fn` \
             runs across polls that interleave and move between threads: annotate the methods \
             it calls instead",
        ));
    }
    if let Some(constness) = function.sig.constness {
        return Err(syn::Error::new(
            constness.span(),
            "`compute` reads the thread's observer, which a `const fn` cannot",
        ));
    }
    Ok(function)
}

/// Whether `bounds = path` names a function to call with every parameter, rather than a value.
///
/// Told by Rust's naming conventions, which its lints hold code to: a function is snake_case,
/// where a constant (`FLOOR`) and a variant (`None`) are not. A parameter is snake_case too and
/// is a value, so a path naming one is taken as written.
fn names_a_function(path: &syn::ExprPath, function: &ItemFn) -> bool {
    let Some(last) = path.path.segments.last() else {
        return false;
    };
    let name = last.ident.to_string();
    let snake_case = name
        .chars()
        .next()
        .is_some_and(|first| first.is_lowercase() || first == '_');
    let parameter = path.path.segments.len() == 1
        && function.sig.inputs.iter().any(|input| match input {
            FnArg::Typed(typed) => {
                matches!(typed.pat.as_ref(), Pat::Ident(binding) if binding.ident == last.ident)
            }
            FnArg::Receiver(_) => false,
        });
    snake_case && !parameter
}

/// One parameter past `self`: the name a bare-path `bounds` is handed it by, and how.
struct Parameter {
    name: syn::Ident,
    /// Declared `&mut`, so a bounds function is handed a `&mut` too: a cache may answer how long
    /// each sequence is only through `&mut` accessors. `&mut T` coerces to `&T`, so a bounds
    /// function that only reads still declares `&T`.
    exclusive: bool,
}

impl Parameter {
    /// The argument a bounds function is called with.
    fn priced(&self) -> TokenStream {
        let name = &self.name;
        match self.exclusive {
            true => quote!(&mut *#name),
            false => quote!(&#name),
        }
    }
}

/// The parameters past `self`, each a plain binding: a bare-path `bounds` is handed them by name,
/// so a pattern parameter (`(a, b): (u32, u32)`) is rejected here rather than producing an error
/// inside generated code.
fn parameters(function: &ItemFn) -> syn::Result<Vec<Parameter>> {
    function
        .sig
        .inputs
        .iter()
        .skip(1)
        .map(|arg| {
            let FnArg::Typed(typed) = arg else {
                return Err(syn::Error::new(arg.span(), "expected a typed parameter"));
            };
            let Pat::Ident(ident) = typed.pat.as_ref() else {
                return Err(syn::Error::new(
                    typed.pat.span(),
                    "a bare-path `bounds` is handed every parameter by name, so each one needs a \
                     plain binding; destructure inside the body instead",
                ));
            };
            Ok(Parameter {
                name: ident.ident.clone(),
                exclusive: matches!(
                    typed.ty.as_ref(),
                    syn::Type::Reference(reference) if reference.mutability.is_some()
                ),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An `async fn` would hold its tag across polls, so futures interleaved on one thread would
    /// nest under each other: it is refused rather than measured wrongly.
    #[test]
    fn an_async_method_is_refused() {
        let refused = expand(
            quote!(),
            quote!(
                async fn forward(&self) {}
            ),
        )
        .unwrap_err();
        assert!(refused.to_string().contains("async fn"), "{refused}");
    }

    /// A `const fn` cannot read the thread's observer, so it is refused with that reason rather
    /// than with errors inside the generated code.
    #[test]
    fn a_const_method_is_refused() {
        let refused = expand(
            quote!(),
            quote!(
                const fn forward(&self) {}
            ),
        )
        .unwrap_err();
        assert!(refused.to_string().contains("const fn"), "{refused}");
    }

    /// A required trait method has no body to wrap, and is told so rather than reported as
    /// missing braces.
    #[test]
    fn a_required_trait_method_is_refused_by_name() {
        let refused = expand(
            quote!(),
            quote!(
                fn forward(&self);
            ),
        )
        .unwrap_err();
        assert!(
            refused.to_string().contains("required trait method"),
            "{refused}"
        );
    }
}
