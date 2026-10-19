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

/// Open a tag around `function`'s body.
pub fn expand(args: TokenStream, mut function: ItemFn) -> syn::Result<TokenStream> {
    let args: Args = syn::parse2(args)?;

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
        // A bare path is called with every parameter by reference.
        Some(Expr::Path(path)) => {
            let priced = parameters(&function)?;
            let priced = priced.iter().map(Parameter::priced);
            quote!(::core::convert::Into::into(#path(self #(, #priced)*)))
        }
        // Anything else is the call the author wrote, evaluated with every parameter in scope.
        Some(bounds) => quote!(::core::convert::Into::into(#bounds)),
    };

    let entry = function.sig.ident.to_string();
    let site = quote!(::core::concat!(::core::module_path!(), "::", #entry));

    let body = &function.block;
    // Bound first, so the tag outlives every local of the body and closes on every return. The
    // declaration is evaluated before the body runs, so a parameter it reads has not been moved.
    function.block = syn::parse_quote!({
        let __burn_instrument_tag = match burn::instrument::armed() {
            None => None,
            Some(observing) => Some(observing.enter(burn::instrument::Declaration {
                label: #label,
                variant: #variant,
                site: #site,
                bounds: #bounds,
            })),
        };
        #body
    });

    Ok(quote!(#function))
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
