//! What `#[compute]` generates.
//!
//! The body moves into a hidden twin and the original name becomes a wrapper that opens a tag
//! around it, so the author still writes `forward` and every caller still calls `forward`.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
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

/// Rewrite `function` into a wrapper plus the hidden twin holding its body.
pub fn expand(args: TokenStream, function: ItemFn) -> syn::Result<TokenStream> {
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

    // The parameters past `self`: forwarded to the body by name, and handed to a bounds
    // function by reference.
    let parameters = parameters(&function)?;
    let forwarded: Vec<&syn::Ident> = parameters.iter().map(|it| &it.name).collect();
    let priced: Vec<TokenStream> = parameters.iter().map(Parameter::priced).collect();

    let name = function.sig.ident.clone();
    let inner = format_ident!("__computed_{}", name);

    let mut body = function.clone();
    body.sig.ident = inner.clone();
    body.vis = syn::Visibility::Inherited;
    body.attrs.push(syn::parse_quote!(#[doc(hidden)]));
    // A parameter only the declaration reads, such as the argument a label is taken from, is
    // used by the wrapper and not by the twin, which would warn. Binding it to `_` in the twin
    // silences that one parameter, where `allow(unused_variables)` would silence every unused
    // parameter of the body.
    let mut prelude: Vec<syn::Stmt> = declared(&args, &parameters)
        .into_iter()
        .map(|name| syn::parse_quote!(let _ = &#name;))
        .collect();
    prelude.append(&mut body.block.stmts);
    body.block.stmts = prelude;

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
        Some(Expr::Path(path)) => quote!(::core::convert::Into::into(#path(self #(, #priced)*))),
        // Anything else is the call the author wrote, evaluated with every parameter in scope.
        Some(bounds) => quote!(::core::convert::Into::into(#bounds)),
    };

    let entry = name.to_string();
    let site = quote!(::core::concat!(::core::module_path!(), "::", #entry));

    let mut wrapper = function;
    wrapper.block = syn::parse_quote!({
        match burn::instrument::armed() {
            None => self.#inner(#(#forwarded),*),
            Some(run) => {
                let _tag = run.enter(burn::instrument::Declaration {
                    label: #label,
                    variant: #variant,
                    site: #site,
                    bounds: #bounds,
                });
                self.#inner(#(#forwarded),*)
            }
        }
    });

    Ok(quote! {
        #wrapper
        #body
    })
}

/// The parameters the declaration reads for itself.
///
/// `label`, `variant`, and `bounds` only as an expression: the bare-path form is handed every
/// parameter, so counting it would exempt all of them. Matched on the name as a token, which
/// over-matches a parameter shadowed inside the declaration (one warning not raised) and never
/// under-matches (a warning nobody could silence without renaming a parameter).
fn declared<'a>(args: &Args, parameters: &'a [Parameter]) -> Vec<&'a syn::Ident> {
    fn names(tokens: TokenStream, into: &mut Vec<String>) {
        for tree in tokens {
            match tree {
                proc_macro2::TokenTree::Ident(ident) => into.push(ident.to_string()),
                proc_macro2::TokenTree::Group(group) => names(group.stream(), into),
                _ => {}
            }
        }
    }

    let priced = match &args.bounds {
        Some(Expr::Path(_)) | None => None,
        Some(bounds) => Some(bounds),
    };
    let mut read = Vec::new();
    for declaration in args.label.iter().chain(&args.variant).chain(priced) {
        names(quote!(#declaration), &mut read);
    }
    parameters
        .iter()
        .map(|parameter| &parameter.name)
        .filter(|name| read.iter().any(|read| *read == name.to_string()))
        .collect()
}

/// One parameter past `self`: the name the wrapper forwards it by, and how a bounds function
/// receives it.
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

/// The parameters past `self`, each a plain binding: the wrapper forwards them by name, so a
/// pattern parameter (`(a, b): (u32, u32)`) is rejected here rather than producing an error
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
                    "`compute` forwards parameters by name, so each one needs a plain binding; \
                     destructure inside the body instead",
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
