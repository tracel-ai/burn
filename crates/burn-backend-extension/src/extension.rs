//! `#[backend_extension]` parsing, validation, and frontend-specific dispatch generation.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{
    Attribute, FnArg, GenericArgument, GenericParam, Ident, ItemTrait, Meta, Pat, PathArguments,
    ReturnType, Signature, Token, TraitItem, Type, TypeParamBound,
};

use crate::{
    BACKENDS,
    ir::{Backend, InputKind, Invocation, Operation, OperationInput, OperationOutput, TensorKind},
    routing::{self, OutputRoute},
};

pub(crate) fn expand(attr: TokenStream2, item: TokenStream2) -> syn::Result<TokenStream2> {
    let backends: Backends = syn::parse2(attr)?;
    let trait_definition: ItemTrait = syn::parse2(item)?;
    let extension = lower_extension(backends, &trait_definition)?;
    Ok(expand_extension(extension, trait_definition))
}

fn validate_backend(ident: &Ident) -> syn::Result<()> {
    let name = ident.to_string();
    if BACKENDS.iter().any(|backend| backend.name == name) {
        Ok(())
    } else {
        Err(syn::Error::new_spanned(
            ident,
            format!("Unsupported backend `{name}`"),
        ))
    }
}

struct Backends {
    concrete: Vec<Backend>,
    autodiff: (bool, Option<Meta>),
}

// Helper to parse backend idents w/ optional cfg
struct BackendArg {
    id: Ident,
    cfg: Option<Meta>,
}

impl Parse for BackendArg {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let id: Ident = input.parse()?;
        let cfg = if input.peek(Token![:]) {
            input.parse::<Token![:]>()?;

            // This parses cfg(feature = "...") or any other meta item
            let meta: syn::Meta = input.parse()?;
            Some(meta)
        } else {
            None
        };

        Ok(Self { id, cfg })
    }
}

impl Parse for Backends {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let args = Punctuated::<BackendArg, Token![,]>::parse_terminated(input)?;

        let mut concrete = vec![];
        let mut autodiff = (false, None);

        for arg in args {
            if arg.id == "Autodiff" {
                autodiff = (true, arg.cfg);
                continue;
            }

            validate_backend(&arg.id)?;
            concrete.push(Backend {
                ident: arg.id,
                cfg_attr: arg.cfg.map(|cfg| quote!(#[#cfg])),
            });
        }

        Ok(Backends { concrete, autodiff })
    }
}

struct Extension {
    trait_name: Ident,
    backends: Backends,
    ops: Vec<ExtensionOperation>,
}

struct ExtensionOperation {
    operation: Operation,
    attrs: Vec<Attribute>,
    signature: Signature,
    returns_future: bool,
}

impl core::ops::Deref for ExtensionOperation {
    type Target = Operation;

    fn deref(&self) -> &Self::Target {
        &self.operation
    }
}

fn extract_future_output_type(ty: &Type) -> Option<&Type> {
    if let Type::ImplTrait(impl_trait) = ty {
        for bound in &impl_trait.bounds {
            if let TypeParamBound::Trait(trait_bound) = bound {
                let last_segment = trait_bound.path.segments.last()?;
                if last_segment.ident == "Future"
                    && let PathArguments::AngleBracketed(args) = &last_segment.arguments
                {
                    for arg in &args.args {
                        if let GenericArgument::AssocType(assoc) = arg
                            && assoc.ident == "Output"
                        {
                            return Some(&assoc.ty);
                        }
                    }
                }
            }
        }
    }
    None
}

fn lower_extension(attr: Backends, item: &ItemTrait) -> syn::Result<Extension> {
    let mut ops = Vec::new();

    for trait_item in &item.items {
        let TraitItem::Fn(f) = trait_item else {
            continue;
        };
        if f.sig.constness.is_some() {
            return Err(syn::Error::new_spanned(
                f.sig.constness,
                "const backend extension operations aren't supported",
            ));
        }
        if let Some(variadic) = &f.sig.variadic {
            return Err(syn::Error::new_spanned(
                variadic,
                "variadic backend extension operations aren't supported",
            ));
        }

        // Parse Inputs
        let mut inputs = Vec::new();
        for arg in &f.sig.inputs {
            let FnArg::Typed(pt) = arg else {
                return Err(syn::Error::new_spanned(
                    arg,
                    "backend extension operations can't take a receiver",
                ));
            };
            let name = match pt.pat.as_ref() {
                Pat::Ident(p) if p.by_ref.is_none() && p.subpat.is_none() => p.ident.clone(),
                _ => return Err(syn::Error::new_spanned(&pt.pat, "Unsupported pattern")),
            };
            // An argument annotated with `#[extension_type]` is a custom struct or enum of tensor
            // primitives (the input counterpart of the `#[derive(ExtensionType)]` output path).
            let is_ext = pt
                .attrs
                .iter()
                .any(|attr| attr.path().is_ident("extension_type"));
            let kind = if is_ext {
                validate_extension_ty(&pt.ty)?;
                InputKind::Extension(Box::new((*pt.ty).clone()))
            } else if let Type::Reference(reference) = pt.ty.as_ref()
                && let Some(kind) = TensorKind::from_type(&reference.elem)
            {
                if reference.mutability.is_some() {
                    return Err(syn::Error::new_spanned(
                        reference,
                        "mutable tensor references aren't supported by backend extensions",
                    ));
                }
                InputKind::Tensor {
                    kind,
                    borrowed: true,
                }
            } else if let Some(kind) = InputKind::owned_tensor(&pt.ty) {
                kind
            } else {
                InputKind::Other
            };
            inputs.push(OperationInput { name, kind });
        }

        // Parse outputs
        let (actual_ty, returns_future) = match &f.sig.output {
            ReturnType::Default => {
                return Err(syn::Error::new_spanned(
                    &f.sig.output,
                    "Operations must return a value",
                ));
            }
            ReturnType::Type(_, ty) => {
                // If it's `impl Future<Output = T>`, extract T and mark as async.
                // Otherwise, use the type as-is and check for `async fn`.
                if let Some(out_ty) = extract_future_output_type(ty) {
                    (out_ty, true)
                } else {
                    (ty.as_ref(), false)
                }
            }
        };

        let output = OperationOutput::extension(actual_ty);
        let generic_args = f
            .sig
            .generics
            .params
            .iter()
            .filter_map(|parameter| match parameter {
                GenericParam::Type(parameter) => Some(parameter.ident.clone()),
                GenericParam::Const(parameter) => Some(parameter.ident.clone()),
                GenericParam::Lifetime(_) => None,
            })
            .collect();
        let mut signature = f.sig.clone();
        strip_extension_type_attributes(&mut signature);

        ops.push(ExtensionOperation {
            operation: Operation {
                name: f.sig.ident.clone(),
                inputs,
                output,
                invocation: Invocation::Trait {
                    trait_name: item.ident.clone(),
                    await_call: f.sig.asyncness.is_some() || returns_future,
                    unsafe_call: f.sig.unsafety.is_some(),
                    generic_args,
                },
            },
            attrs: f.attrs.clone(),
            signature,
            returns_future,
        });
    }

    Ok(Extension {
        trait_name: item.ident.clone(),
        backends: attr,
        ops,
    })
}

fn strip_extension_type_attributes(signature: &mut Signature) {
    for argument in &mut signature.inputs {
        if let FnArg::Typed(argument) = argument {
            argument
                .attrs
                .retain(|attribute| !attribute.path().is_ident("extension_type"));
        }
    }
}

fn expand_extension(ir: Extension, mut original_trait: ItemTrait) -> TokenStream2 {
    let trait_name = &ir.trait_name;

    // `#[extension_type]` is a helper attribute understood only by this macro. Strip it from the
    // argument list before re-emitting the trait, otherwise rustc rejects it as an unknown attribute.
    for item in &mut original_trait.items {
        if let TraitItem::Fn(f) = item {
            strip_extension_type_attributes(&mut f.sig);
        }
    }

    // Generate Dispatch Implementation
    let dispatch_methods = ir.ops.iter().map(|op| gen_dispatch_method(&ir, op));

    quote! {
        #original_trait

        impl #trait_name for burn::backend::Dispatch {
            #( #dispatch_methods )*
        }
    }
}

fn gen_dispatch_method(ir: &Extension, op: &ExtensionOperation) -> TokenStream2 {
    let name = &op.name;
    let has_ad = ir.backends.autodiff.0;

    let has_tensor_input = op
        .inputs
        .iter()
        .any(|a| matches!(a.kind, InputKind::Tensor { .. }));

    let has_ext_input = op
        .inputs
        .iter()
        .any(|a| matches!(a.kind, InputKind::Extension(_)));

    let body = if !has_tensor_input && !has_ext_input {
        // No tensor input to select the backend from (e.g. `fn load_data(i: usize) -> FloatTensor`).
        // There is nothing to match on, so this is only well-defined for a single backend — the
        // remote backend is the motivating case (`#[backend_extension(Remote)]`), where the op is
        // shipped to the server. Dispatch directly to that backend; reject the ambiguous cases.
        if has_ad {
            quote! { compile_error!("A backend extension operation with no tensor inputs can't be combined with `Autodiff` — there is no input tensor to carry the autodiff graph.") }
        } else if ir.backends.concrete.len() == 1 {
            let backend = &ir.backends.concrete[0];
            let call = gen_backend_call(ir, op, backend, OutputRoute::ConcreteDisabled);
            match &backend.cfg_attr {
                // Ungated backend: dispatch straight to it.
                None => quote! {
                    let __ad_ctx = burn::backend::DispatchAutodiffContext::Disabled;
                    #call
                },
                // The single backend is `cfg`-gated. Mirror the match path: gate the call on the
                // backend's cfg and fall back to `unimplemented!` when it's compiled out, so the
                // method still has a valid body instead of referencing a backend that doesn't
                // exist.
                Some(cfg_attr) => quote! {
                    match () {
                        #cfg_attr
                        () => {
                            let __ad_ctx = burn::backend::DispatchAutodiffContext::Disabled;
                            #call
                        }
                        #[allow(unreachable_patterns)]
                        _ => unimplemented!("Backend not supported for custom op `{}`", stringify!(#name)),
                    }
                },
            }
        } else {
            quote! { compile_error!("A backend extension operation with no tensor inputs must list exactly one backend (e.g. `#[backend_extension(Remote)]`), since there is no input tensor to select the backend from.") }
        }
    } else {
        // Select the concrete backend by peeking through the dispatch representation before moving
        // any inputs. This is required for autodiff float tensors, whose backend identity is nested
        // inside `DispatchTensorKind::Autodiff`, and also supports extension structs and enums.
        gen_tensor_input_dispatch_body(ir, op)
    };

    let signature = &op.signature;
    let attrs = &op.attrs;
    if op.returns_future {
        quote! {
            #(#attrs)*
            #signature {
                async move { #body }
            }
        }
    } else {
        quote! {
            #(#attrs)*
            #signature { #body }
        }
    }
}

/// Validate the type of a `#[extension_type]`-marked argument, emitting a clear `compile_error!` for
/// the common misuses instead of letting them surface as obscure trait/type errors deeper in codegen.
///
/// The argument must be a struct or enum with exactly one generic backend parameter (`MyType<Self>`),
/// since the shared IR rewrites that single parameter to the backend when unwrapping.
/// Marking a bare tensor, a non-path type, or a multi-parameter type is rejected here.
fn validate_extension_ty(ty: &Type) -> syn::Result<()> {
    if TensorKind::from_type(ty).is_some() {
        return Err(syn::Error::new_spanned(
            ty,
            "`#[extension_type]` marks a struct or enum of tensor primitives, not a tensor argument. \
             Remove the attribute to pass a plain tensor.",
        ));
    }

    let Type::Path(tp) = ty else {
        return Err(syn::Error::new_spanned(
            ty,
            "`#[extension_type]` requires a struct or enum type with a single generic backend \
             parameter, e.g. `MyType<Self>`.",
        ));
    };

    let last = tp.path.segments.last().ok_or_else(|| {
        syn::Error::new_spanned(
            ty,
            "`#[extension_type]` type must be a named struct or enum",
        )
    })?;
    let PathArguments::AngleBracketed(arguments) = &last.arguments else {
        return Err(syn::Error::new_spanned(
            ty,
            "`#[extension_type]` type must have exactly one generic backend parameter, e.g. \
             `MyType<Self>`.",
        ));
    };
    let valid_backend_argument = matches!(
        arguments.args.first(),
        Some(GenericArgument::Type(Type::Path(path)))
            if arguments.args.len() == 1 && path.path.is_ident("Self")
    );
    if !valid_backend_argument {
        return Err(syn::Error::new_spanned(
            ty,
            "`#[extension_type]` requires exactly one generic argument and it must be `Self`, e.g. `MyType<Self>`.",
        ));
    }

    Ok(())
}

/// Generate the dispatch body for an operation with tensor-bearing inputs. General "peek then
/// unwrap" path: handles any mix of bare tensors and extension structs/enums (including several),
/// for both concrete backends and (when `Autodiff` is listed) the autodiff wrapper.
///
/// Backend identity is nested inside `DispatchTensorKind::Autodiff` for tracked float tensors, and
/// extension structs can't be destructured in a dispatch-kind match. Therefore we:
/// 1. Find a routing [`DispatchTensor`] (a bare tensor itself, or a struct's via
///    `ExtensionType::routing_tensor`) and fold its `.kind`, autodiff context, and float status into
///    small values. This drops all borrows before any input is moved.
/// 2. In the matched context/backend arm, unwrap every input and re-wrap the output.
///
/// For an autodiff arm the target backend is `Autodiff<B>`: float tensors/fields unwrap through the
/// `Autodiff(Box(#b(BackendTensor::Autodiff(_))))` nesting into `FloatTensor<Autodiff<B>>`, while
/// int/bool/quantized ones stay plain (autodiff only tracks floats). The op's own
/// `impl ... for Autodiff<B, C>` still hand-writes the backward pass exactly as for a bare-tensor
/// autodiff op; the macro only routes and re-wraps.
fn gen_tensor_input_dispatch_body(ir: &Extension, op: &Operation) -> TokenStream2 {
    let name = &op.name;
    let has_ad = ir.backends.autodiff.0;
    // cfg gating the `Autodiff` entry itself (e.g. `Autodiff: cfg(feature = "autodiff")`). Every
    // generated autodiff arm must carry it, mirroring the pure-tensor path, so the arms vanish when
    // autodiff is compiled out. Otherwise their `DispatchTensorKind::Autodiff` / `Autodiff<B>`
    // references (themselves feature-gated) would fail to compile.
    let ad_cfg = ir.backends.autodiff.1.as_ref().map(|meta| quote!(#meta));
    let ad_cfg_attr = ad_cfg.as_ref().map(|cfg| quote!(#[#cfg]));

    // Backend selection and float detection share the same routing-tensor walk as
    // `#[backend_dispatch]`: optional/tensor-less extension values fall through, floats are
    // preferred, and any tensor is a final fallback.
    let paths = routing::RoutingPaths::extension();
    let routing_tensor_init = routing::operation_routing_tensor(
        op,
        &paths,
        "backend extension op received no tensor input to select a backend from (e.g. an enum input on a tensor-less variant with no other tensor input)",
    );

    // The routing tensor selects the concrete backend and autodiff context. Whether the active
    // inputs contain a float decides if an enabled context needs the autodiff implementation.
    let kind_arms = routing::backend_kind_arms(
        &paths,
        &ir.backends.concrete,
        None,
        has_ad.then(|| routing::AutodiffKindMatch {
            cfg_attr: ad_cfg_attr.clone(),
            inner_kind: quote!(__inner.as_ref()),
            nested_autodiff: None,
            fallback: quote!(usize::MAX),
        }),
        |index, _, _| quote!(#index),
    );
    let concrete_tag_arms = kind_arms.concrete;
    let ad_tag_arm = kind_arms.autodiff;

    let dispatch_arms = ir
        .backends
        .concrete
        .iter()
        .enumerate()
        .map(|(i, backend)| gen_tensor_input_backend_arm(ir, op, backend, i, &ad_cfg_attr));

    quote! {
        // Compute the autodiff context and backend tag in a scoped block so the routing tensor's
        // borrow of the inputs ends before the dispatch arms below move them.
        let (__burn_backend_tag, __ad_ctx, __has_float_input): (
            usize,
            burn::backend::DispatchAutodiffContext,
            bool,
        ) = {
            #routing_tensor_init
            let __ad_ctx = __routing_tensor.autodiff;
            let __backend_tag = match &__routing_tensor.kind {
                #ad_tag_arm
                #( #concrete_tag_arms )*
                #[allow(unreachable_patterns)]
                _ => usize::MAX,
            };
            (
                __backend_tag,
                __ad_ctx,
                __routing_tensor_is_float,
            )
        };
        match (__burn_backend_tag, __has_float_input) {
            #( #dispatch_arms )*
            _ => unimplemented!("Backend not supported for custom op `{}`", stringify!(#name)),
        }
    }
}

/// Generate all context routes for one selected concrete backend.
fn gen_tensor_input_backend_arm(
    ir: &Extension,
    op: &Operation,
    backend: &Backend,
    i: usize,
    ad_cfg_attr: &Option<TokenStream2>,
) -> TokenStream2 {
    let cfg_attr = backend.cfg_attr.clone();
    let routes = routing::extension_backend_routes(
        op,
        &backend.ident,
        ir.backends.autodiff.0,
        ad_cfg_attr.as_ref(),
        &op.name,
    );
    quote! {
        #cfg_attr
        (#i, __has_float_input) => { #routes },
    }
}

/// Generate the body that unwraps the dispatch tensors, calls the backend's trait impl and wraps the
/// result back into a [`DispatchTensor`]. Used by the concrete dispatch arms and the direct
/// no-tensor-input dispatch path.
fn gen_backend_call(
    _ir: &Extension,
    op: &Operation,
    backend: &Backend,
    route: OutputRoute,
) -> TokenStream2 {
    let b_ident = &backend.ident;
    let paths = routing::RoutingPaths::extension();
    let invoke = routing::invoke(op, &paths);
    let wrap_out = routing::wrap_output(&op.output, &paths, b_ident, route, quote!(__output));

    quote! {
        type _B = burn::backend::#b_ident;
        #invoke
        #wrap_out
    }
}
