//! Implementation of `#[backend_dispatch]`.
//!
//! This frontend validates impl blocks and lowers their methods to the operation IR shared with
//! backend extensions. Device-selected creation remains specialized here because it has no tensor
//! input to route from.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{FnArg, GenericArgument, ImplItem, ItemImpl, Pat, PathArguments, Type};

use crate::{
    BACKENDS,
    ir::{Backend, InputKind, Invocation, Operation, OperationInput, OperationOutput, TensorKind},
    routing::{self, OutputRoute},
};

pub(crate) fn expand(attr: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    if !attr.is_empty() {
        return Err(syn::Error::new_spanned(
            attr,
            "`backend_dispatch` takes no arguments",
        ));
    }

    let mut item: ItemImpl = syn::parse2(item)?;
    let inherent = validate_impl(&item)?;

    for impl_item in &mut item.items {
        let ImplItem::Fn(method) = impl_item else {
            continue;
        };

        let mut skip = false;
        let mut error = None;
        method.attrs.retain(|attr| {
            if !attr.path().is_ident("backend_dispatch") {
                return true;
            }
            let result = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("skip") {
                    skip = true;
                    Ok(())
                } else {
                    Err(meta.error("expected `skip`"))
                }
            });
            if let Err(err) = result {
                error = Some(err);
            }
            false
        });
        if let Some(error) = error {
            return Err(error);
        }
        if skip {
            continue;
        }

        if inherent {
            if !matches!(method.vis, syn::Visibility::Inherited) {
                return Err(syn::Error::new_spanned(
                    &method.vis,
                    "dispatched inherent helpers must be private",
                ));
            }
            method.attrs.push(syn::parse_quote!(#[inline(always)]));
            method.attrs.push(syn::parse_quote!(
                #[allow(clippy::too_many_arguments, clippy::type_complexity)]
            ));
        }

        let mut inputs = Vec::new();
        let mut device = None;
        for arg in &method.sig.inputs {
            let FnArg::Typed(arg) = arg else {
                return Err(syn::Error::new_spanned(
                    arg,
                    "receiver arguments aren't supported",
                ));
            };
            let Pat::Ident(pat) = arg.pat.as_ref() else {
                return Err(syn::Error::new_spanned(
                    &arg.pat,
                    "tensor arguments must use identifier patterns",
                ));
            };
            if is_dispatch_device(&arg.ty) {
                device = Some(pat.ident.clone());
            } else if let Some(input) = OperationInput::dispatch(pat.ident.clone(), &arg.ty) {
                inputs.push(input);
            }
        }

        if inputs.is_empty() && device.is_none() {
            return Err(syn::Error::new_spanned(
                &method.sig,
                "a dispatched method needs a tensor input or `DispatchDevice`; use `#[backend_dispatch(skip)]` for handwritten routing",
            ));
        }

        let output = parse_output(&method.sig.output)?;
        let body = &method.block;
        let tokens = if inputs.is_empty() {
            expand_creation(device.as_ref().unwrap(), &output, body)
        } else {
            expand_tensor_dispatch(&Operation {
                name: method.sig.ident.clone(),
                inputs,
                output,
                invocation: Invocation::Body(body.clone()),
            })
        };
        method.block = syn::parse2(quote!({ #tokens }))?;
    }

    Ok(quote!(#item))
}

fn expand_tensor_dispatch(operation: &Operation) -> TokenStream {
    match routing::backend_selection(operation) {
        routing::BackendSelection::FromInput(source) => expand_from_input(operation, source),
        routing::BackendSelection::FromCandidates => expand_from_candidates(operation),
    }
}

fn expand_from_input(operation: &Operation, source: &OperationInput) -> TokenStream {
    let name = &source.name;
    let InputKind::Tensor { borrowed, .. } = source.kind else {
        unreachable!("direct backend source must be a tensor input")
    };
    let source_kind = if borrowed {
        quote!(&#name.kind)
    } else {
        quote!(#name.kind)
    };
    let inner_kind = if borrowed {
        quote!(__inner.as_ref())
    } else {
        quote!(*__inner)
    };
    expand_backend_match(
        operation,
        Some(source),
        quote! {
            let __ad_ctx = #name.autodiff;
            let __burn_selected_context = __ad_ctx;
        },
        source_kind,
        inner_kind,
    )
}

fn expand_from_candidates(operation: &Operation) -> TokenStream {
    let paths = routing::RoutingPaths::dispatch();
    let routing_tensor_init = routing::operation_routing_tensor(
        operation,
        &paths,
        "dispatched operation received no tensor input to select a backend from",
    );
    expand_backend_match(
        operation,
        None,
        quote! {
            #routing_tensor_init
            let __ad_ctx = __routing_tensor.autodiff;
        },
        quote!(&__routing_tensor.kind),
        quote!(__inner.as_ref()),
    )
}

fn expand_backend_match(
    operation: &Operation,
    source: Option<&OperationInput>,
    selection: TokenStream,
    source_kind: TokenStream,
    inner_kind: TokenStream,
) -> TokenStream {
    let float_inputs = float_input_presence(operation, source);
    let backends = dispatch_backends();
    let source_is_non_float = source.is_some_and(|source| {
        matches!(
            source.kind,
            InputKind::Tensor { kind, .. } if kind != TensorKind::Float
        )
    });
    let paths = routing::RoutingPaths::dispatch();
    let selected =
        source.map(|_| syn::Ident::new("__burn_selected", proc_macro2::Span::call_site()));
    let autodiff_match = (!source_is_non_float).then(|| routing::AutodiffKindMatch {
        cfg_attr: Some(quote!(#[cfg(feature = "autodiff")])),
        inner_kind,
        nested_autodiff: Some(quote! {
            panic!("Autodiff should not wrap an autodiff tensor.")
        }),
        fallback: quote!(panic!("unsupported dispatch backend")),
    });
    let arms = routing::backend_kind_arms(
        &paths,
        &backends,
        selected.as_ref(),
        autodiff_match,
        |_, backend, autodiff| {
            backend_route_body(backend, operation, &float_inputs, autodiff, source)
        },
    );
    let direct_arms = arms.concrete;
    let autodiff_route = if source_is_non_float {
        quote! {
            #[cfg(feature = "autodiff")]
            crate::DispatchTensorKind::Autodiff(_) => {
                panic!("only float tensors may use an autodiff primitive")
            }
        }
    } else {
        arms.autodiff.expect("autodiff backend arms should exist")
    };

    quote! {
        #selection
        match #source_kind {
            #(#direct_arms)*
            #autodiff_route
            #[allow(unreachable_patterns)]
            _ => panic!("unsupported dispatch backend"),
        }
    }
}

fn float_input_presence(
    operation: &Operation,
    source: Option<&OperationInput>,
) -> routing::FloatInputPresence {
    if operation.inputs.iter().any(|input| {
        matches!(
            input.kind,
            InputKind::Tensor {
                kind: TensorKind::Float,
                ..
            } | InputKind::QuantizationParameters
        )
    }) {
        return routing::FloatInputPresence::Always;
    }

    let conditional: Vec<_> = operation
        .inputs
        .iter()
        .filter_map(|input| {
            let name = &input.name;
            match input.kind {
                InputKind::OptionTensor(TensorKind::Float) => Some(quote!(#name.is_some())),
                InputKind::VecTensor(TensorKind::Float) => Some(quote!(!#name.is_empty())),
                _ => None,
            }
        })
        .collect();
    if conditional.is_empty() {
        routing::FloatInputPresence::Never
    } else if source.is_none() {
        routing::FloatInputPresence::Runtime(quote!(__routing_tensor_is_float))
    } else {
        routing::FloatInputPresence::Runtime(quote!(false #(|| #conditional)*))
    }
}

fn dispatch_backends() -> Vec<Backend> {
    BACKENDS
        .iter()
        .map(|backend| {
            let ident = syn::Ident::new(backend.name, proc_macro2::Span::call_site());
            let cfg: TokenStream = backend.cfg.parse().expect("valid backend cfg");
            Backend {
                ident,
                cfg_attr: Some(quote!(#[cfg(#cfg)])),
            }
        })
        .collect()
}

fn backend_route_body(
    backend: &Backend,
    operation: &Operation,
    float_inputs: &routing::FloatInputPresence,
    routing_tensor_is_autodiff: bool,
    source: Option<&OperationInput>,
) -> TokenStream {
    let ident = &backend.ident;
    let autodiff_attr = quote!(#[cfg(feature = "autodiff")]);
    let no_autodiff_attr = quote!(#[cfg(not(feature = "autodiff"))]);
    routing::dispatch_backend_routes(
        operation,
        ident,
        float_inputs,
        routing_tensor_is_autodiff,
        source,
        &autodiff_attr,
        &no_autodiff_attr,
    )
}

fn parse_output(output: &syn::ReturnType) -> syn::Result<OperationOutput> {
    let syn::ReturnType::Type(_, ty) = output else {
        return Ok(OperationOutput::Plain);
    };
    OperationOutput::dispatch(ty)
}

fn is_dispatch_device(ty: &Type) -> bool {
    match ty {
        Type::Reference(reference) => is_dispatch_device(&reference.elem),
        Type::Path(path) => path
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "DispatchDevice"),
        _ => false,
    }
}

fn expand_creation(
    device: &syn::Ident,
    output: &OperationOutput,
    body: &syn::Block,
) -> TokenStream {
    let direct_arms = BACKENDS
        .iter()
        .map(|backend| creation_arm(backend, output, body, false));
    let autodiff_arms = BACKENDS
        .iter()
        .map(|backend| creation_arm(backend, output, body, true));
    quote! {
        match #device {
            #(#direct_arms)*
            #[cfg(feature = "autodiff")]
            crate::DispatchDevice::Autodiff(__device) => match __device.inner.as_ref() {
                #(#autodiff_arms)*
                crate::DispatchDevice::Autodiff(_) => {
                    panic!("Autodiff should not wrap an autodiff device.")
                }
                #[allow(unreachable_patterns)]
                __other => panic!("unsupported dispatch device: {__other:?}"),
            },
            #[allow(unreachable_patterns)]
            __other => panic!("unsupported dispatch device: {__other:?}"),
        }
    }
}

fn creation_arm(
    backend: &crate::BackendSpec,
    output: &OperationOutput,
    body: &syn::Block,
    autodiff_device: bool,
) -> TokenStream {
    let ident = syn::Ident::new(backend.name, proc_macro2::Span::call_site());
    let cfg: TokenStream = backend.cfg.parse().expect("valid backend cfg");
    if !autodiff_device {
        let wrapped = routing::wrap_output(
            output,
            &routing::RoutingPaths::dispatch(),
            &ident,
            OutputRoute::ConcreteDisabled,
            quote!(__output),
        );
        quote! {
            #[cfg(#cfg)]
            crate::DispatchDevice::#ident(device) => {
                type B = crate::backends::#ident;
                let __ad_ctx = crate::DispatchAutodiffContext::Disabled;
                let __output = #body;
                #wrapped
            },
        }
    } else if output.contains_float() {
        let wrapped = routing::wrap_output(
            output,
            &routing::RoutingPaths::dispatch(),
            &ident,
            OutputRoute::Autodiff,
            quote!(__output),
        );
        let alias = syn::Ident::new("B", proc_macro2::Span::call_site());
        let call = routing::with_autodiff_backend(
            &routing::RoutingPaths::dispatch(),
            &ident,
            quote!(__strategy),
            &alias,
            quote! {
                let __ad_ctx = crate::DispatchAutodiffContext::Enabled(__strategy);
                let __output = #body;
                #wrapped
            },
        );
        quote! {
            #[cfg(#cfg)]
            crate::DispatchDevice::#ident(device) => {
                let __strategy = __device.checkpointing;
                #call
            },
        }
    } else {
        let wrapped = routing::wrap_output(
            output,
            &routing::RoutingPaths::dispatch(),
            &ident,
            OutputRoute::ConcreteEnabled,
            quote!(__output),
        );
        quote! {
            #[cfg(#cfg)]
            crate::DispatchDevice::#ident(device) => {
                type B = crate::backends::#ident;
                let __strategy = __device.checkpointing;
                let __ad_ctx = crate::DispatchAutodiffContext::Enabled(__strategy);
                let __output = #body;
                #wrapped
            },
        }
    }
}

fn validate_impl(item: &ItemImpl) -> syn::Result<bool> {
    let Type::Path(self_ty) = item.self_ty.as_ref() else {
        return Err(syn::Error::new_spanned(
            &item.self_ty,
            "expected `Dispatch`",
        ));
    };
    if self_ty
        .path
        .segments
        .last()
        .is_none_or(|segment| segment.ident != "Dispatch")
    {
        return Err(syn::Error::new_spanned(
            &item.self_ty,
            "expected `Dispatch`",
        ));
    }

    let Some((_, trait_path, _)) = &item.trait_ else {
        return Ok(true);
    };
    let has_self = trait_path.segments.last().is_some_and(|segment| {
        let PathArguments::AngleBracketed(args) = &segment.arguments else {
            return false;
        };
        args.args.iter().any(
            |arg| matches!(arg, GenericArgument::Type(Type::Path(path)) if path.path.is_ident("Self")),
        )
    });
    if !has_self {
        return Err(syn::Error::new_spanned(
            trait_path,
            "expected `impl Trait<Self> for Dispatch`",
        ));
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    //! Frontend tests for `#[backend_dispatch]`.
    //!
    //! Generated code is compiled and executed by Burn's dispatch and backend-extension integration
    //! tests. Unit tests here cover frontend attributes and diagnostics without matching rendered
    //! token streams.

    use super::*;
    use syn::{Expr, ImplItemFn, Stmt};

    fn expand_impl(input: TokenStream) -> ItemImpl {
        syn::parse2(expand(TokenStream::new(), input).expect("dispatch expansion should succeed"))
            .expect("expanded dispatch should remain a valid impl")
    }

    fn method<'a>(item: &'a ItemImpl, name: &str) -> &'a ImplItemFn {
        item.items
            .iter()
            .find_map(|item| match item {
                ImplItem::Fn(method) if method.sig.ident == name => Some(method),
                _ => None,
            })
            .expect("expanded method should exist")
    }

    #[test]
    fn private_inherent_helpers_are_marked_inline() {
        let input = quote! {
            impl Dispatch {
                fn helper(x: FloatTensor<Self>) -> FloatTensor<Self> {
                    B::helper(x)
                }
            }
        };
        let expanded = expand_impl(input);
        let helper = method(&expanded, "helper");

        assert!(
            helper
                .attrs
                .iter()
                .any(|attr| attr.path().is_ident("inline"))
        );
    }

    #[test]
    fn skip_removes_the_helper_attribute_without_rewriting_the_body() {
        let input = quote! {
            impl Ops<Self> for Dispatch {
                #[backend_dispatch(skip)]
                fn empty(device: &Device<Self>) -> FloatTensor<Self> { special(device) }
            }
        };
        let expanded = expand_impl(input);
        let empty = method(&expanded, "empty");

        assert!(
            empty
                .attrs
                .iter()
                .all(|attr| !attr.path().is_ident("backend_dispatch"))
        );
        let [Stmt::Expr(Expr::Call(call), None)] = empty.block.stmts.as_slice() else {
            panic!("the handwritten call should be preserved")
        };
        let Expr::Path(function) = call.func.as_ref() else {
            panic!("the handwritten function path should be preserved")
        };
        assert!(function.path.is_ident("special"));
        let [Expr::Path(argument)] = call.args.iter().collect::<Vec<_>>().as_slice() else {
            panic!("the handwritten argument should be preserved")
        };
        assert!(argument.path.is_ident("device"));
    }

    #[test]
    fn rejects_named_outputs() {
        let input = quote! {
            impl Ops<Self> for Dispatch {
                fn module(x: FloatTensor<Self>) -> ModuleOutput<Self> { B::module(x) }
            }
        };
        assert!(expand(TokenStream::new(), input).is_err());
    }

    #[test]
    fn rejects_arguments_on_the_outer_attribute() {
        let input = quote! {
            impl Ops<Self> for Dispatch {
                fn op(x: IntTensor<Self>) -> IntTensor<Self> { B::op(x) }
            }
        };
        assert!(expand(quote!(unexpected), input).is_err());
    }
}
