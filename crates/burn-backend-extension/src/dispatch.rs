//! Implementation of `#[backend_dispatch]`.
//!
//! This is deliberately separate from the extension lowering pipeline: extension
//! traits are a public API and their generated representation is compatibility
//! sensitive, while this attribute is an implementation convenience used by
//! `burn-dispatch` itself.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{FnArg, GenericArgument, ImplItem, ItemImpl, Pat, PathArguments, Type};

use crate::TensorKind;

pub(crate) fn expand(attr: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    if !attr.is_empty() {
        return Err(syn::Error::new_spanned(
            attr,
            "`backend_dispatch` takes no arguments",
        ));
    }

    let mut item: ItemImpl = syn::parse2(item)?;
    validate_impl(&item)?;

    for impl_item in &mut item.items {
        let ImplItem::Fn(method) = impl_item else {
            continue;
        };

        let mut skip = false;
        let mut base = false;
        method.attrs.retain(|attr| {
            if !attr.path().is_ident("backend_dispatch") {
                return true;
            }
            let result = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("skip") {
                    skip = true;
                    Ok(())
                } else if meta.path.is_ident("base") {
                    base = true;
                    Ok(())
                } else {
                    Err(meta.error("expected `skip` or `base`"))
                }
            });
            if let Err(err) = result {
                // Retaining makes rustc produce a useful error as well; the normal
                // successful forms are consumed here.
                let _ = err;
                return true;
            }
            false
        });
        if skip {
            continue;
        }

        let mut required = Vec::new();
        let mut optional = Vec::new();
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
            if let Some(kind) = TensorKind::from_type(&arg.ty) {
                required.push((pat.ident.clone(), kind_ident(kind)));
            } else if let Some(inner) = container_inner(&arg.ty, "Option") {
                if let Some(kind) = TensorKind::from_type(inner) {
                    optional.push((pat.ident.clone(), kind_ident(kind)));
                }
            }
        }

        if required.is_empty() {
            return Err(syn::Error::new_spanned(
                &method.sig,
                "a dispatched method needs a required tensor input; use `#[backend_dispatch(skip)]` for device/creation routing",
            ));
        }

        let output = match &method.sig.output {
            syn::ReturnType::Type(_, ty) => TensorKind::from_type(ty),
            syn::ReturnType::Default => None,
        }
        .ok_or_else(|| {
            syn::Error::new_spanned(
                &method.sig.output,
                "this output isn't directly mappable yet; use `#[backend_dispatch(skip)]`",
            )
        })?;

        let body = &method.block;
        let first = &required[0].0;
        let includes = required.iter().skip(1).map(|(name, _)| {
            quote! {
                __context.include(&#name);
            }
        });
        let opt_includes = optional.iter().map(|(name, _)| {
            quote! {
                __context.include_optional(#name.as_ref());
            }
        });
        let arms = backends()
            .into_iter()
            .map(|backend| concrete_arm(backend, &required, &optional, output, body, base));
        let tokens = quote! {
            let mut __context = crate::tensor::DispatchContext::new(&#first, #base);
            #(#includes)*
            #(#opt_includes)*
            match (__context.backend(), __context.uses_autodiff()) {
                #(#arms)*
                #[allow(unreachable_patterns)]
                _ => unreachable!("backend dispatch context is inconsistent"),
            }
        };
        method.block = syn::parse2(quote!({ #tokens }))?;
    }

    Ok(quote!(#item))
}

#[derive(Clone)]
struct ConcreteBackend {
    ident: syn::Ident,
    cfg: TokenStream,
}

fn backends() -> Vec<ConcreteBackend> {
    [
        ("Cpu", quote!(feature = "cpu")),
        ("Cuda", quote!(feature = "cuda")),
        ("Metal", quote!(feature = "metal")),
        ("Rocm", quote!(feature = "rocm")),
        ("Vulkan", quote!(feature = "vulkan")),
        ("Wgpu", quote!(feature = "wgpu")),
        ("WebGpu", quote!(feature = "webgpu")),
        ("Flex", quote!(any(feature = "flex", default_backend))),
        ("NdArray", quote!(feature = "ndarray")),
        ("LibTorch", quote!(feature = "tch")),
        ("Remote", quote!(feature = "remote")),
    ]
    .into_iter()
    .map(|(name, cfg)| ConcreteBackend {
        ident: syn::Ident::new(name, proc_macro2::Span::call_site()),
        cfg,
    })
    .collect()
}

fn concrete_arm(
    backend: ConcreteBackend,
    required: &[(syn::Ident, syn::Ident)],
    optional: &[(syn::Ident, syn::Ident)],
    output: TensorKind,
    body: &syn::Block,
    base: bool,
) -> TokenStream {
    let ConcreteBackend { ident, cfg } = backend;
    let normal_inputs = required
        .iter()
        .map(|(name, kind)| extract_normal(name, kind, &ident));
    let normal_options = optional
        .iter()
        .map(|(name, kind)| extract_normal_option(name, kind, &ident));
    let normal_wrap = wrap_output(output, &ident, false);

    let autodiff_inputs = required
        .iter()
        .map(|(name, kind)| extract_autodiff(name, kind, &ident));
    let autodiff_options = optional
        .iter()
        .map(|(name, kind)| extract_autodiff_option(name, kind, &ident));
    let autodiff_wrap = wrap_output(output, &ident, true);
    let autodiff_arm = if base {
        TokenStream::new()
    } else {
        quote! {
            #[cfg(all(feature = "autodiff", #cfg))]
            (crate::tensor::DispatchBackendId::#ident, true) => {
                with_autodiff_backend!(#ident, __context.checkpointing(), |B| {
                    #(#autodiff_inputs)*
                    #(#autodiff_options)*
                    let __output = #body;
                    #autodiff_wrap
                })
            },
        }
    };

    quote! {
        #[cfg(#cfg)]
        (crate::tensor::DispatchBackendId::#ident, false) => {
            type B = crate::backends::#ident;
            #(#normal_inputs)*
            #(#normal_options)*
            let __output = #body;
            #normal_wrap
        },
        #autodiff_arm
    }
}

fn extract_normal(name: &syn::Ident, kind: &syn::Ident, backend: &syn::Ident) -> TokenStream {
    quote! {
        let #name = match #name.kind {
            crate::DispatchTensorKind::#backend(inner) => inner.#kind(),
            _ => panic!("input tensor `{}` is on the wrong backend", stringify!(#name)),
        };
    }
}

fn extract_normal_option(
    name: &syn::Ident,
    kind: &syn::Ident,
    backend: &syn::Ident,
) -> TokenStream {
    quote! {
        let #name = #name.map(|tensor| match tensor.kind {
            crate::DispatchTensorKind::#backend(inner) => inner.#kind(),
            _ => panic!("optional tensor `{}` is on the wrong backend", stringify!(#name)),
        });
    }
}

fn extract_autodiff(name: &syn::Ident, kind: &syn::Ident, backend: &syn::Ident) -> TokenStream {
    let tracked = if kind == "float" {
        quote!(inner.autodiff())
    } else {
        quote!(inner.#kind())
    };
    let associated = if kind == "float" {
        quote!(panic!(
            "an autodiff operation received an untracked float tensor"
        ))
    } else {
        quote!(inner.#kind())
    };
    quote! {
        let #name = match #name.kind {
            crate::DispatchTensorKind::Autodiff(inner) => match *inner {
                crate::DispatchTensorKind::#backend(inner) => #tracked,
                _ => panic!("input tensor `{}` is on the wrong backend", stringify!(#name)),
            },
            crate::DispatchTensorKind::#backend(inner) => #associated,
            _ => panic!("input tensor `{}` is on the wrong backend", stringify!(#name)),
        };
    }
}

fn extract_autodiff_option(
    name: &syn::Ident,
    kind: &syn::Ident,
    backend: &syn::Ident,
) -> TokenStream {
    let inner = extract_autodiff(&syn::Ident::new("tensor", name.span()), kind, backend);
    quote! { let #name = #name.map(|tensor| { #inner tensor }); }
}

fn wrap_output(kind: TensorKind, backend: &syn::Ident, autodiff: bool) -> TokenStream {
    let variant = kind.variant();
    if autodiff && kind == TensorKind::Float {
        quote! {
            crate::DispatchTensor {
                kind: crate::DispatchTensorKind::Autodiff(alloc::boxed::Box::new(
                    crate::DispatchTensorKind::#backend(crate::BackendTensor::Autodiff(__output))
                )),
                checkpointing: __context.checkpointing(),
            }
        }
    } else {
        quote! {
            crate::DispatchTensor {
                kind: crate::DispatchTensorKind::#backend(crate::BackendTensor::#variant(__output)),
                checkpointing: __context.checkpointing(),
            }
        }
    }
}

fn validate_impl(item: &ItemImpl) -> syn::Result<()> {
    let Some((_, trait_path, _)) = &item.trait_ else {
        return Err(syn::Error::new_spanned(item, "expected a trait impl"));
    };
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
        .is_none_or(|s| s.ident != "Dispatch")
    {
        return Err(syn::Error::new_spanned(
            &item.self_ty,
            "expected an impl for `Dispatch`",
        ));
    }
    let has_self = trait_path.segments.last().is_some_and(|segment| {
        let PathArguments::AngleBracketed(args) = &segment.arguments else {
            return false;
        };
        args.args.iter().any(
            |arg| matches!(arg, GenericArgument::Type(Type::Path(p)) if p.path.is_ident("Self")),
        )
    });
    if !has_self {
        return Err(syn::Error::new_spanned(
            trait_path,
            "expected `impl Trait<Self> for Dispatch`",
        ));
    }
    Ok(())
}

fn container_inner<'a>(ty: &'a Type, name: &str) -> Option<&'a Type> {
    let Type::Path(path) = ty else { return None };
    let segment = path.path.segments.last()?;
    if segment.ident != name {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    match args.args.first()? {
        GenericArgument::Type(ty) => Some(ty),
        _ => None,
    }
}

fn kind_ident(kind: TensorKind) -> syn::Ident {
    syn::Ident::new(
        &format!("{kind:?}").to_lowercase(),
        proc_macro2::Span::call_site(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowers_plain_and_optional_tensor_inputs() {
        let input = quote! {
            impl Ops<Self> for Dispatch {
                async fn op(x: FloatTensor<Self>, mask: Option<BoolTensor<Self>>, n: usize)
                    -> IntTensor<Self>
                {
                    B::op(x, mask, n).await
                }
            }
        };
        let output = expand(TokenStream::new(), input).unwrap().to_string();
        assert!(!output.contains("multi_op !"));
        assert!(output.contains("DispatchContext :: new"));
        assert!(output.contains("mask . map"));
        assert!(output.contains("BackendTensor :: Int"));
        assert!(output.contains(". await"));
    }

    #[test]
    fn skip_preserves_special_body_and_consumes_attribute() {
        let input = quote! {
            impl Ops<Self> for Dispatch {
                #[backend_dispatch(skip)]
                fn empty(device: &Device<Self>) -> FloatTensor<Self> { special(device) }
            }
        };
        let output = expand(TokenStream::new(), input).unwrap().to_string();
        assert!(!output.contains("backend_dispatch"));
        assert!(output.contains("special (device)"));
    }

    #[test]
    fn base_uses_non_autodiff_dispatch() {
        let input = quote! {
            impl Ops<Self> for Dispatch {
                #[backend_dispatch(base)]
                fn op(x: IntTensor<Self>) -> BoolTensor<Self> { B::op(x) }
            }
        };
        let output = expand(TokenStream::new(), input).unwrap().to_string();
        assert!(!output.contains("unary_op !"));
        assert!(output.contains("DispatchContext :: new (& x , true)"));
        assert!(output.contains("BackendTensor :: Bool"));
    }

    #[test]
    fn rejects_creation_without_skip() {
        let input = quote! {
            impl Ops<Self> for Dispatch {
                fn empty(n: usize) -> FloatTensor<Self> { B::empty(n) }
            }
        };
        assert!(expand(TokenStream::new(), input).is_err());
    }
}
