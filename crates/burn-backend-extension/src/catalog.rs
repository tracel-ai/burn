//! Runtime backend catalog shared by generated and handwritten dispatch paths.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{Ident, Token};

pub(crate) struct BackendSpec {
    pub(crate) name: &'static str,
    pub(crate) cfg: &'static str,
    distributed: bool,
    unidirectional_transfer: bool,
}

pub(crate) const BACKENDS: &[BackendSpec] = &[
    BackendSpec {
        // Every cubecl runtime is this one backend; a device says which of them
        // it runs on. `cube_backend` is set when any of their features is.
        name: "Cube",
        cfg: "cube_backend",
        // One entry covers every runtime, so this claims collectives for all of them where only
        // CUDA implements them today: a collective on a wgpu or CPU device reaches cubecl's
        // `ComputeServer::all_reduce` and panics there, rather than being turned away here as it
        // was when each runtime had its own spec. That is the intended direction — the remaining
        // runtimes are meant to implement it — but until they do the panic comes from cubecl with
        // no mention of the device that caused it.
        distributed: true,
        unidirectional_transfer: false,
    },
    BackendSpec {
        name: "Flex",
        cfg: "any(feature = \"flex\", default_backend)",
        distributed: false,
        unidirectional_transfer: false,
    },
    BackendSpec {
        name: "NdArray",
        cfg: "feature = \"ndarray\"",
        distributed: false,
        unidirectional_transfer: false,
    },
    BackendSpec {
        name: "LibTorch",
        cfg: "feature = \"tch\"",
        distributed: false,
        unidirectional_transfer: false,
    },
    BackendSpec {
        name: "Remote",
        cfg: "feature = \"remote\"",
        distributed: true,
        unidirectional_transfer: false,
    },
    BackendSpec {
        name: "Capture",
        cfg: "feature = \"capture\"",
        distributed: false,
        // Capture is one-way and is handled by dedicated transfer arms.
        unidirectional_transfer: true,
    },
];

enum CatalogKind {
    Backends,
    Distributed,
    Matrix,
}

struct CatalogInput {
    kind: CatalogKind,
    callback: Ident,
    extra: TokenStream,
}

impl Parse for CatalogInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let kind: Ident = input.parse()?;
        let kind = match kind.to_string().as_str() {
            "backends" => CatalogKind::Backends,
            "distributed" => CatalogKind::Distributed,
            "matrix" => CatalogKind::Matrix,
            _ => {
                return Err(syn::Error::new_spanned(
                    kind,
                    "expected `backends`, `distributed`, or `matrix`",
                ));
            }
        };
        input.parse::<Token![,]>()?;
        let callback = input.parse()?;
        input.parse::<Token![;]>()?;
        let extra = input.parse()?;

        Ok(Self {
            kind,
            callback,
            extra,
        })
    }
}

pub(crate) fn expand(input: TokenStream) -> syn::Result<TokenStream> {
    let CatalogInput {
        kind,
        callback,
        extra,
    } = syn::parse2(input)?;

    let item = |backend: &BackendSpec| {
        let ident = format_ident!("{}", backend.name);
        let cfg: TokenStream = backend.cfg.parse().expect("valid backend cfg");
        quote! { [#ident, #cfg] }
    };
    let backends: Vec<_> = BACKENDS.iter().map(item).collect();
    let distributed: Vec<_> = BACKENDS
        .iter()
        .filter(|backend| backend.distributed)
        .map(item)
        .collect();
    let matrix: Vec<_> = BACKENDS
        .iter()
        .filter(|backend| !backend.unidirectional_transfer)
        .map(|source| {
            let source_name = source.name;
            let source = item(source);
            let destinations = BACKENDS
                .iter()
                .filter(|target| !target.unidirectional_transfer && target.name != source_name)
                .map(item);
            quote! { #source => [#(#destinations),*] }
        })
        .collect();

    Ok(match kind {
        CatalogKind::Backends => quote!(#callback! { #extra; #(#backends),* }),
        CatalogKind::Distributed => quote!(#callback! { #extra; #(#distributed),* }),
        CatalogKind::Matrix => quote!(#callback! { #extra; #(#matrix);* }),
    })
}
