use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use syn::{Data, DeriveInput, Fields, GenericArgument, PathArguments, Type};

/// How a state field is serialized.
enum FieldKind {
    /// A `Tensor<D>` leaf.
    Tensor,
    /// An `Option<Tensor<D>>` leaf.
    OptionTensor,
    /// A `Vec<Tensor<D>>` field.
    VecTensor,
    /// A scalar leaf serialized via `burn_pack::Scalar`'s `From`/`TryFrom` conversions.
    Scalar,
    /// An `Option<scalar>` leaf (carrying the inner scalar type).
    OptionScalar(Type),
    /// A nested [`RecordState`] field.
    Nested,
    /// An `Option<Nested>` field (carrying the inner nested type).
    OptionNested(Type),
}

/// Returns the identifier of the last path segment of a type, if any (`Tensor<D>` -> `Tensor`).
fn head_ident(ty: &Type) -> Option<String> {
    match ty {
        Type::Path(path) => path.path.segments.last().map(|s| s.ident.to_string()),
        _ => None,
    }
}

/// Returns the single generic type argument of a type (the `T` in `Option<T>` / `Vec<T>`).
fn single_generic_arg(ty: &Type) -> Option<Type> {
    let Type::Path(path) = ty else { return None };
    let segment = path.path.segments.last()?;
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    args.args.iter().find_map(|arg| match arg {
        GenericArgument::Type(inner) => Some(inner.clone()),
        _ => None,
    })
}

fn is_scalar(ident: &str) -> bool {
    // Only the widths `burn_pack::Scalar` has `From`/`TryFrom` conversions for. `i128`/`u128` are
    // intentionally excluded — `Scalar` tops out at 64-bit, so a 128-bit field is treated as a
    // nested `RecordState` and fails with a clear trait-bound error rather than a silent truncation.
    matches!(
        ident,
        "usize"
            | "isize"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "i8"
            | "i16"
            | "i32"
            | "i64"
            | "f32"
            | "f64"
            | "bool"
    )
}

fn classify(ty: &Type) -> Result<FieldKind, syn::Error> {
    Ok(match head_ident(ty).as_deref() {
        Some("Tensor") => FieldKind::Tensor,
        Some("Vec") => match single_generic_arg(ty) {
            Some(inner) if head_ident(&inner).as_deref() == Some("Tensor") => FieldKind::VecTensor,
            _ => {
                return Err(syn::Error::new_spanned(
                    ty,
                    "RecordState only supports `Vec<Tensor<D>>` for sequence fields.",
                ));
            }
        },
        Some("Option") => match single_generic_arg(ty) {
            Some(inner) => match head_ident(&inner).as_deref() {
                Some("Tensor") => FieldKind::OptionTensor,
                Some(ident) if is_scalar(ident) => FieldKind::OptionScalar(inner),
                _ => FieldKind::OptionNested(inner),
            },
            None => FieldKind::Nested,
        },
        Some(ident) if is_scalar(ident) => FieldKind::Scalar,
        _ => FieldKind::Nested,
    })
}

pub(crate) fn derive_impl(ast: &DeriveInput) -> proc_macro::TokenStream {
    match try_derive_impl(ast) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error().into(),
    }
}

fn try_derive_impl(ast: &DeriveInput) -> Result<proc_macro::TokenStream, syn::Error> {
    let (optim, device) = record_state_paths()?;
    let name = &ast.ident;
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

    let fields = match &ast.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(named) => &named.named,
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "RecordState can only be derived for structs with named fields.",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                name,
                "RecordState can only be derived for structs.",
            ));
        }
    };

    let mut flatten = Vec::new();
    let mut unflatten = Vec::new();
    let mut ctor = Vec::new();

    for field in fields {
        let ident = field.ident.as_ref().unwrap();
        let leaf = ident.to_string();
        let ty = &field.ty;
        ctor.push(quote! { #ident });

        let (write, read) = match classify(ty)? {
            FieldKind::Tensor => (
                quote! {
                    out.push_tensor(prefix, #leaf, self.#ident.clone().into_data());
                },
                quote! {
                    let #ident = src.restore_tensor(prefix, #leaf, device)?;
                },
            ),
            FieldKind::OptionTensor => (
                quote! {
                    if let Some(value) = &self.#ident {
                        out.push_tensor(prefix, #leaf, value.clone().into_data());
                    }
                },
                quote! {
                    let #ident = src.restore_tensor(prefix, #leaf, device);
                },
            ),
            FieldKind::VecTensor => (
                quote! {
                    out.push_scalar(
                        prefix,
                        concat!(#leaf, ".len"),
                        ::core::convert::Into::into(self.#ident.len()),
                    );
                    for (__index, __value) in self.#ident.iter().enumerate() {
                        let __leaf = #optim::join_index(#leaf, __index);
                        out.push_tensor(prefix, &__leaf, __value.clone().into_data());
                    }
                },
                quote! {
                    let #ident = {
                        let __len = <usize as ::core::convert::TryFrom<_>>::try_from(
                            src.take_scalar(prefix, concat!(#leaf, ".len"))?,
                        )
                        .ok()?;
                        let mut __items = <#ty>::with_capacity(__len);
                        for __index in 0..__len {
                            let __leaf = #optim::join_index(#leaf, __index);
                            __items.push(src.restore_tensor(prefix, &__leaf, device)?);
                        }
                        __items
                    };
                },
            ),
            FieldKind::Scalar => (
                quote! {
                    out.push_scalar(prefix, #leaf, ::core::convert::Into::into(self.#ident));
                },
                quote! {
                    let #ident = <#ty as ::core::convert::TryFrom<_>>::try_from(
                        src.take_scalar(prefix, #leaf)?,
                    )
                    .ok()?;
                },
            ),
            FieldKind::OptionScalar(inner) => (
                quote! {
                    if let Some(value) = self.#ident {
                        out.push_scalar(prefix, #leaf, ::core::convert::Into::into(value));
                    }
                },
                quote! {
                    let #ident: Option<#inner> = src.take_scalar(prefix, #leaf).and_then(|__s| {
                        <#inner as ::core::convert::TryFrom<_>>::try_from(__s).ok()
                    });
                },
            ),
            FieldKind::Nested => (
                quote! {
                    {
                        let __path = #optim::join_path(prefix, #leaf);
                        #optim::RecordState::state_flatten(&self.#ident, &__path, out);
                    }
                },
                quote! {
                    let #ident = {
                        let __path = #optim::join_path(prefix, #leaf);
                        <#ty as #optim::RecordState>::state_unflatten(&__path, src, device)?
                    };
                },
            ),
            FieldKind::OptionNested(inner) => (
                quote! {
                    if let Some(value) = &self.#ident {
                        let __path = #optim::join_path(prefix, #leaf);
                        #optim::RecordState::state_flatten(value, &__path, out);
                    }
                },
                quote! {
                    let #ident = {
                        let __path = #optim::join_path(prefix, #leaf);
                        if src.has_under(&__path) {
                            <#inner as #optim::RecordState>::state_unflatten(&__path, src, device)
                        } else {
                            None
                        }
                    };
                },
            ),
        };
        flatten.push(write);
        unflatten.push(read);
    }

    Ok(quote! {
        impl #impl_generics #optim::RecordState for #name #ty_generics #where_clause {
            fn state_flatten(&self, prefix: &str, out: &mut #optim::StateSink) {
                #(#flatten)*
            }

            fn state_unflatten(
                prefix: &str,
                src: &mut #optim::StateSource,
                device: &#device,
            ) -> Option<Self> {
                #(#unflatten)*
                Some(Self { #(#ctor),* })
            }
        }
    }
    .into())
}

fn record_state_paths() -> syn::Result<(TokenStream, TokenStream)> {
    if let Some(optim) = dependency_path("burn-optim") {
        return Ok((optim, device_path()?));
    }
    if let Some(burn) = dependency_path("burn") {
        return Ok((quote!(#burn::optim), quote!(#burn::tensor::Device)));
    }
    Err(syn::Error::new(
        Span::call_site(),
        "RecordState requires a dependency on burn or burn-optim",
    ))
}

fn device_path() -> syn::Result<TokenStream> {
    for package in ["burn-core", "burn-tensor", "burn"] {
        if let Some(path) = dependency_path(package) {
            return Ok(if package == "burn-tensor" {
                quote!(#path::Device)
            } else {
                quote!(#path::tensor::Device)
            });
        }
    }
    Err(syn::Error::new(
        Span::call_site(),
        "deriving RecordState with burn-optim requires a direct dependency on burn, burn-core, or burn-tensor",
    ))
}

fn dependency_path(package: &str) -> Option<TokenStream> {
    Some(match crate_name(package).ok()? {
        FoundCrate::Itself => quote!(crate),
        FoundCrate::Name(name) => {
            let ident = Ident::new_raw(&name, Span::call_site());
            quote!(::#ident)
        }
    })
}
