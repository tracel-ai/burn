use quote::quote;
use syn::{Data, DeriveInput, Fields, GenericArgument, PathArguments, Type};

/// How an optimizer-state field is serialized.
enum FieldKind {
    /// A `Tensor<D>` leaf.
    Tensor,
    /// An `Option<Tensor<D>>` leaf (carrying the inner `Tensor<D>` type).
    OptionTensor(Type),
    /// A `Vec<Tensor<D>>` field (carrying the inner `Tensor<D>` type).
    VecTensor(Type),
    /// A scalar leaf serialized via `burn_pack::Scalar`'s `From`/`TryFrom` conversions.
    Scalar,
    /// An `Option<scalar>` leaf (carrying the inner scalar type).
    OptionScalar(Type),
    /// A nested [`OptimState`] field.
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
    matches!(
        ident,
        "usize"
            | "isize"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "u128"
            | "i8"
            | "i16"
            | "i32"
            | "i64"
            | "i128"
            | "f32"
            | "f64"
            | "bool"
    )
}

fn classify(ty: &Type) -> FieldKind {
    match head_ident(ty).as_deref() {
        Some("Tensor") => FieldKind::Tensor,
        Some("Vec") => match single_generic_arg(ty) {
            Some(inner) if head_ident(&inner).as_deref() == Some("Tensor") => {
                FieldKind::VecTensor(inner)
            }
            _ => panic!("OptimState only supports `Vec<Tensor<D>>` for sequence fields."),
        },
        Some("Option") => match single_generic_arg(ty) {
            Some(inner) => match head_ident(&inner).as_deref() {
                Some("Tensor") => FieldKind::OptionTensor(inner),
                Some(ident) if is_scalar(ident) => FieldKind::OptionScalar(inner),
                _ => FieldKind::OptionNested(inner),
            },
            None => FieldKind::Nested,
        },
        Some(ident) if is_scalar(ident) => FieldKind::Scalar,
        _ => FieldKind::Nested,
    }
}

pub(crate) fn derive_impl(ast: &DeriveInput) -> proc_macro::TokenStream {
    let name = &ast.ident;
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

    let fields = match &ast.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(named) => &named.named,
            _ => panic!("OptimState can only be derived for structs with named fields."),
        },
        _ => panic!("OptimState can only be derived for structs."),
    };

    let mut flatten = Vec::new();
    let mut unflatten = Vec::new();
    let mut ctor = Vec::new();

    for field in fields {
        let ident = field.ident.as_ref().unwrap();
        let leaf = ident.to_string();
        let ty = &field.ty;
        ctor.push(quote! { #ident });

        match classify(ty) {
            FieldKind::Tensor => {
                flatten.push(quote! {
                    out.push_tensor(prefix, #leaf, self.#ident.clone().into_data());
                });
                unflatten.push(quote! {
                    let #ident = <#ty>::from_data(src.take_tensor(prefix, #leaf)?, device);
                });
            }
            FieldKind::OptionTensor(inner) => {
                flatten.push(quote! {
                    if let Some(value) = &self.#ident {
                        out.push_tensor(prefix, #leaf, value.clone().into_data());
                    }
                });
                unflatten.push(quote! {
                    let #ident = src
                        .take_tensor(prefix, #leaf)
                        .map(|data| <#inner>::from_data(data, device));
                });
            }
            FieldKind::VecTensor(inner) => {
                flatten.push(quote! {
                    out.push_scalar(
                        prefix,
                        concat!(#leaf, ".len"),
                        burn_pack::Scalar::from(self.#ident.len()),
                    );
                    for (__index, __value) in self.#ident.iter().enumerate() {
                        let __leaf = crate::join_index(#leaf, __index);
                        out.push_tensor(prefix, &__leaf, __value.clone().into_data());
                    }
                });
                unflatten.push(quote! {
                    let #ident = {
                        let __len = <usize as ::core::convert::TryFrom<burn_pack::Scalar>>::try_from(
                            src.take_scalar(prefix, concat!(#leaf, ".len"))?,
                        )
                        .ok()?;
                        let mut __items = ::alloc::vec::Vec::with_capacity(__len);
                        for __index in 0..__len {
                            let __leaf = crate::join_index(#leaf, __index);
                            __items.push(<#inner>::from_data(
                                src.take_tensor(prefix, &__leaf)?,
                                device,
                            ));
                        }
                        __items
                    };
                });
            }
            FieldKind::Scalar => {
                flatten.push(quote! {
                    out.push_scalar(prefix, #leaf, burn_pack::Scalar::from(self.#ident));
                });
                unflatten.push(quote! {
                    let #ident = <#ty as ::core::convert::TryFrom<burn_pack::Scalar>>::try_from(
                        src.take_scalar(prefix, #leaf)?,
                    )
                    .ok()?;
                });
            }
            FieldKind::OptionScalar(inner) => {
                flatten.push(quote! {
                    if let Some(value) = self.#ident {
                        out.push_scalar(prefix, #leaf, burn_pack::Scalar::from(value));
                    }
                });
                unflatten.push(quote! {
                    let #ident: Option<#inner> = src.take_scalar(prefix, #leaf).and_then(|__s| {
                        <#inner as ::core::convert::TryFrom<burn_pack::Scalar>>::try_from(__s).ok()
                    });
                });
            }
            FieldKind::Nested => {
                flatten.push(quote! {
                    {
                        let __path = crate::join_path(prefix, #leaf);
                        crate::OptimState::state_flatten(&self.#ident, &__path, out);
                    }
                });
                unflatten.push(quote! {
                    let #ident = {
                        let __path = crate::join_path(prefix, #leaf);
                        <#ty as crate::OptimState>::state_unflatten(&__path, src, device)?
                    };
                });
            }
            FieldKind::OptionNested(inner) => {
                flatten.push(quote! {
                    if let Some(value) = &self.#ident {
                        let __path = crate::join_path(prefix, #leaf);
                        crate::OptimState::state_flatten(value, &__path, out);
                    }
                });
                unflatten.push(quote! {
                    let #ident = {
                        let __path = crate::join_path(prefix, #leaf);
                        <#inner as crate::OptimState>::state_unflatten(&__path, src, device)
                    };
                });
            }
        }
    }

    quote! {
        impl #impl_generics crate::OptimState for #name #ty_generics #where_clause {
            fn state_flatten(&self, prefix: &str, out: &mut crate::OptimStateSink) {
                #(#flatten)*
            }

            fn state_unflatten(
                prefix: &str,
                src: &mut crate::OptimStateSource,
                device: &burn::tensor::Device,
            ) -> Option<Self> {
                #(#unflatten)*
                Some(Self { #(#ctor),* })
            }
        }
    }
    .into()
}
