//! `ExtensionType` derive implementation for structs and enums crossing the dispatch boundary.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, Ident, Type};

use crate::ir::{self, TensorKind};

/// Field layout of a struct or a single enum variant.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CaseStyle {
    Named,
    Unnamed,
    Unit,
}

/// One derive case: a struct is one case and an enum contributes one case per variant.
struct DeriveCase {
    path: TokenStream,
    style: CaseStyle,
    fields: Vec<CaseField>,
}

struct CaseField {
    bind: Ident,
    member: Option<Ident>,
    ty: Type,
    is_ext: bool,
    tensor_kind: Option<TensorKind>,
}

fn build_case(path: TokenStream, fields: &Fields) -> DeriveCase {
    let (style, raw): (CaseStyle, Vec<&syn::Field>) = match fields {
        Fields::Named(fields) => (CaseStyle::Named, fields.named.iter().collect()),
        Fields::Unnamed(fields) => (CaseStyle::Unnamed, fields.unnamed.iter().collect()),
        Fields::Unit => (CaseStyle::Unit, Vec::new()),
    };
    let fields = raw
        .iter()
        .enumerate()
        .map(|(index, field)| CaseField {
            bind: format_ident!("__ext_f{index}"),
            member: field.ident.clone(),
            ty: field.ty.clone(),
            is_ext: field
                .attrs
                .iter()
                .any(|attr| attr.path().is_ident("extension_type")),
            tensor_kind: TensorKind::from_type(&field.ty),
        })
        .collect();
    DeriveCase {
        path,
        style,
        fields,
    }
}

fn collect_cases(input: &DeriveInput) -> syn::Result<Vec<DeriveCase>> {
    let name = &input.ident;
    match &input.data {
        Data::Struct(data) => Ok(vec![build_case(quote!(#name), &data.fields)]),
        Data::Enum(data) => Ok(data
            .variants
            .iter()
            .map(|variant| {
                let ident = &variant.ident;
                build_case(quote!(#name::#ident), &variant.fields)
            })
            .collect()),
        Data::Union(_) => Err(syn::Error::new_spanned(
            name,
            "ExtensionType cannot be derived for unions",
        )),
    }
}

/// Destructure a case, binding only the fields selected by `needed`.
fn gen_case_pattern(case: &DeriveCase, needed: impl Fn(usize) -> bool) -> TokenStream {
    let path = &case.path;
    match case.style {
        CaseStyle::Unit => quote!(#path),
        CaseStyle::Named => {
            let entries = case.fields.iter().enumerate().map(|(index, field)| {
                let member = field.member.as_ref().expect("named field has an ident");
                if needed(index) {
                    let bind = &field.bind;
                    quote!(#member: #bind)
                } else {
                    quote!(#member: _)
                }
            });
            quote!(#path { #(#entries),* })
        }
        CaseStyle::Unnamed => {
            let entries = case.fields.iter().enumerate().map(|(index, field)| {
                if needed(index) {
                    let bind = &field.bind;
                    quote!(#bind)
                } else {
                    quote!(_)
                }
            });
            quote!(#path(#(#entries),*))
        }
    }
}

fn gen_case_ctor(case: &DeriveCase, expressions: &[TokenStream]) -> TokenStream {
    let path = &case.path;
    match case.style {
        CaseStyle::Unit => quote!(#path),
        CaseStyle::Named => {
            let entries = case
                .fields
                .iter()
                .zip(expressions)
                .map(|(field, expression)| {
                    let member = field.member.as_ref().expect("named field has an ident");
                    quote!(#member: #expression)
                });
            quote!(#path { #(#entries),* })
        }
        CaseStyle::Unnamed => quote!(#path(#(#expressions),*)),
    }
}

/// Generate one routing lookup arm, preferring direct fields before nested extension values.
fn gen_routing_tensor_arm(case: &DeriveCase, float_only: bool) -> TokenStream {
    let float_index = case
        .fields
        .iter()
        .position(|field| !field.is_ext && field.tensor_kind == Some(TensorKind::Float));
    let any_index = if float_only {
        None
    } else {
        case.fields
            .iter()
            .position(|field| !field.is_ext && field.tensor_kind.is_some())
    };
    let extension_indices: Vec<_> = case
        .fields
        .iter()
        .enumerate()
        .filter(|(_, field)| field.is_ext)
        .map(|(index, _)| index)
        .collect();
    let method = if float_only {
        format_ident!("routing_float_tensor")
    } else {
        format_ident!("routing_tensor")
    };

    let (needed, expression): (Vec<usize>, TokenStream) = if let Some(index) = float_index {
        let bind = &case.fields[index].bind;
        (vec![index], quote!(Some(#bind)))
    } else if let Some(index) = any_index {
        let bind = &case.fields[index].bind;
        (vec![index], quote!(Some(#bind)))
    } else if !extension_indices.is_empty() {
        let calls = extension_indices.iter().map(|&index| {
            let field = &case.fields[index];
            let bind = &field.bind;
            let dispatch_ty = ir::with_backend(&field.ty, quote!(burn::backend::Dispatch));
            quote! {
                .or_else(|| <#dispatch_ty as burn::backend::ExtensionType<burn::backend::Dispatch>>::#method(#bind))
            }
        });
        (
            extension_indices.clone(),
            quote!(Option::<&burn::backend::DispatchTensor>::None #(#calls)*),
        )
    } else {
        (
            Vec::new(),
            quote!(Option::<&burn::backend::DispatchTensor>::None),
        )
    };

    let pattern = gen_case_pattern(case, |index| needed.contains(&index));
    quote!(#pattern => #expression,)
}

pub(crate) fn expand(input: TokenStream) -> syn::Result<TokenStream> {
    let input: DeriveInput = syn::parse2(input)?;
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let cases = collect_cases(&input)?;

    let wrap_arms = cases.iter().map(|case| {
        let pattern = gen_case_pattern(case, |_| true);
        let expressions: Vec<_> = case
            .fields
            .iter()
            .map(|field| {
                let bind = &field.bind;
                if field.is_ext {
                    quote!(#bind.map_to_dispatch(&map_kind, autodiff))
                } else if let Some(kind) = field.tensor_kind {
                    let variant = kind.variant();
                    quote! {
                        burn::backend::DispatchTensor {
                            kind: map_kind(burn::backend::BackendTensor::#variant(#bind)),
                            autodiff,
                        }
                    }
                } else {
                    quote!(#bind)
                }
            })
            .collect();
        let constructor = gen_case_ctor(case, &expressions);
        quote!(#pattern => #constructor,)
    });

    let unwrap_arms = cases.iter().map(|case| {
        let pattern = gen_case_pattern(case, |_| true);
        let expressions: Vec<_> = case
            .fields
            .iter()
            .map(|field| {
                let bind = &field.bind;
                if field.is_ext {
                    quote!(burn::backend::ExtensionType::map_from_dispatch(#bind, &unwrap_kind))
                } else if let Some(kind) = field.tensor_kind {
                    let method = kind.accessor();
                    quote!(unwrap_kind(#bind.kind).#method())
                } else {
                    quote!(#bind)
                }
            })
            .collect();
        let constructor = gen_case_ctor(case, &expressions);
        quote!(#pattern => #constructor,)
    });

    let routing_tensor_arms = cases.iter().map(|case| gen_routing_tensor_arm(case, false));
    let float_routing_tensor_arms = cases.iter().map(|case| gen_routing_tensor_arm(case, true));

    Ok(quote! {
        impl #impl_generics burn::backend::ExtensionType<B> for #name #ty_generics #where_clause {
            type Target = #name<burn::backend::Dispatch>;

            #[allow(unused_variables)]
            fn map_to_dispatch<F>(
                self,
                map_kind: F,
                autodiff: burn::backend::DispatchAutodiffContext,
            ) -> Self::Target
            where
                F: Fn(burn::backend::BackendTensor<B>) -> burn::backend::DispatchTensorKind,
            {
                match self { #(#wrap_arms)* }
            }

            #[allow(unused_variables)]
            fn map_from_dispatch<F>(target: Self::Target, unwrap_kind: F) -> Self
            where
                F: Fn(burn::backend::DispatchTensorKind) -> burn::backend::BackendTensor<B>,
            {
                match target { #(#unwrap_arms)* }
            }

            fn routing_tensor(target: &Self::Target) -> Option<&burn::backend::DispatchTensor> {
                match target { #(#routing_tensor_arms)* }
            }

            fn routing_float_tensor(target: &Self::Target) -> Option<&burn::backend::DispatchTensor> {
                match target { #(#float_routing_tensor_arms)* }
            }
        }
    })
}
