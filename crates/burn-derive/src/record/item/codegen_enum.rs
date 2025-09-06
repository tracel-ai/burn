use crate::shared::enum_variant::{EnumVariant, parse_variants};
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::{Generics, Visibility, parse_quote};

use super::codegen::RecordItemCodegen;

pub(crate) struct EnumRecordItemCodegen {
    /// Enum variants.
    variants: Vec<EnumVariant>,
    vis: Visibility,
}

impl RecordItemCodegen for EnumRecordItemCodegen {
    fn from_ast(ast: &syn::DeriveInput) -> Self {
        Self {
            variants: parse_variants(ast),
            vis: ast.vis.clone(),
        }
    }

    fn gen_item_type(
        &self,
        item_name: &Ident,
        generics: &Generics,
        has_backend: bool,
    ) -> TokenStream {
        let mut variants = quote! {};
        let mut serde_bounds = quote! {};
        let mut clone_bounds = vec![];
        let mut clone_match_arms = quote! {};
        let vis = &self.vis;

        // Capture the Record enum variant types and names to transpose them in RecordItem
        for variant in self.variants.iter() {
            let ty = &variant.ty;
            let name = &variant.ident;

            variants.extend(quote! {
                /// Variant to be serialized.
                #name(<#ty as burn::record::Record<B>>::Item<S>),
            });

            // Item types must implement serialization/deserialization
            serde_bounds.extend(quote! {
                <#ty as burn::record::Record<B>>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
            });
            clone_bounds.push(parse_quote! {
                <#ty as burn::record::Record<B>>::Item<S>: Clone
            });

            clone_match_arms.extend(quote! {
                Self::#name(inner) => Self::#name(inner.clone()),
            });
        }
        let serde_bound = serde_bounds.to_string();

        // Capture the type's generics and bounds in where clauses
        let mut generics = generics.clone();
        if !has_backend {
            let param: syn::TypeParam = parse_quote! { B: burn::tensor::backend::Backend };
            generics.params.push(syn::GenericParam::Type(param));
        }
        let (generics, type_generics, generics_where) = generics.split_for_impl();

        let clone_bounds = generics_where.cloned().map(|mut where_clause| {
            for predicate in clone_bounds {
                where_clause.predicates.push(predicate);
            }
            where_clause
        });

        let clone_impl = quote! {
            impl #generics Clone for #item_name #type_generics #clone_bounds {
                fn clone(&self) -> Self {
                    match self {
                        #clone_match_arms
                    }
                }
            }
        };

        // Return the generated stream of token trees (i.e., code to be generated)
        quote! {

            /// The record item type for the module.
            #[derive(burn::serde::Serialize, burn::serde::Deserialize)]
            #[serde(crate = "burn::serde")]
            #[serde(bound = #serde_bound)]
            #vis enum #item_name #generics #generics_where {
                #variants
            }

            #clone_impl
        }
    }

    fn gen_into_item(&self, _item_name: &Ident) -> TokenStream {
        let mut into_item_match_arms = quote! {};

        for variant in self.variants.iter() {
            let name = &variant.ident;

            into_item_match_arms.extend(quote! {
                Self::#name(record) => Self::Item::#name(burn::record::Record::<B>::into_item::<S>(record)),
            });
        }

        quote! {
            fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
                match self {
                    #into_item_match_arms
                }
            }
        }
    }

    fn gen_from_item(&self) -> TokenStream {
        let mut from_item_match_arms = quote! {};

        for variant in self.variants.iter() {
            let name = &variant.ident;

            from_item_match_arms.extend(quote! {
                Self::Item::#name(item) => Self::#name(burn::record::Record::<B>::from_item::<S>(item, device)),
            });
        }

        quote! {
            fn from_item<S: burn::record::PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
                match item {
                    #from_item_match_arms
                }
            }
        }
    }
}
