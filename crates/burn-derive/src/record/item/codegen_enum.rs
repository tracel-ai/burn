use crate::shared::enum_variant::{parse_variants, EnumVariant};
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::{parse_quote, Generics};

use super::codegen::RecordItemCodegen;

pub(crate) struct EnumRecordItemCodegen {
    /// Enum variants.
    variants: Vec<EnumVariant>,
}

impl RecordItemCodegen for EnumRecordItemCodegen {
    fn from_ast(ast: &syn::DeriveInput) -> Self {
        Self {
            variants: parse_variants(ast),
        }
    }

    fn gen_item_type(
        &self,
        item_name: &Ident,
        generics: &Generics,
        has_backend: bool,
    ) -> TokenStream {
        let mut variants = quote! {};
        let mut bounds = quote! {};

        // Capture the Record enum variant types and names to transpose them in RecordItem
        for variant in self.variants.iter() {
            let ty = &variant.ty;
            let name = &variant.ident;

            variants.extend(quote! {
                /// Variant to be serialized.
                #name(<#ty as burn::record::Record<B>>::Item<S>),
            });

            // Item types must implement serialization/deserialization
            bounds.extend(quote! {
              <#ty as burn::record::Record<B>>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
          });
        }
        let bound = bounds.to_string();

        // Capture the type's generics and bounds in where clauses
        let (generics, generics_where) = if !has_backend {
            let mut generics = generics.clone();
            let param: syn::TypeParam = parse_quote! { B: burn::tensor::backend::Backend };
            generics.params.push(syn::GenericParam::Type(param));
            let (generics, _, generics_where) = generics.split_for_impl();
            (quote! { #generics }, quote! { #generics_where })
        } else {
            let (generics, _, generics_where) = generics.split_for_impl();
            (quote! { #generics }, quote! { #generics_where })
        };

        // Return the generated stream of token trees (i.e., code to be generated)
        quote! {

            /// The record item type for the module.
            #[derive(burn::serde::Serialize, burn::serde::Deserialize)]
            #[serde(crate = "burn::serde")]
            #[serde(bound = #bound)]
            pub enum #item_name #generics #generics_where {
                #variants
            }
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
