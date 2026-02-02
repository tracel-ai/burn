use crate::shared::field::{FieldTypeAnalyzer, parse_fields};
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::{Generics, Visibility, parse_quote};

use super::codegen::RecordItemCodegen;

pub(crate) struct StructRecordItemCodegen {
    fields: Vec<FieldTypeAnalyzer>,
    vis: Visibility,
}

impl RecordItemCodegen for StructRecordItemCodegen {
    fn from_ast(ast: &syn::DeriveInput) -> syn::Result<Self> {
        Ok(Self {
            fields: parse_fields(ast)
                .into_iter()
                .map(FieldTypeAnalyzer::new)
                .collect(),
            vis: ast.vis.clone(),
        })
    }

    fn gen_item_type(
        &self,
        item_name: &Ident,
        generics: &Generics,
        has_backend: bool,
    ) -> TokenStream {
        let mut fields = quote! {};
        let mut serde_bounds = quote! {};
        let mut clone_bounds = vec![];
        let mut clone_delegate = quote! {};
        let vis = &self.vis;

        for field in self.fields.iter() {
            let ty = &field.field.ty;
            let name = &field.field.ident;

            fields.extend(quote! {
                /// Field to be serialized.
                pub #name: <#ty as burn::record::Record<B>>::Item<S>,
            });

            serde_bounds.extend(quote! {
                <#ty as burn::record::Record<B>>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
            });

            clone_bounds.push(parse_quote! {
                <#ty as burn::record::Record<B>>::Item<S>: Clone
            });

            clone_delegate.extend(quote! {
                #name: self.#name.clone(),
            });
        }
        let serde_bound = serde_bounds.to_string();

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
                    Self {
                        #clone_delegate
                    }
                }
            }
        };

        quote! {

            /// The record item type for the module.
            #[derive(burn::serde::Serialize, burn::serde::Deserialize)]
            #[serde(crate = "burn::serde")]
            #[serde(bound = #serde_bound)]
            #vis struct #item_name #generics #generics_where {
                #fields
            }

            #clone_impl
        }
    }

    fn gen_into_item(&self, item_name: &Ident) -> TokenStream {
        let mut body_into_item = quote! {};

        for field in self.fields.iter() {
            let name = &field.field.ident;

            body_into_item.extend(quote! {
                #name: burn::record::Record::<B>::into_item::<S>(self.#name),
            });
        }

        quote! {
            fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
                #item_name {
                    #body_into_item
                }
            }
        }
    }

    fn gen_from_item(&self) -> TokenStream {
        let mut body_from_item = quote! {};

        for field in self.fields.iter() {
            let name = &field.field.ident;

            body_from_item.extend(quote! {
                #name: burn::record::Record::<B>::from_item::<S>(item.#name, device),
            });
        }

        quote! {
            fn from_item<S: burn::record::PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
                Self {
                    #body_from_item
                }
            }
        }
    }
}
