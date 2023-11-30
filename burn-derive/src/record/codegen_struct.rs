use crate::shared::field::FieldTypeAnalyzer;
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::Generics;

use super::codegen::RecordItemCodegen;

#[derive(new)]
pub(crate) struct StructRecordItemCodegen {
    fields: Vec<FieldTypeAnalyzer>,
}

impl RecordItemCodegen for StructRecordItemCodegen {
    fn gen_item_type(&self, item_name: &Ident, generics: &Generics) -> TokenStream {
        let mut fields = quote! {};
        let mut bounds = quote! {};

        for field in self.fields.iter() {
            let ty = &field.field.ty;
            let name = &field.field.ident;

            fields.extend(quote! {
                /// Field to be serialized.
                pub #name: <#ty as burn::record::Record>::Item<S>,
            });
            bounds.extend(quote! {
          <#ty as burn::record::Record>::Item<S>: serde::Serialize + serde::de::DeserializeOwned,
      });
        }
        let bound = bounds.to_string();

        quote! {

            /// The record item type for the module.
            #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
            #[serde(bound = #bound)]
            pub struct #item_name #generics {
                #fields
            }
        }
    }

    fn gen_into_item(&self, item_name: &Ident) -> TokenStream {
        let mut body_into_item = quote! {};

        for field in self.fields.iter() {
            let name = &field.field.ident;

            body_into_item.extend(quote! {
                #name: burn::record::Record::into_item::<S>(self.#name),
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
                #name: burn::record::Record::from_item::<S>(item.#name),
            });
        }

        quote! {
            fn from_item<S: burn::record::PrecisionSettings>(item: Self::Item<S>) -> Self {
                Self {
                    #body_from_item
                }
            }
        }
    }
}
