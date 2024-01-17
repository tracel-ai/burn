use crate::shared::field::FieldTypeAnalyzer;
use proc_macro2::{Ident, TokenStream};
use quote::{quote, ToTokens};
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

            if type_with_defaults(ty) {
                fields.extend(quote! {
                    /// Field to be ignored.
                    #[serde(default)]
                    pub #name: <#ty as burn::record::Record>::Item<S>,
                });
            } else {
                fields.extend(quote! {
                    /// Field to be serialized.
                    pub #name: <#ty as burn::record::Record>::Item<S>,
                });
            }

            bounds.extend(quote! {
          <#ty as burn::record::Record>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
      });
        }
        let bound = bounds.to_string();

        quote! {

            /// The record item type for the module.
            #[derive(burn::serde::Serialize, burn::serde::Deserialize)]
            #[serde(crate = "burn::serde")]
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
            if type_with_defaults(&field.field.ty) {
                body_into_item.extend(quote! {
                    #name: Default::default(),
                });
            } else {
                body_into_item.extend(quote! {
                    #name: burn::record::Record::into_item::<S>(self.#name),
                });
            }
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
            if type_with_defaults(&field.field.ty) {
                body_from_item.extend(quote! {
                    #name: Default::default(),
                });
            } else {
                body_from_item.extend(quote! {
                    #name: burn::record::Record::from_item::<S>(item.#name),
                });
            }
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

/// Identify if the type should be labeled as `#[serde(default)]`.
///
/// This allows deserializing missing field types as their default values.
///
/// These types are: `bool`, `usize`, `f32`, `f64`, `half::bf16`, `half::f16`, `u64`, `u32`, `u16`,
/// `u8`, `i64`, `i32`, `i16`, `i8`, `alloc::string::String`, which are ignored typically during record
/// serialization.
///
///
fn type_with_defaults(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Path(type_path) => {
            if let Some(ident) = type_path.path.get_ident() {
                ident == "bool"
                    || ident == "usize"
                    || ident == "f32"
                    || ident == "f64"
                    || ident == "half::bf16"
                    || ident == "half::f16"
                    || ident == "u64"
                    || ident == "u32"
                    || ident == "u16"
                    || ident == "u8"
                    || ident == "i64"
                    || ident == "i32"
                    || ident == "i16"
                    || ident == "i8"
                    || ident == "alloc::string::String"
            } else {
                // Check if the path represents an array type
                let path_as_string = type_path.to_token_stream().to_string();
                path_as_string.contains('[') && path_as_string.contains(']')
            }
        }
        // Handle other types if necessary
        _ => false,
    }
}
