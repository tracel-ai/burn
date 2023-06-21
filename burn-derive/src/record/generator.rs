use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::{parse_quote, Generics};

use crate::shared::field::{parse_fields, FieldTypeAnalyzer};

pub struct RecordGenerator {
    name: Ident,
    fields: Vec<FieldTypeAnalyzer>,
    generics: Generics,
}

impl RecordGenerator {
    pub fn from_ast(ast: &syn::DeriveInput) -> Self {
        Self {
            name: ast.ident.clone(),
            fields: parse_fields(ast)
                .into_iter()
                .map(FieldTypeAnalyzer::new)
                .collect(),
            generics: ast.generics.clone(),
        }
    }

    pub fn gen_record_item_struct(&self) -> TokenStream {
        let mut fields = quote! {};
        let mut bounds = quote! {};

        for field in self.fields.iter() {
            let ty = &field.field.ty;
            let name = &field.field.ident;

            fields.extend(quote! {
                /// The #name field.
                pub #name: <#ty as burn::record::Record>::Item<S>,
            });
            bounds.extend(quote!{
                <#ty as burn::record::Record>::Item<S>: serde::Serialize + serde::de::DeserializeOwned,
            });
        }
        let name = self.record_item_name();
        let generics = self.record_item_generics();
        let bound = bounds.to_string();

        quote! {

            /// The record item type for the module.
            #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
            #[serde(bound = #bound)]
            pub struct #name #generics {
                #fields
            }
        }
    }

    pub fn gen_impl_record(&self) -> TokenStream {
        let mut body_into_item = quote! {};
        let mut body_from_item = quote! {};

        for field in self.fields.iter() {
            let name = &field.field.ident;

            body_into_item.extend(quote! {
                #name: burn::record::Record::into_item::<S>(self.#name),
            });
            body_from_item.extend(quote! {
                #name: burn::record::Record::from_item::<S>(item.#name),
            });
        }
        let name = &self.name;
        let name_item = self.record_item_name();
        let item_generics = self.record_item_generics();
        let (_, ty_generics_item, _) = item_generics.split_for_impl();
        let (impl_generics, ty_generics, where_clause) = self.generics.split_for_impl();

        quote! {
            impl #impl_generics burn::record::Record for #name #ty_generics #where_clause {
                type Item<S: burn::record::PrecisionSettings> = #name_item #ty_generics_item;

                fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
                    #name_item {
                        #body_into_item
                    }
                }

                fn from_item<S: burn::record::PrecisionSettings>(item: Self::Item<S>) -> Self {
                    Self {
                        #body_from_item
                    }
                }
            }
        }
    }

    pub fn record_item_name(&self) -> Ident {
        Ident::new(format!("{}Item", self.name).as_str(), self.name.span())
    }

    pub fn record_item_generics(&self) -> Generics {
        let param: syn::Generics = parse_quote! { <S: burn::record::PrecisionSettings >};
        let mut generics = self.generics.clone();
        for param in param.params.into_iter() {
            generics.params.push(param);
        }

        generics
    }
}
