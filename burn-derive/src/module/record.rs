use crate::shared::field::FieldTypeAnalyzer;
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::Generics;

pub struct ModuleRecordGenerator {
    name: Ident,
    fields: Vec<FieldTypeAnalyzer>,
    generics: Generics,
}

impl ModuleRecordGenerator {
    pub fn new(name: Ident, fields: Vec<FieldTypeAnalyzer>, generics: Generics) -> Self {
        Self {
            name,
            fields,
            generics,
        }
    }

    pub fn gen_record_struct(&self) -> TokenStream {
        let mut fields = quote! {};

        for field in self.fields.iter() {
            let ty = &field.field.ty;
            let name = &field.field.ident;

            fields.extend(quote! {
                /// The #name field.
                pub #name: <#ty as burn::module::Module<B>>::Record,
            });
        }
        let name = self.record_name();
        let generics = &self.generics;

        quote! {

            /// The record type for the module.
            #[derive(burn::record::Record, Debug, Clone)]
            pub struct #name #generics {
                #fields
            }
        }
    }

    pub fn record_name(&self) -> Ident {
        Ident::new(format!("{}Record", self.name).as_str(), self.name.span())
    }
}
