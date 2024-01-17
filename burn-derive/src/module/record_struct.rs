use crate::shared::field::FieldTypeAnalyzer;
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::Generics;

use super::record::ModuleRecordCodegen;

#[derive(new)]
pub(crate) struct StructModuleRecordCodegen {
    fields: Vec<FieldTypeAnalyzer>,
}

impl ModuleRecordCodegen for StructModuleRecordCodegen {
    fn gen_record_type(&self, record_name: &Ident, generics: &Generics) -> TokenStream {
        let mut fields = quote! {};

        for field in self.fields.iter() {
            let ty = &field.field.ty;
            let name = &field.field.ident;

            fields.extend(quote! {
                /// The module record associative type.
                pub #name: <#ty as burn::module::Module<B>>::Record,
            });
        }

        let (generics, _generics_ty, generics_where) = generics.split_for_impl();

        quote! {

            /// The record type for the module.
            #[derive(burn::record::Record)]
            pub struct #record_name #generics #generics_where {
                #fields
            }
        }
    }
}
