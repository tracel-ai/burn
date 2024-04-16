use crate::shared::enum_variant::EnumVariant;
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::Generics;

use super::record::ModuleRecordCodegen;

#[derive(new)]
pub(crate) struct EnumModuleRecordCodegen {
    variants: Vec<EnumVariant>,
}

impl ModuleRecordCodegen for EnumModuleRecordCodegen {
    fn gen_record_type(&self, record_name: &Ident, generics: &Generics) -> TokenStream {
        let mut variants = quote! {};

        // Capture the Record enum variant types
        for variant in self.variants.iter() {
            let ty = &variant.ty;
            let name = &variant.ident;

            variants.extend(quote! {
                /// The module record associative type.
                #name(<#ty as burn::module::Module<B>>::Record),
            });
        }

        let (generics, _generics_ty, generics_where) = generics.split_for_impl();

        quote! {

            /// The record type for the module.
            #[derive(burn::record::Record)]
            pub enum #record_name #generics #generics_where {
                #variants
            }
        }
    }
}
