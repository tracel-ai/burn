use crate::shared::enum_variant::EnumVariant;
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::{Generics, Visibility};

use super::record::ModuleRecordCodegen;

#[derive(new)]
pub(crate) struct EnumModuleRecordCodegen {
    variants: Vec<EnumVariant>,
    vis: Visibility,
}

impl ModuleRecordCodegen for EnumModuleRecordCodegen {
    fn gen_record_type(&self, record_name: &Ident, generics: &Generics) -> (TokenStream, Generics) {
        let mut variants = quote! {};
        let vis = &self.vis;

        // Capture the Record enum variant types
        for variant in self.variants.iter() {
            let ty = &variant.ty;
            let name = &variant.ident;

            variants.extend(quote! {
                /// The module record associative type.
                #name(<#ty as burn::module::Module<B>>::Record),
            });
        }

        let (impl_generics, _generics_ty, generics_where) = generics.split_for_impl();

        (
            quote! {

                /// The record type for the module.
                #[derive(burn::record::Record)]
                #vis enum #record_name #impl_generics #generics_where {
                    #variants
                }
            },
            generics.clone(),
        )
    }
}
