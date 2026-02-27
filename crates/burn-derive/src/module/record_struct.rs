use std::collections::HashSet;

use crate::module::codegen_struct::{ModuleField, ModuleFieldAttribute};
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::{Generics, Visibility};

use super::record::ModuleRecordCodegen;

#[derive(new)]
pub(crate) struct StructModuleRecordCodegen {
    fields: Vec<ModuleField>,
    vis: Visibility,
}

impl ModuleRecordCodegen for StructModuleRecordCodegen {
    fn gen_record_type(&self, record_name: &Ident, generics: &Generics) -> (TokenStream, Generics) {
        let mut fields = quote! {};
        let vis = &self.vis;

        let mut used_generics = HashSet::new();

        for field in self.fields.iter() {
            let ty = &field.field.ty;
            let name = &field.field.ident;

            if field.field_type.is_persistent_module() || field.field_type.maybe_generic_module() {
                fields.extend(quote! {
                    /// The module record associative type.
                    #vis #name: <#ty as burn::module::Module<B>>::Record,
                });

                used_generics.extend(&field.field_type.generic_idents);
            } else {
                match field.field_type.attr {
                    Some(ModuleFieldAttribute::Constant) => {
                        // If constant, store the type directly in the record.
                        // This assumes the type implements Clone and Serialize/Deserialize.
                        fields.extend(quote! {
                            #[allow(missing_docs)]
                            #vis #name: burn::module::ValueRecord<#ty>,
                        });

                        used_generics.extend(&field.field_type.generic_idents);
                    }
                    // Default (None) gets skipped
                    None | Some(ModuleFieldAttribute::Skip) => {
                        fields.extend(quote! {
                            #[allow(missing_docs)]
                            #vis #name: burn::module::EmptyRecord,
                        });

                        // Do not capture generics from this field since it produces an empty record
                    }
                }
            }
        }

        let mut filtered_generics = generics.clone();
        filtered_generics.params = generics
            .params
            .iter()
            .filter(|param| match param {
                syn::GenericParam::Type(ty) if ty.ident == "B" => true,
                syn::GenericParam::Type(ty) => used_generics.contains(&ty.ident),
                _ => true,
            })
            .cloned()
            .collect();

        let (impl_generics, _generics_ty, generics_where) = filtered_generics.split_for_impl();

        (
            quote! {

                /// The record type for the module.
                #[derive(burn::record::Record)]
                #vis struct #record_name #impl_generics #generics_where {
                    #fields
                }
            },
            filtered_generics,
        )
    }
}
