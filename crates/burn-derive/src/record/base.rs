use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::{parse_quote, Generics};

use super::{codegen::RecordItemCodegen, codegen_struct::StructRecordItemCodegen};
use crate::shared::field::{parse_fields, FieldTypeAnalyzer};

pub(crate) fn derive_impl(ast: &syn::DeriveInput) -> proc_macro::TokenStream {
    let record_gen = RecordDeriveCodegen::from_ast(ast);
    let item_struct = record_gen.gen_record_type();
    let record_impl = record_gen.gen_impl_record();

    quote! {
        #item_struct
        #record_impl
    }
    .into()
}

struct RecordDeriveCodegen {
    name_record: Ident,
    name_item: Ident,
    gen: StructRecordItemCodegen,
    generics: Generics,
    has_backend: bool,
}

impl RecordDeriveCodegen {
    pub(crate) fn from_ast(ast: &syn::DeriveInput) -> Self {
        let name_record = ast.ident.clone();
        let name_item = Ident::new(format!("{}Item", name_record).as_str(), name_record.span());
        let has_backend = ast
            .generics
            .type_params()
            .map(|param| param.ident == "B")
            .reduce(|accum, is_backend| is_backend || accum)
            .unwrap_or(false);

        Self {
            name_record,
            name_item,
            gen: StructRecordItemCodegen::new(
                parse_fields(ast)
                    .into_iter()
                    .map(FieldTypeAnalyzer::new)
                    .collect(),
            ),
            generics: ast.generics.clone(),
            has_backend,
        }
    }

    /// Generate the record type with the correct generics.
    pub(crate) fn gen_record_type(&self) -> TokenStream {
        let param: syn::Generics = parse_quote! { <S: burn::record::PrecisionSettings >};
        let mut generics = self.generics.clone();

        for param in param.params.into_iter() {
            generics.params.push(param);
        }

        self.gen
            .gen_item_type(&self.name_item, &generics, self.has_backend)
    }

    /// Generate the implementation for the Record trait.
    pub(crate) fn gen_impl_record(&self) -> TokenStream {
        let name = &self.name_record;
        let item_generics = self.record_item_generics();
        let (_, ty_generics_item, _) = item_generics.split_for_impl();
        let (impl_generics, ty_generics, where_clause) = self.generics.split_for_impl();

        let impl_generics = if let Some(impl_generic) = self.impl_generics() {
            impl_generic
        } else {
            quote! { #impl_generics }
        };

        let name_item = &self.name_item;
        let into_item_fn = self.gen.gen_into_item(name_item);
        let from_item_fn = self.gen.gen_from_item();

        quote! {
            impl #impl_generics burn::record::Record<B> for #name #ty_generics #where_clause {
                type Item<S: burn::record::PrecisionSettings> = #name_item #ty_generics_item;

                #into_item_fn
                #from_item_fn

            }
        }
    }

    fn impl_generics(&self) -> Option<TokenStream> {
        if self.has_backend {
            return None;
        }

        let param: syn::TypeParam = parse_quote! { B: burn::tensor::backend::Backend };
        let mut generics = self.generics.clone();
        generics.params.push(syn::GenericParam::Type(param));

        let (impl_generics, _ty_generics, _where_clause) = generics.split_for_impl();

        Some(quote! {#impl_generics})
    }

    fn record_item_generics(&self) -> Generics {
        let param: syn::Generics = parse_quote! { <S: burn::record::PrecisionSettings >};
        let mut generics = self.generics.clone();
        for param in param.params.into_iter() {
            generics.params.push(param);
        }

        if !self.has_backend {
            let param: syn::TypeParam = parse_quote! { B: burn::tensor::backend::Backend };
            generics.params.push(syn::GenericParam::Type(param));
        }

        generics
    }
}
