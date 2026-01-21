use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::{Generics, parse_quote};

use crate::record::item::codegen::RecordItemCodegen;

pub(crate) fn generate_record<G: RecordItemCodegen>(ast: &syn::DeriveInput) -> TokenStream {
    let record_gen: syn::Result<RecordCodegen<G>> = RecordCodegen::from_ast(ast);
    match record_gen {
        Ok(record_gen) => {
            let item_type = record_gen.gen_record_type();
            let record_impl = record_gen.gen_impl_record();

            quote! {
                #item_type
                #record_impl
            }
        }
        Err(err) => err.to_compile_error(),
    }
}

pub(crate) struct RecordCodegen<G: RecordItemCodegen> {
    /// Record type info.
    ty: RecordType,
    /// Record item code gen.
    codegen: G,
}

impl<G: RecordItemCodegen> RecordCodegen<G> {
    /// Generate the record type with the correct generics.
    pub(crate) fn gen_record_type(&self) -> TokenStream {
        // Add precision settings type bound
        let param: syn::Generics = parse_quote! { <S: burn::record::PrecisionSettings >};
        let mut generics = self.ty.generics.clone();

        for param in param.params.into_iter() {
            generics.params.push(param);
        }

        // Generate the record item definition
        self.codegen
            .gen_item_type(&self.ty.item, &generics, self.ty.has_backend)
    }

    /// Generate the implementation for the Record trait.
    pub(crate) fn gen_impl_record(&self) -> TokenStream {
        // Capture the record type's generics and bounds in where clauses
        let item_generics = self.record_item_generics();
        let (_, ty_generics_item, _) = item_generics.split_for_impl();
        let (impl_generics, ty_generics, where_clause) = self.ty.generics.split_for_impl();

        let impl_generics = if let Some(impl_generic) = self.impl_generics() {
            impl_generic
        } else {
            quote! { #impl_generics }
        };

        let name_item = &self.ty.item;
        let into_item_fn = self.codegen.gen_into_item(name_item);
        let from_item_fn = self.codegen.gen_from_item();

        // Return the generated stream of token trees (i.e., code to be generated)
        let name = &self.ty.name;
        quote! {
            impl #impl_generics burn::record::Record<B> for #name #ty_generics #where_clause {
                type Item<S: burn::record::PrecisionSettings> = #name_item #ty_generics_item;

                #into_item_fn
                #from_item_fn

            }
        }
    }

    /// Add backend generic type to the implementation block.
    fn impl_generics(&self) -> Option<TokenStream> {
        if self.ty.has_backend {
            return None;
        }

        let param: syn::TypeParam = parse_quote! { B: burn::tensor::backend::Backend };
        let mut generics = self.ty.generics.clone();
        generics.params.push(syn::GenericParam::Type(param));

        let (impl_generics, _ty_generics, _where_clause) = generics.split_for_impl();

        Some(quote! {#impl_generics})
    }

    /// Get the generics attached to the record item type.
    fn record_item_generics(&self) -> Generics {
        let param: syn::Generics = parse_quote! { <S: burn::record::PrecisionSettings >};
        let mut generics = self.ty.generics.clone();
        for param in param.params.into_iter() {
            generics.params.push(param);
        }

        if !self.ty.has_backend {
            let param: syn::TypeParam = parse_quote! { B: burn::tensor::backend::Backend };
            generics.params.push(syn::GenericParam::Type(param));
        }

        generics
    }

    pub(crate) fn from_ast(ast: &syn::DeriveInput) -> syn::Result<Self> {
        Ok(Self {
            ty: RecordType::from_ast(ast),
            codegen: G::from_ast(ast)?,
        })
    }
}

/// Information about a record type.
struct RecordType {
    /// Record type name.
    name: Ident,
    /// Record item type name.
    item: Ident,
    /// Lifetimes and type parameters attached to the record type declaration.
    generics: Generics,
    /// Whether or not the record type should specify a backend generic.
    has_backend: bool,
}

impl RecordType {
    fn from_ast(ast: &syn::DeriveInput) -> Self {
        let name = ast.ident.clone();
        let item = Ident::new(format!("{name}Item").as_str(), name.span());
        let has_backend = ast
            .generics
            .type_params()
            .map(|param| param.ident == "B")
            .reduce(|accum, is_backend| is_backend || accum)
            .unwrap_or(false);

        Self {
            name,
            item,
            generics: ast.generics.clone(),
            has_backend,
        }
    }
}
