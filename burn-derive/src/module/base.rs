use super::{generator::FnGenerator, record::ModuleRecordGenerator};
use crate::module::display;
use proc_macro::TokenStream;
use quote::quote;
use syn::parse_quote;

pub(crate) fn module_derive_impl(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let has_backend = ast
        .generics
        .type_params()
        .map(|param| param.ident == "B")
        .reduce(|accum, is_backend| is_backend || accum)
        .unwrap_or(false);

    if !has_backend {
        return constant_impl(ast);
    }

    let (generics, generics_ty, generics_where) = ast.generics.split_for_impl();

    let display_fn = display::display_fn(name);

    let generator = FnGenerator::from_ast(ast);
    let num_params_fn = generator.gen_num_params_fn();
    let visit = generator.gen_visit_fn();
    let map_mut = generator.gen_map_fn();
    let valid_fn = generator.gen_valid_fn();
    let into_record_fn = generator.gen_into_record_fn();
    let load_record_fn = generator.gen_load_record_fn();
    let clone_fn = generator.gen_clone_fn();
    let generics_names_except_backend = generics_names_except_backend(&ast.generics);

    let record_gen =
        ModuleRecordGenerator::new(name.clone(), generator.fields, ast.generics.clone());

    let record_name = record_gen.record_name();
    let record_struct = record_gen.gen_record_struct();

    let gen = quote! {
        impl #generics burn::module::Module<B> for #name #generics_ty #generics_where {
            type Record = #record_name #generics_ty;

            #load_record_fn
            #into_record_fn

            #num_params_fn

            #visit
            #map_mut
        }

        impl #generics burn::module::ADModule<B> for #name #generics_ty where B: burn::tensor::backend::ADBackend, {
            type InnerModule=#name<B::InnerBackend, #generics_names_except_backend>;

            #valid_fn
        }

        impl #generics core::fmt::Display for #name #generics_ty #generics_where {
            #display_fn
        }

        impl #generics Clone for #name #generics_ty #generics_where {
            #clone_fn
        }

        #record_struct
    };

    gen.into()
}

// When there is no backend in the generic parameter, the struct is considered as a constant.
fn constant_impl(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let (_, generics_ty, generics_where) = ast.generics.split_for_impl();

    let backend: syn::Generics = parse_quote! { <B: burn::tensor::backend::Backend >};
    let backend_ad: syn::Generics = parse_quote! { <B: burn::tensor::backend::ADBackend >};

    let mut generics_module = ast.generics.clone();
    let mut generics_module_ad = ast.generics.clone();

    for param in backend.params.into_iter() {
        generics_module.params.push(param);
    }
    for param in backend_ad.params.into_iter() {
        generics_module_ad.params.push(param);
    }
    let (generics_module, _, _) = generics_module.split_for_impl();
    let (generics_module_ad, _, _) = generics_module_ad.split_for_impl();

    let gen = quote! {
        impl #generics_module burn::module::Module<B> for #name #generics_ty #generics_where {
            burn::constant!(module);
        }

        impl #generics_module_ad burn::module::ADModule<B> for #name #generics_ty #generics_where {
            burn::constant!(ad_module, #name #generics_ty);
        }
    };

    gen.into()
}

fn generics_names_except_backend(generics: &syn::Generics) -> proc_macro2::TokenStream {
    let mut named = quote! {};

    generics.params.iter().for_each(|param| {
        match param {
            syn::GenericParam::Type(ty) => {
                if ty.ident != "B" {
                    let ident = &ty.ident;
                    named.extend(quote! { #ident, });
                }
            }
            syn::GenericParam::Lifetime(_) => panic!("Lifetime not supported in module"),
            syn::GenericParam::Const(c) => {
                let ident = &c.ident;
                named.extend(quote! { #ident, });
            }
        };
    });

    named
}
