use super::{
    codegen::ModuleCodegen, codegen_struct::StructModuleCodegen, record::ModuleRecordCodegen,
    record_struct::StructModuleRecordCodegen,
};
use crate::module::display;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_quote, Ident};

pub(crate) fn derive_impl(ast: &syn::DeriveInput) -> TokenStream {
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
    let backend_trait = fetch_backend_trait(&ast.generics);

    let display_fn = display::display_fn(name);

    let generator = StructModuleCodegen::from_ast(ast);
    let num_params_fn = generator.gen_num_params();
    let visit = generator.gen_visit();
    let map_mut = generator.gen_map();
    let collect_devices = generator.gen_collect_devices();
    let to_device = generator.gen_to_device();
    let fork = generator.gen_fork();
    let valid_fn = generator.gen_valid();
    let into_record_fn = generator.gen_into_record();
    let load_record_fn = generator.gen_load_record();
    let clone_fn = generator.gen_clone();
    let generics_names_except_backend = generics_names_except_backend(&ast.generics);

    let record_name = Ident::new(format!("{}Record", name).as_str(), name.span());
    let record_gen = StructModuleRecordCodegen::new(generator.fields);
    let record_struct = record_gen.gen_record_type(&record_name, &ast.generics);

    let gen = quote! {
        impl #generics burn::module::Module<B> for #name #generics_ty #generics_where {
            type Record = #record_name #generics_ty;

            #load_record_fn
            #into_record_fn

            #num_params_fn

            #visit
            #map_mut

            #collect_devices
            #to_device
            #fork
        }

        impl #generics burn::module::AutodiffModule<B> for #name #generics_ty
        where
            B: burn::tensor::backend::AutodiffBackend,
            <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: #backend_trait,
        {
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
    let backend_ad: syn::Generics = parse_quote! { <B: burn::tensor::backend::AutodiffBackend >};

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

        impl #generics_module_ad burn::module::AutodiffModule<B>
            for #name #generics_ty #generics_where {
            burn::constant!(ad_module, #name #generics_ty);
        }
    };

    gen.into()
}

fn fetch_backend_trait(generics: &syn::Generics) -> proc_macro2::TokenStream {
    static BACKEND_TRAIT_COMPILATION_ERROR_MSG: &str = "Modules should be generic over a backend.
    - The generic argument named `B` should have its first trait bound being a backend trait.
    - The default backend trait is `burn::tensor::backend::Backend`.
    - Any backend trait is supported.";

    for param in generics.params.iter() {
        if let syn::GenericParam::Type(ty) = &param {
            if ty.ident == "B" {
                let bound = ty
                    .bounds
                    .first()
                    .expect(BACKEND_TRAIT_COMPILATION_ERROR_MSG);

                return quote! {
                    #bound
                };
            }
        }
    }

    panic!("{BACKEND_TRAIT_COMPILATION_ERROR_MSG}");
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
