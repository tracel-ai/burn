use super::{
    codegen::ModuleCodegen, codegen_struct::StructModuleCodegen, record::ModuleRecordCodegen,
    record_struct::StructModuleRecordCodegen,
};
use crate::{module::display, shared::generics::GenericsHelper};
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse2, parse_quote, Ident};

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

    let mut generics_for_module = GenericsHelper::new(ast.generics.clone());
    let mut generics_for_autodiff_module = GenericsHelper::new(ast.generics.clone());
    let backend_trait = generics_for_module.fetch_backend_trait();

    generics_for_autodiff_module.add_predicate(
        parse2(quote! {
                B: burn::tensor::backend::AutodiffBackend
        })
        .expect("Can add autodiff backend predicate."),
    );
    generics_for_autodiff_module.add_predicate(
        parse2(quote! {
                <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: #backend_trait
        })
        .expect("Can add backend trait predicate"),
    );

    let mut generics_names_except_backend = quote! {};

    generics_for_module
        .types()
        .into_iter()
        .filter(|ident| ident != "B")
        .for_each(|ident| {
            generics_names_except_backend.extend(quote! { #ident, });
            generics_for_module.add_predicate(
                parse2(quote! {
                    #ident: burn::module::Module<B>
                })
                .unwrap(),
            );
            generics_for_autodiff_module.add_predicate(
                parse2(quote! {
                    #ident: burn::module::AutodiffModule<B>
                })
                .unwrap(),
            );
            generics_for_autodiff_module.add_predicate(
                parse2(quote! {
                    #ident: burn::module::Module<B::InnerBackend>
                })
                .unwrap(),
            );
        });

    generics_for_module.consts().into_iter().for_each(|ident| {
        generics_names_except_backend.extend(quote! { #ident, });
    });

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

    let record_name = Ident::new(format!("{}Record", name).as_str(), name.span());
    let record_gen = StructModuleRecordCodegen::new(generator.fields);
    let record_struct = record_gen.gen_record_type(&record_name, &generics_for_module.generics);

    let (generics_m, generics_ty_m, generics_where_m) =
        generics_for_module.generics.split_for_impl();
    let (generics_a, generics_ty_a, generics_where_a) =
        generics_for_autodiff_module.generics.split_for_impl();

    let gen = quote! {
        impl #generics_m burn::module::Module<B> for #name #generics_ty_m #generics_where_m {
            type Record = #record_name #generics_ty_m;

            #load_record_fn
            #into_record_fn

            #num_params_fn

            #visit
            #map_mut

            #collect_devices
            #to_device
            #fork
        }

        impl #generics_a burn::module::AutodiffModule<B> for #name #generics_ty_a #generics_where_a
        {
            type InnerModule=#name<B::InnerBackend, #generics_names_except_backend>;

            #valid_fn
        }

        impl #generics_m core::fmt::Display for #name #generics_ty_m #generics_where_m {
            #display_fn
        }

        impl #generics_m Clone for #name #generics_ty_a #generics_where_m {
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
