use super::fn_generator::FnGenerator;
use crate::module::display;
use proc_macro::TokenStream;
use quote::quote;

pub(crate) fn module_derive_impl(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let (generics, generics_ty, generics_where) = ast.generics.split_for_impl();

    let display_fn = display::display_fn(name);

    let generator = FnGenerator::from_ast(ast);
    let num_params_fn = generator.gen_num_params_fn();
    let visit = generator.gen_visit_fn();
    let map_mut = generator.gen_map_fn();
    let devices_fn = generator.gen_devices_fn();
    let to_device_fn = generator.gen_to_device_fn();
    let state_fn = generator.gen_state_fn();
    let load_fn = generator.gen_load_fn();
    let inner_fn = generator.gen_inner_fn();
    let from_inner_fn = generator.gen_from_inner_fn();
    let detach_fn = generator.gen_detach_fn();
    let clone_fn = generator.gen_clone_fn();
    let generics_names_except_backend = generics_names_except_backend(&ast.generics);

    let gen = quote! {
        impl #generics burn::module::Module<B> for #name #generics_ty #generics_where {
            #devices_fn
            #to_device_fn

            #state_fn
            #load_fn

            #num_params_fn
            #detach_fn

            #visit
            #map_mut
        }

        impl #generics burn::module::ADModule<B> for #name #generics_ty where B: burn::tensor::backend::ADBackend, {
            type InnerModule=#name<B::InnerBackend, #generics_names_except_backend>;

            #inner_fn
            #from_inner_fn
        }

        impl #generics core::fmt::Display for #name #generics_ty #generics_where {
            #display_fn
        }

        impl #generics Clone for #name #generics_ty #generics_where {
            #clone_fn
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
