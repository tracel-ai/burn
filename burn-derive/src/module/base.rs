use super::param::Param;
use crate::module::display;
use proc_macro::TokenStream;
use quote::quote;

pub(crate) fn module_derive_impl(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let (generics, generics_ty, generics_where) = ast.generics.split_for_impl();

    let display_fn = display::display_fn(name);

    let param = Param::from_ast(ast);
    let num_params_fn = param.gen_num_params_fn();
    let visit = param.gen_visit_fn();
    let visit_mut = param.gen_visit_mut_fn();
    let devices_fn = param.gen_devices_fn();
    let to_device_fn = param.gen_to_device_fn();
    let state_fn = param.gen_state_fn();
    let load_fn = param.gen_load_fn();
    let inner_fn = param.gen_inner_fn();
    let from_inner_fn = param.gen_from_inner_fn();
    let detach_fn = param.gen_detach_fn();
    let clone_fn = param.gen_clone_fn();
    let generics_names_except_backend = generics_names_except_backend(&ast.generics);

    let gen = quote! {
        impl #generics burn::module::Module for #name #generics_ty #generics_where {
            type Backend=B;

            #devices_fn
            #to_device_fn

            #state_fn
            #load_fn

            #num_params_fn
            #detach_fn

            #visit
            #visit_mut
        }

        impl #generics burn::module::ADModule for #name #generics_ty where B: burn::tensor::backend::ADBackend, {
            type ADBackend=B;
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
