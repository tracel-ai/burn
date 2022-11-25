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
    let update_params_fn = param.gen_update_params_fn();
    let load_optim_state = param.gen_load_optim_state_fn();
    let register_optim_state = param.gen_register_optim_state_fn();
    let devices_fn = param.gen_devices_fn();
    let to_device_fn = param.gen_to_device_fn();
    let state_fn = param.gen_state_fn();
    let load_fn = param.gen_load_fn();
    let inner_fn = param.gen_inner_fn();
    let detach_fn = param.gen_detach_fn();
    let generics_names_except_backend = generics_names_except_backend(&ast.generics);

    let gen = quote! {
        impl #generics burn::module::Module for #name #generics_ty #generics_where {
            type Backend=B;

            #devices_fn
            #to_device_fn
            #detach_fn

            #state_fn
            #load_fn

            #num_params_fn
            #update_params_fn

            #load_optim_state
            #register_optim_state
        }

        impl #generics burn::module::ADModule for #name #generics_ty where B: burn::tensor::backend::ADBackend, {
            type ADBackend=B;
            type InnerModule=#name<B::InnerBackend, #generics_names_except_backend>;

            #inner_fn
        }

        impl #generics std::fmt::Display for #name #generics_ty #generics_where {
            #display_fn
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
