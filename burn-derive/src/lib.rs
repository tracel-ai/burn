use proc_macro::TokenStream;
use quote::quote;

pub(crate) mod field;

mod display;
mod param;

use param::Param;

#[proc_macro_derive(Module)]
pub fn module_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    module_derive_impl(&ast)
}

fn module_derive_impl(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let generics = ast.generics.clone();
    let generics_where = ast.generics.where_clause.clone();

    let param = Param::from_ast(ast);

    let num_params_fn = param.gen_num_params_fn();
    let update_params_fn = param.gen_update_params_fn();
    let devices_fn = param.gen_devices_fn();
    let to_device_fn = param.gen_to_device_fn();
    let display_fn = display::display_fn(name);

    let gen = quote! {
        impl #generics burn::module::Module<B> for #name #generics #generics_where {
            #num_params_fn
            #update_params_fn
            #devices_fn
            #to_device_fn
        }


        impl #generics std::fmt::Display for #name #generics #generics_where {
            #display_fn
        }
    };

    // panic!("{}", gen);
    gen.into()
}
