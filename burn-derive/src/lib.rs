use proc_macro::TokenStream;
use quote::quote;

pub(crate) mod field;
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

    let gen = quote! {
        impl #generics burn::module::Module<B> for #name #generics #generics_where {
            fn save(&self) {
                todo!()
            }
            #num_params_fn
        }

        impl #generics std::fmt::Display for #name #generics #generics_where {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "display")
            }
        }
    };

    // panic!("{}", gen);
    gen.into()
}
