use proc_macro::TokenStream;

pub(crate) mod module;

use module::param::Param;

#[proc_macro_derive(Module)]
pub fn module_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    module::module_derive_impl(&ast)
}
