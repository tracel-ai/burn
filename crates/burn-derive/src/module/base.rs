use super::{
    codegen::{generate_module_const, generate_module_standard},
    codegen_enum::EnumModuleCodegen,
    codegen_struct::StructModuleCodegen,
};
use proc_macro::TokenStream;

pub(crate) fn derive_impl(ast: &syn::DeriveInput) -> TokenStream {
    let has_backend = ast
        .generics
        .type_params()
        .map(|param| param.ident == "B")
        .reduce(|accum, is_backend| is_backend || accum)
        .unwrap_or(false);

    match &ast.data {
        syn::Data::Struct(_) => {
            if has_backend {
                generate_module_standard(ast, StructModuleCodegen::from_ast(ast))
            } else {
                generate_module_const(ast)
            }
        }
        syn::Data::Enum(_data) => {
            if has_backend {
                generate_module_standard(ast, EnumModuleCodegen::from_ast(ast))
            } else {
                generate_module_const(ast)
            }
        }
        syn::Data::Union(_) => panic!("Union modules aren't supported yet."),
    }
    .into()
}
