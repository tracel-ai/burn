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
        syn::Data::Struct(_) => match StructModuleCodegen::from_ast(ast) {
            Ok(struct_codegen) => {
                if has_backend {
                    generate_module_standard(ast, struct_codegen)
                } else {
                    generate_module_const(ast)
                }
            }
            Err(err) => err.to_compile_error(),
        },
        syn::Data::Enum(_data) => match EnumModuleCodegen::from_ast(ast) {
            Ok(enum_codegen) => {
                if has_backend {
                    generate_module_standard(ast, enum_codegen)
                } else {
                    generate_module_const(ast)
                }
            }
            Err(err) => err.to_compile_error(),
        },
        syn::Data::Union(_) => {
            syn::Error::new_spanned(ast, "Union modules aren't supported").to_compile_error()
        }
    }
    .into()
}
