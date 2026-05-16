use super::{
    codegen::{generate_module_standard, generate_module_stateless},
    codegen_enum::EnumModuleCodegen,
    codegen_struct::StructModuleCodegen,
};
use proc_macro::TokenStream;

pub(crate) fn derive_impl(ast: &syn::DeriveInput) -> TokenStream {
    match &ast.data {
        syn::Data::Struct(_) => match StructModuleCodegen::from_ast(ast) {
            Ok(struct_codegen) => {
                if struct_codegen.has_module_fields() {
                    generate_module_standard(ast, struct_codegen)
                } else {
                    generate_module_stateless(ast, struct_codegen)
                }
            }
            Err(err) => err.to_compile_error(),
        },
        syn::Data::Enum(_data) => match EnumModuleCodegen::from_ast(ast) {
            Ok(enum_codegen) => {
                if enum_codegen.has_module_fields() {
                    generate_module_standard(ast, enum_codegen)
                } else {
                    generate_module_stateless(ast, enum_codegen)
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
