use quote::quote;

use super::codegen_struct::StructRecordCodegen;

pub(crate) fn derive_impl(ast: &syn::DeriveInput) -> proc_macro::TokenStream {
    match &ast.data {
        syn::Data::Struct(_) => {
            let record_gen = StructRecordCodegen::from_ast(ast);
            let item_struct = record_gen.gen_record_type();
            let record_impl = record_gen.gen_impl_record();

            quote! {
                #item_struct
                #record_impl
            }
        }
        syn::Data::Enum(_data) => {
            panic!("Enum records aren't supported yet.")
        }
        syn::Data::Union(_) => panic!("Union modules aren't supported yet."),
    }
    .into()
}
