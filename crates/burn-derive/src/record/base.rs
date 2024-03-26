use super::{
    codegen::generate_record,
    item::{codegen_enum::EnumRecordItemCodegen, codegen_struct::StructRecordItemCodegen},
};

pub(crate) fn derive_impl(ast: &syn::DeriveInput) -> proc_macro::TokenStream {
    match &ast.data {
        syn::Data::Struct(_) => generate_record::<StructRecordItemCodegen>(ast),
        syn::Data::Enum(_) => generate_record::<EnumRecordItemCodegen>(ast),
        syn::Data::Union(_) => panic!("Union modules aren't supported yet."),
    }
    .into()
}
