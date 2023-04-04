use super::generator::RecordGenerator;
use proc_macro::TokenStream;
use quote::quote;

pub(crate) fn record_derive_impl(ast: &syn::DeriveInput) -> TokenStream {
    let record_gen = RecordGenerator::from_ast(ast);

    let item_struct = record_gen.gen_record_item_struct();
    let record_impl = record_gen.gen_impl_record();

    let gen = quote! {
        #item_struct

        #record_impl
    };

    gen.into()
}
