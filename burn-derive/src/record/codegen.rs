use proc_macro2::{Ident, TokenStream};
use syn::Generics;

/// Basic trait to be implemented for record generation.
pub(crate) trait RecordItemCodegen {
    /// Generate the record item type (i.e a struct)
    fn gen_item_type(&self, item_name: &Ident, generics: &Generics) -> TokenStream;
    /// Generate the into_item function.
    fn gen_into_item(&self, item_name: &Ident) -> TokenStream;
    /// Generate the from item function.
    fn gen_from_item(&self) -> TokenStream;
}
