use proc_macro2::{Ident, TokenStream};
use syn::Generics;

/// Basic trait to generate a record type based on the Module struct.
pub(crate) trait ModuleRecordCodegen {
    /// Generate the record type (i.e a struct)
    fn gen_record_type(&self, record_name: &Ident, generics: &Generics) -> TokenStream;
}
