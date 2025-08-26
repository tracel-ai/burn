use proc_macro2::{Ident, TokenStream};
use syn::{
    GenericArgument, Generics, TypePath,
    visit_mut::{self, VisitMut},
};

/// Basic trait to be implemented for record generation.
pub(crate) trait RecordItemCodegen {
    /// Initialize the record item.
    fn from_ast(ast: &syn::DeriveInput) -> Self;
    /// Generate the record item type.
    fn gen_item_type(
        &self,
        item_name: &Ident,
        generics: &Generics,
        has_backend: bool,
    ) -> TokenStream;
    /// Generate the into_item function.
    fn gen_into_item(&self, item_name: &Ident) -> TokenStream;
    /// Generate the from item function.
    fn gen_from_item(&self) -> TokenStream;
}

pub(crate) struct ReplaceBackend<'a> {
    pub(crate) replacement: &'a TypePath,
}

impl<'a> VisitMut for ReplaceBackend<'a> {
    fn visit_type_path_mut(&mut self, tp: &mut syn::TypePath) {
        if tp.qself.is_none() && tp.path.segments.len() == 1 && tp.path.segments[0].ident == "B" {
            *tp = self.replacement.clone();
            return;
        }

        // Otherwise recurse and also scrub generic args like Foo<B, T>
        for seg in tp.path.segments.iter_mut() {
            if let syn::PathArguments::AngleBracketed(ab) = &mut seg.arguments {
                for arg in ab.args.iter_mut() {
                    if let GenericArgument::Type(ty) = arg {
                        self.visit_type_mut(ty);
                    }
                }
            }
        }

        visit_mut::visit_type_path_mut(self, tp);
    }
}
