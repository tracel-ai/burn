use proc_macro2::Ident;
use quote::quote;
use syn::{Generics, WhereClause, WherePredicate, parse_quote};

#[derive(new)]
pub struct GenericsHelper {
    pub(crate) generics: Generics,
}

impl GenericsHelper {
    pub fn add_predicate(&mut self, predicate: WherePredicate) {
        let where_clause: WhereClause = match &self.generics.where_clause {
            Some(val) => parse_quote! {
                #val
                    #predicate,
            },
            None => parse_quote! {
                where
                    #predicate,
            },
        };
        self.generics.where_clause = Some(where_clause);
    }

    pub fn consts(&self) -> Vec<Ident> {
        self.generics
            .const_params()
            .map(|c| c.ident.clone())
            .collect()
    }

    pub fn types(&self) -> Vec<Ident> {
        self.generics
            .type_params()
            .map(|tp| tp.ident.clone())
            .collect()
    }

    pub fn fetch_backend_trait(&self) -> proc_macro2::TokenStream {
        static BACKEND_TRAIT_COMPILATION_ERROR_MSG: &str =
            "Modules should be generic over a backend.
    - The generic argument named `B` should have its first trait bound being a backend trait.
    - The default backend trait is `burn::tensor::backend::Backend`.
    - Any backend trait is supported.";

        for param in self.generics.params.iter() {
            if let syn::GenericParam::Type(ty) = &param
                && ty.ident == "B"
            {
                let bound = ty
                    .bounds
                    .first()
                    .expect(BACKEND_TRAIT_COMPILATION_ERROR_MSG);

                return quote! {
                    #bound
                };
            }
        }

        panic!("{BACKEND_TRAIT_COMPILATION_ERROR_MSG}");
    }
}
