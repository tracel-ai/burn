use proc_macro2::Ident;
use quote::quote;
use syn::{parse2, Generics, WhereClause, WherePredicate};

#[derive(new)]
pub struct GenericsHelper {
    generics: Generics,
}

impl GenericsHelper {
    pub fn add_predicate(&mut self, predicate: WherePredicate) {
        let where_clause: WhereClause = match &self.generics.where_clause {
            Some(val) => parse2(quote! {
                #val
                    #predicate,
            })
            .unwrap(),
            None => parse2(quote! {
                where
                    #predicate,
            })
            .unwrap(),
        };
        self.generics.where_clause = Some(where_clause);
    }

    pub fn idents_except_backend(&self) -> Vec<Ident> {
        self.generics
            .type_params()
            .into_iter()
            .map(|tp| tp.ident.clone())
            .filter(|ident| ident != "B")
            .collect()
    }

    pub fn fetch_backend_trait(&self) -> proc_macro2::TokenStream {
        static BACKEND_TRAIT_COMPILATION_ERROR_MSG: &str =
            "Modules should be generic over a backend.
    - The generic argument named `B` should have its first trait bound being a backend trait.
    - The default backend trait is `burn::tensor::backend::Backend`.
    - Any backend trait is supported.";

        for param in self.generics.params.iter() {
            if let syn::GenericParam::Type(ty) = &param {
                if ty.ident == "B" {
                    let bound = ty
                        .bounds
                        .first()
                        .expect(BACKEND_TRAIT_COMPILATION_ERROR_MSG);

                    return quote! {
                        #bound
                    };
                }
            }
        }

        panic!("{BACKEND_TRAIT_COMPILATION_ERROR_MSG}");
    }
}
