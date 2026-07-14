use proc_macro2::Ident;
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

    pub fn types(&self) -> Vec<Ident> {
        self.generics
            .type_params()
            .map(|tp| tp.ident.clone())
            .collect()
    }
}
