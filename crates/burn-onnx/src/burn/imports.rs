use proc_macro2::TokenStream;
use quote::quote;
use std::collections::HashSet;

/// Keep track of imported modules.
#[derive(Debug, Default)]
pub struct BurnImports {
    imports: HashSet<String>,
}

impl BurnImports {
    /// Register an import type.
    ///
    /// # Notes
    ///
    /// Each import statement will be generated just once no matter how many times it was
    /// registered.
    pub fn register<S: Into<String>>(&mut self, import: S) {
        self.imports.insert(import.into());
    }

    /// Generate the import tokens.
    pub fn codegen(&self) -> TokenStream {
        let mut import_tokens = vec![];

        // Sort imports for deterministic output
        let mut sorted_imports: Vec<_> = self.imports.iter().collect();
        sorted_imports.sort();

        for import in sorted_imports {
            let path: syn::Path =
                syn::parse_str(import).expect("Unable to parse input string as a path");

            import_tokens.push(quote! { #path });
        }

        quote! {
            use burn::prelude::*;

            #(use #import_tokens;)*
        }
    }
}
