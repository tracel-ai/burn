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

        for import in self.imports.iter() {
            let path: syn::Path =
                syn::parse_str(import).expect("Unable to parse input string as a path");

            import_tokens.push(quote! { #path });
        }

        quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

            #(use #import_tokens;)*
        }
    }
}
