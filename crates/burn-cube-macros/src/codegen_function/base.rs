use proc_macro2::TokenStream;
use quote::ToTokens;

use super::{expr::codegen_expr, variable::codegen_local};
use crate::tracker::VariableTracker;

/// Codegen for a statement (generally one line)
/// Entry point of code generation
pub fn codegen_statement(
    statement: &syn::Stmt,
    loop_level: usize,
    variable_tracker: &mut VariableTracker,
) -> TokenStream {
    match statement {
        syn::Stmt::Local(local) => codegen_local(local, loop_level, variable_tracker),
        syn::Stmt::Expr(expr, semi) => {
            let expr = codegen_expr(expr, loop_level, variable_tracker).tokens;

            match semi {
                Some(_semi) => quote::quote!(
                    #expr;
                ),
                None => expr,
            }
        }
        _ => todo!("Codegen: statement {statement:?} not supported"),
    }
}

/// Codegen for a code block (a list of statements)
pub(crate) fn codegen_block(
    block: &syn::Block,
    loop_level: usize,
    variable_tracker: &mut VariableTracker,
) -> TokenStream {
    let mut statements = quote::quote!();

    for statement in block.stmts.iter() {
        statements.extend(codegen_statement(statement, loop_level, variable_tracker));
    }

    quote::quote! {
        {
            #statements
        }
    }
}

pub(crate) struct Codegen {
    pub tokens: proc_macro2::TokenStream,
    pub is_comptime: bool,
    pub array_indexing: Option<ArrayIndexing>,
}

pub(crate) struct ArrayIndexing {
    pub array: proc_macro2::TokenStream,
    pub index: proc_macro2::TokenStream,
}

impl From<proc_macro2::TokenStream> for Codegen {
    fn from(tokens: proc_macro2::TokenStream) -> Self {
        Self {
            tokens,
            is_comptime: false,
            array_indexing: None,
        }
    }
}

impl Codegen {
    pub fn new<S: Into<proc_macro2::TokenStream>>(tokens: S, is_comptime: bool) -> Self {
        Self {
            tokens: tokens.into(),
            is_comptime,
            array_indexing: None,
        }
    }

    pub fn split(self) -> (proc_macro2::TokenStream, bool) {
        (self.tokens, self.is_comptime)
    }
}

impl ToTokens for Codegen {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.extend(self.tokens.clone());
    }
    fn into_token_stream(self) -> TokenStream
    where
        Self: Sized,
    {
        self.tokens
    }
}
