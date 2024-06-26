use crate::tracker::VariableTracker;
use proc_macro2::TokenStream;

use super::{
    base::{codegen_block, Codegen},
    branch::{
        codegen_break, codegen_for_loop, codegen_if, codegen_loop, codegen_return,
        codegen_while_loop,
    },
    function::{codegen_call, codegen_closure, codegen_expr_method_call},
    operation::{codegen_binary, codegen_unary},
    variable::{
        codegen_array_lit, codegen_assign, codegen_field, codegen_index, codegen_lit,
        codegen_path_var, codegen_struct,
    },
};

/// Codegen for expressions
pub(crate) fn codegen_expr(
    expr: &syn::Expr,
    loop_level: usize,
    variable_tracker: &mut VariableTracker,
) -> Codegen {
    match expr {
        syn::Expr::Call(call) => codegen_call(call, loop_level, variable_tracker),
        syn::Expr::Paren(paren) => codegen_expr(&paren.expr, loop_level, variable_tracker),
        _ => {
            let mut array_indexing = None;
            let tokens = match expr {
                syn::Expr::Path(path) => {
                    return codegen_path_var(path, loop_level, variable_tracker)
                }
                syn::Expr::Binary(op) => return codegen_binary(op, loop_level, variable_tracker),
                syn::Expr::Unary(op) => return codegen_unary(op, loop_level, variable_tracker),
                syn::Expr::Lit(lit) => codegen_lit(lit),
                syn::Expr::Closure(closure) => {
                    codegen_closure(closure, loop_level, variable_tracker)
                }
                syn::Expr::Block(block) => codegen_expr_block(block, loop_level, variable_tracker),
                syn::Expr::Assign(assign) => codegen_assign(assign, loop_level, variable_tracker),
                syn::Expr::ForLoop(for_loop) => {
                    codegen_for_loop(for_loop, loop_level, variable_tracker)
                }
                syn::Expr::While(while_loop) => {
                    codegen_while_loop(while_loop, loop_level, variable_tracker)
                }
                syn::Expr::Loop(loop_expr) => codegen_loop(loop_expr, loop_level, variable_tracker),
                syn::Expr::Break(_) => codegen_break(),
                syn::Expr::Return(return_expr) => codegen_return(return_expr),
                syn::Expr::If(expr_if) => codegen_if(expr_if, loop_level, variable_tracker),
                syn::Expr::MethodCall(call) => {
                    codegen_expr_method_call(call, loop_level, variable_tracker)
                }
                syn::Expr::Index(index) => {
                    let codegen = codegen_index(index, loop_level, variable_tracker);
                    array_indexing = codegen.array_indexing;
                    codegen.tokens
                }
                syn::Expr::Array(array) => codegen_array_lit(array),
                syn::Expr::Reference(reference) => {
                    codegen_ref(reference, loop_level, variable_tracker)
                }
                syn::Expr::Field(field) => codegen_field(field, loop_level, variable_tracker),
                syn::Expr::Struct(struct_) => codegen_struct(struct_, loop_level, variable_tracker),
                syn::Expr::Range(range) => syn::Error::new_spanned(
                    range,
                    "Range is not supported, use [range](cubecl::prelude::range) instead.",
                )
                .to_compile_error(),
                _ => {
                    syn::Error::new_spanned(expr, "Expression is not supported").to_compile_error()
                }
            };

            let mut codegen = Codegen::new(tokens, false);
            codegen.array_indexing = array_indexing;
            codegen
        }
    }
}

/// Codegen for an expression containing a block
pub(crate) fn codegen_expr_block(
    block: &syn::ExprBlock,
    loop_level: usize,
    variable_tracker: &mut VariableTracker,
) -> TokenStream {
    codegen_block(&block.block, loop_level, variable_tracker)
}

pub(crate) fn codegen_ref(
    reference: &syn::ExprReference,
    loop_level: usize,
    variable_tracker: &mut VariableTracker,
) -> TokenStream {
    let inner = codegen_expr(&reference.expr, loop_level, variable_tracker);
    quote::quote! { & #inner }
}
