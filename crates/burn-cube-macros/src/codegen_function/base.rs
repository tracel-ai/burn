use proc_macro2::TokenStream;

use crate::tracker::VariableTracker;

use super::{
    branch::{
        codegen_break, codegen_for_loop, codegen_if, codegen_loop, codegen_return,
        codegen_while_loop,
    },
    function::{codegen_call, codegen_closure, codegen_expr_method_call},
    operation::{codegen_binary, codegen_unary},
    variable::{
        codegen_array_lit, codegen_assign, codegen_field, codegen_index, codegen_lit,
        codegen_local, codegen_path_rhs, codegen_struct,
    },
};

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
            let expr = codegen_expr(expr, loop_level, variable_tracker);
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

pub(crate) fn codegen_expr(
    expr: &syn::Expr,
    loop_level: usize,
    variable_tracker: &mut VariableTracker,
) -> TokenStream {
    codegen_expr_with_comptime(expr, loop_level, variable_tracker).0
}

/// Codegen for expressions
/// There are many variants of expression, treated differently
pub(crate) fn codegen_expr_with_comptime(
    expr: &syn::Expr,
    loop_level: usize,
    variable_tracker: &mut VariableTracker,
) -> (TokenStream, bool) {
    match expr {
        syn::Expr::Call(call) => codegen_call(call, loop_level, variable_tracker),
        syn::Expr::Paren(paren) => {
            codegen_expr_with_comptime(&paren.expr, loop_level, variable_tracker)
        }
        _ => {
            let tokens = match expr {
                syn::Expr::Binary(op) => codegen_binary(op, loop_level, variable_tracker),
                syn::Expr::Path(path) => codegen_path_rhs(path, loop_level, variable_tracker),
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
                syn::Expr::Index(index) => codegen_index(index, loop_level, variable_tracker),
                syn::Expr::Array(array) => codegen_array_lit(array),
                syn::Expr::Reference(reference) => {
                    codegen_ref(reference, loop_level, variable_tracker)
                }
                syn::Expr::Unary(op) => codegen_unary(op, loop_level, variable_tracker),
                syn::Expr::Field(field) => codegen_field(field, loop_level, variable_tracker),
                syn::Expr::Struct(struct_) => codegen_struct(struct_, loop_level, variable_tracker),
                _ => panic!("Codegen: Unsupported {:?}", expr),
            };

            (tokens, false)
        }
    }
}
