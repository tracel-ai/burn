use proc_macro2::TokenStream;

use crate::{analysis::CodeAnalysis, codegen::base::codegen_expr};

use super::{base::codegen_block, operation::codegen_binary, variable::codegen_lit};

/// Codegen of for loops
/// Supports range:
/// for i in range(start, end, unroll) {...}
pub(crate) fn codegen_for_loop(
    for_loop: &syn::ExprForLoop,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    let i = &for_loop.pat;
    let block = codegen_block(&for_loop.body, loop_level + 1, variable_analyses);

    match for_loop.expr.as_ref() {
        syn::Expr::Call(call) => {
            let func_name = match call.func.as_ref() {
                syn::Expr::Path(path) => path
                    .path
                    .get_ident()
                    .expect("Codegen: func in for loop should have ident"),
                _ => todo!("Codegen: Only path call supported"),
            };

            if &func_name.to_string() == "range" {
                let mut args = quote::quote! {
                    context,
                };

                for argument in call.args.iter() {
                    let arg = codegen_expr(argument, loop_level, variable_analyses);
                    args.extend(quote::quote! { #arg, });
                }

                quote::quote! {
                    range_expand(#args |context, #i| #block);
                }
            } else {
                todo!("Codegen: Only range is supported")
            }
        }
        _ => todo!("Codegen: Only call is supported {for_loop:?}"),
    }
}

/// Codegen for condition of an if or a while
pub(crate) fn codegen_cond(
    cond: &syn::Expr,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    match cond {
        syn::Expr::Binary(expr) => codegen_binary(expr, loop_level, variable_analyses),
        syn::Expr::Lit(expr) => codegen_lit(expr),
        _ => todo!("{cond:?} cond not supported"),
    }
}

/// Codegen for break statement
pub(crate) fn codegen_break() -> TokenStream {
    quote::quote! {
        break_expand(context);
    }
}

/// Codegen for if and if/else statements
/// Supports:
/// if cond {...}
/// if cond {...} else {...}
pub(crate) fn codegen_if(
    expr_if: &syn::ExprIf,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    let cond = codegen_cond(&expr_if.cond, loop_level, variable_analyses);

    let then_block = codegen_block(&expr_if.then_branch, loop_level + 1, variable_analyses);

    if let Some((_, expr)) = &expr_if.else_branch {
        if let syn::Expr::Block(expr_block) = &**expr {
            let else_block = codegen_block(&expr_block.block, loop_level + 1, variable_analyses);

            quote::quote! {
                let _cond = #cond;
                if_else_expand(context, _cond, |context| #then_block, |context| #else_block);
            }
        } else {
            todo!("Analysis: Only block else expr is supported")
        }
    } else {
        quote::quote! {
            let _cond = #cond;
            if_expand(context, _cond, |context| #then_block);
        }
    }
}

/// Codegen of loop
pub(crate) fn codegen_loop(
    loop_expr: &syn::ExprLoop,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    let block = codegen_block(&loop_expr.body, loop_level + 1, variable_analyses);

    quote::quote! {
        loop_expand(context, |context| #block);
    }
}

/// Codegen for while loop
pub(crate) fn codegen_while_loop(
    while_loop: &syn::ExprWhile,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    let cond = codegen_cond(&while_loop.cond, loop_level + 1, variable_analyses);
    let block = codegen_block(&while_loop.body, loop_level + 1, variable_analyses);

    quote::quote! {
        while_loop_expand(context, |context| #cond, |context| #block);
    }
}
