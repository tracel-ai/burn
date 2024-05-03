use proc_macro2::TokenStream;
use syn::token::If;

use crate::analysis::CodeAnalysis;

pub fn codegen_statement(
    statement: &syn::Stmt,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    match statement {
        syn::Stmt::Local(local) => codegen_local(local, loop_level, variable_analyses),
        syn::Stmt::Expr(expr, semi) => {
            let expr = codegen_expr(expr, loop_level, variable_analyses);
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

fn codegen_local(
    local: &syn::Local,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    let init = local
        .init
        .as_ref()
        .expect("Can't use let without an initialization.");

    let init = codegen_expr(&init.expr, loop_level, variable_analyses);

    let let_tok = local.let_token;

    if let syn::Pat::Wild(_) = &local.pat {
        return quote::quote! {
            #let_tok _ = #init;
        };
    }

    let ident = match &local.pat {
        syn::Pat::Ident(ident) => ident,
        syn::Pat::Type(pat_type) => match &*pat_type.pat {
            syn::Pat::Ident(pat_ident) => pat_ident,
            _ => todo!("Codegen: Unsupported typed path {:?}", pat_type.pat),
        },
        syn::Pat::Wild(_) => unreachable!(),
        _ => todo!("Codegen: Declaration {:?} is unsupported.", local.pat),
    };
    quote::quote! {
        #let_tok #ident = #init;
    }
}

fn codegen_expr(
    expr: &syn::Expr,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    match expr {
        syn::Expr::Binary(op) => codegen_binary(op, loop_level, variable_analyses),
        syn::Expr::Path(path) => codegen_path(path, loop_level, variable_analyses),
        syn::Expr::Call(call) => codegen_call(call, loop_level, variable_analyses),
        syn::Expr::Lit(lit) => codegen_lit(lit),
        syn::Expr::Closure(closure) => codegen_closure(closure, loop_level, variable_analyses),
        syn::Expr::Block(block) => codegen_expr_block(block, loop_level, variable_analyses),
        syn::Expr::Assign(assign) => codegen_assign(assign, loop_level, variable_analyses),
        syn::Expr::ForLoop(for_loop) => codegen_for_loop(for_loop, loop_level, variable_analyses),
        syn::Expr::While(while_loop) => {
            codegen_while_loop(while_loop, loop_level, variable_analyses)
        }
        syn::Expr::If(expr_if) => codegen_if(expr_if, loop_level, variable_analyses),
        syn::Expr::MethodCall(call) => codegen_expr_method_call(call),
        syn::Expr::Index(index) => codegen_expr_index(index, loop_level, variable_analyses),
        _ => panic!("Codegen: Unsupported {:?}", expr),
    }
}

fn codegen_lit(lit: &syn::ExprLit) -> TokenStream {
    quote::quote! { #lit.into() }
}

fn codegen_expr_index(
    index: &syn::ExprIndex,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    let array = codegen_expr(&index.expr, loop_level, variable_analyses);
    let index = codegen_expr(&index.index, loop_level, variable_analyses);

    quote::quote! {
        {
        let _array = #array;
        let _index = #index;
        burn_cube::index::expand(context, _array, _index)
        }
    }
}

fn codegen_expr_method_call(call: &syn::ExprMethodCall) -> TokenStream {
    quote::quote!( #call )
}

fn codegen_for_loop(
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

fn codegen_cond(
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

fn codegen_if(
    expr_if: &syn::ExprIf,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    if expr_if.else_branch.is_some() {
        todo!("Codegen: else branch not supported");
    }

    let cond = codegen_cond(&expr_if.cond, loop_level, variable_analyses);

    let block = codegen_block(&expr_if.then_branch, loop_level + 1, variable_analyses);

    quote::quote! {
        let _cond = #cond;
        if_expand(context, _cond, |context| #block);
    }
}

fn codegen_while_loop(
    while_loop: &syn::ExprWhile,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    let block = codegen_block(&while_loop.body, loop_level + 1, variable_analyses);

    let cond = codegen_cond(&while_loop.cond, loop_level, variable_analyses);

    quote::quote! {
        loop_expand(context, |context| #cond, |context| #block);
    }
}

fn codegen_assign(
    assign: &syn::ExprAssign,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    if let syn::Expr::Index(index) = assign.left.as_ref() {
        let array = codegen_expr(&index.expr, loop_level, variable_analyses);
        let index = codegen_expr(&index.index, loop_level, variable_analyses);
        let value = codegen_expr(&assign.right, loop_level, variable_analyses);

        return quote::quote! {
            {
            let _array = #array;
            let _index = #index;
            let _value = #value;
            burn_cube::index_assign::expand(context, _array, _index, _value)
            }
        };
    };

    let lhs = codegen_expr(&assign.left, loop_level, variable_analyses);
    let rhs = codegen_expr(&assign.right, loop_level, variable_analyses);

    quote::quote! {
        {
            let _assign_lhs = #lhs;
            let _assign_rhs = #rhs;
            #lhs = burn_cube::assign::expand(context, _assign_lhs, _assign_rhs)
        }
    }
}

fn codegen_block(
    block: &syn::Block,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    let mut statements = quote::quote!();

    for statement in block.stmts.iter() {
        statements.extend(codegen_statement(statement, loop_level, variable_analyses));
    }

    quote::quote! {
        {
            #statements
        }
    }
}

fn codegen_expr_block(
    block: &syn::ExprBlock,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    codegen_block(&block.block, loop_level, variable_analyses)
}

fn codegen_closure(
    closure: &syn::ExprClosure,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    let mut inputs = quote::quote! {};
    for input in closure.inputs.iter() {
        let ident = match input {
            syn::Pat::Ident(ident) => &ident.ident,
            _ => panic!("Codegen: Unsupported {:?}", input),
        };
        inputs.extend(quote::quote! {
            #ident,
        });
    }

    let body = codegen_expr(closure.body.as_ref(), loop_level, variable_analyses);

    quote::quote! {
        |context, #inputs| #body
    }
}

fn codegen_call(
    call: &syn::ExprCall,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    let func_name = match call.func.as_ref() {
        syn::Expr::Path(path) => path
            .path
            .get_ident()
            .expect("Codegen: func called path should have ident"),
        _ => todo!("Codegen: Only path call supported"),
    };

    let mut args = quote::quote! {
        context,
    };

    let func_name_expand =
        syn::Ident::new(format!("{func_name}_expand").as_str(), func_name.span());

    for argument in call.args.iter() {
        let arg = codegen_expr(argument, loop_level, variable_analyses);
        args.extend(quote::quote! { #arg, });
    }

    quote::quote! {
        #func_name_expand(#args)
    }
}

fn codegen_path(
    path: &syn::ExprPath,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    let ident = path
        .path
        .get_ident()
        .expect("Codegen: Only ident path are supported.");

    let will_be_used_again = variable_analyses.should_clone(ident, loop_level);

    if will_be_used_again {
        quote::quote! {
            #ident.clone()
        }
    } else {
        quote::quote! {
            #ident
        }
    }
}

fn codegen_binary(
    binary: &syn::ExprBinary,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    let lhs = codegen_expr(&binary.left, loop_level, variable_analyses);
    let rhs = codegen_expr(&binary.right, loop_level, variable_analyses);

    match binary.op {
        syn::BinOp::Add(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::add::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::Sub(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::sub::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::Mul(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::mul::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::Div(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::div::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::Rem(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::rem::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::Ne(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::ne::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::Gt(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::gt::expand(context, _lhs, _rhs)
            }
        },
        _ => todo!("Codegen: unsupported op {:?}", binary.op),
    }
}
