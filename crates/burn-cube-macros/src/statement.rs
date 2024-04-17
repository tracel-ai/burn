use proc_macro2::TokenStream;

pub fn codegen_statement(statement: &syn::Stmt, loop_level: usize) -> TokenStream {
    match statement {
        syn::Stmt::Local(local) => codegen_local(local, loop_level),
        syn::Stmt::Item(_) => todo!(),
        syn::Stmt::Expr(expr, semi) => {
            let expr = codegen_expr(expr, loop_level);
            match semi {
                Some(_semi) => quote::quote!(
                    #expr;
                ),
                None => expr,
            }
        }
        syn::Stmt::Macro(_) => todo!(),
    }
}

fn codegen_local(local: &syn::Local, loop_level: usize) -> TokenStream {
    let init = local
        .init
        .as_ref()
        .expect("Can't use let without an initialization.");
    let ident = match &local.pat {
        syn::Pat::Ident(ident) => ident,
        _ => panic!("Only ident declaration is supported."),
    };
    let init = codegen_expr(&init.expr, loop_level);

    let let_tok = local.let_token;

    quote::quote! {
        #let_tok #ident = #init;
    }
}

fn codegen_expr(expr: &syn::Expr, loop_level: usize) -> TokenStream {
    match expr {
        syn::Expr::Binary(op) => codegen_binary(op, loop_level),
        syn::Expr::Path(path) => codegen_path(path, loop_level),
        syn::Expr::Call(call) => codegen_call(call, loop_level),
        syn::Expr::Lit(lit) => quote::quote! { #lit.into() },
        syn::Expr::Closure(closure) => codegen_closure(closure, loop_level),
        syn::Expr::Block(block) => codegen_expr_block(block, loop_level),
        syn::Expr::Assign(assign) => codegen_assign(assign, loop_level),
        syn::Expr::ForLoop(for_loop) => codegen_for_loop(for_loop, loop_level),
        syn::Expr::MethodCall(call) => codegen_expr_method_call(call),
        syn::Expr::Index(index) => codegen_expr_index(index, loop_level),
        _ => panic!("Unsupported {:?}", expr),
    }
}

fn codegen_expr_index(index: &syn::ExprIndex, loop_level: usize) -> TokenStream {
    let array = codegen_expr(&index.expr, loop_level);
    let index = codegen_expr(&index.index, loop_level);

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

fn codegen_for_loop(for_loop: &syn::ExprForLoop, loop_level: usize) -> TokenStream {
    let i = &for_loop.pat;
    let block = codegen_block(&for_loop.body, loop_level + 1);

    match for_loop.expr.as_ref() {
        syn::Expr::Call(call) => {
            let func_name = match call.func.as_ref() {
                syn::Expr::Path(path) => path.path.get_ident().unwrap(),
                _ => todo!("Only path call supported"),
            };

            if &func_name.to_string() == "range" {
                let mut args = quote::quote! {
                    context,
                };

                for argument in call.args.iter() {
                    let arg = codegen_expr(argument, loop_level);
                    args.extend(quote::quote! { #arg, });
                }

                quote::quote! {
                    range_expand(#args |context, #i| #block);
                }
            } else {
                todo!("Only range is supported")
            }
        }
        _ => todo!("Only call is supported {for_loop:?}"),
    }
}

fn codegen_assign(assign: &syn::ExprAssign, loop_level: usize) -> TokenStream {
    if let syn::Expr::Index(index) = assign.left.as_ref() {
        let array = codegen_expr(&index.expr, loop_level);
        let index = codegen_expr(&index.index, loop_level);
        let value = codegen_expr(&assign.right, loop_level);

        return quote::quote! {
            {
            let _array = #array;
            let _index = #index;
            let _value = #value;
            burn_cube::index_assign::expand(context, _array, _index, _value)
            }
        };
    };

    let lhs = codegen_expr(&assign.left, loop_level);
    let rhs = codegen_expr(&assign.right, loop_level);

    quote::quote! {
        {
            let _assign_lhs = #lhs;
            let _assign_rhs = #rhs;
            #lhs = burn_cube::assign::expand(context, _assign_lhs, _assign_rhs)
        }
    }
}

fn codegen_block(block: &syn::Block, loop_level: usize) -> TokenStream {
    let mut statements = quote::quote!();

    for statement in block.stmts.iter() {
        statements.extend(codegen_statement(statement, loop_level));
    }

    quote::quote! {
        {
            #statements
        }
    }
}

fn codegen_expr_block(block: &syn::ExprBlock, loop_level: usize) -> TokenStream {
    codegen_block(&block.block, loop_level)
}

fn codegen_closure(closure: &syn::ExprClosure, loop_level: usize) -> TokenStream {
    let mut inputs = quote::quote! {};
    for input in closure.inputs.iter() {
        let ident = match input {
            syn::Pat::Ident(ident) => &ident.ident,
            _ => panic!("Unsupported {:?}", input),
        };
        inputs.extend(quote::quote! {
            #ident,
        });
    }

    let body = codegen_expr(closure.body.as_ref(), loop_level);

    quote::quote! {
        |context, #inputs| #body
    }
}

fn codegen_call(call: &syn::ExprCall, loop_level: usize) -> TokenStream {
    let func_name = match call.func.as_ref() {
        syn::Expr::Path(path) => path.path.get_ident().unwrap(),
        _ => todo!("Only path call supported"),
    };

    let mut args = quote::quote! {
        context,
    };

    let func_name_expand =
        syn::Ident::new(format!("{func_name}_expand").as_str(), func_name.span());

    for argument in call.args.iter() {
        let arg = codegen_expr(argument, loop_level);
        args.extend(quote::quote! { #arg, });
    }

    quote::quote! {
        #func_name_expand(#args)
    }
}

fn codegen_path(path: &syn::ExprPath, loop_level: usize) -> TokenStream {
    let ident = path
        .path
        .get_ident()
        .expect("Only ident path are supported.");

    // TODO: Check in the following statements if the ident is overriden, or reused.
    let will_be_used_again = true;

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

fn codegen_binary(binary: &syn::ExprBinary, loop_level: usize) -> TokenStream {
    let lhs = codegen_expr(&binary.left, loop_level);
    let rhs = codegen_expr(&binary.right, loop_level);

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
        _ => todo!("{:?}", binary.op),
    }
}
