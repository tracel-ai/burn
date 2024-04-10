use proc_macro2::TokenStream;

pub fn parse_statement(statement: &syn::Stmt) -> TokenStream {
    match statement {
        syn::Stmt::Local(local) => parse_local(local),
        syn::Stmt::Item(_) => todo!(),
        syn::Stmt::Expr(expr, _) => parse_expr(expr),
        syn::Stmt::Macro(_) => todo!(),
    }
}

fn parse_local(local: &syn::Local) -> TokenStream {
    let init = local
        .init
        .as_ref()
        .expect("Can't use let without an initialization.");
    let ident = match &local.pat {
        syn::Pat::Ident(ident) => &ident.ident,
        _ => panic!("Only ident declaration is supported."),
    };
    let init = parse_expr(&init.expr);

    quote::quote! {
        let #ident = #init;
    }
}

fn parse_expr(expr: &syn::Expr) -> TokenStream {
    match expr {
        syn::Expr::Binary(op) => parse_binary(op),
        syn::Expr::Path(path) => parse_path(path),
        _ => panic!("Unsupported {:?}", expr),
    }
}

fn parse_path(path: &syn::ExprPath) -> TokenStream {
    let ident = path
        .path
        .get_ident()
        .expect("Only ident path are supported.");

    quote::quote! {
        #ident
    }
}

fn parse_binary(binary: &syn::ExprBinary) -> TokenStream {
    let lhs = &binary.left;
    let rhs = &binary.right;

    match binary.op {
        syn::BinOp::Add(_) => quote::quote! {
            burn_cube::add::expand(context, #lhs, #rhs)
        },
        syn::BinOp::Sub(_) => quote::quote! {
            burn_cube::sub::expand(context, #lhs, #rhs)
        },
        syn::BinOp::Mul(_) => quote::quote! {
            burn_cube::mul::expand(context, #lhs, #rhs)
        },
        syn::BinOp::Div(_) => quote::quote! {
            burn_cube::div::expand(context, #lhs, #rhs)
        },
        _ => todo!("{:?}", binary.op),
    }
}
