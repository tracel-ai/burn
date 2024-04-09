use proc_macro::TokenStream;

/// Derive macro for the module.
#[proc_macro_attribute]
pub fn cube(_attr: TokenStream, tokens: TokenStream) -> TokenStream {
    let func: syn::ItemFn = syn::parse(tokens).unwrap();
    let signature = expand_sig(&func.sig);
    let mut body = quote::quote! {};

    for statement in func.block.stmts.iter() {
        let mut statement_gen = quote::quote! {};
        let mut skipped = false;
        match statement {
            syn::Stmt::Expr(expr, _) => match expr {
                syn::Expr::Binary(binary) => {
                    let lhs = &binary.left;
                    let rhs = &binary.right;

                    match binary.op {
                        syn::BinOp::Add(_add) => {
                            skipped = true;
                            statement_gen.extend(quote::quote! {
                                burn_cube::float_add_expand(context, #lhs, #rhs)
                            });
                        }
                        _ => (),
                    }
                }
                _ => (),
            },
            syn::Stmt::Local(local) => {
                skipped = true;
                let output = &local.pat;
                match &output {
                    syn::Pat::Const(_) => todo!(),
                    syn::Pat::Ident(ident) => panic!("path ident {:?}", ident),
                    syn::Pat::Lit(_) => todo!(),
                    syn::Pat::Macro(_) => todo!(),
                    syn::Pat::Or(_) => todo!(),
                    syn::Pat::Paren(_) => todo!(),
                    syn::Pat::Path(_) => todo!(),
                    syn::Pat::Range(_) => todo!(),
                    syn::Pat::Reference(_) => todo!(),
                    syn::Pat::Rest(_) => todo!(),
                    syn::Pat::Slice(_) => todo!(),
                    syn::Pat::Struct(_) => todo!(),
                    syn::Pat::Tuple(_) => todo!(),
                    syn::Pat::TupleStruct(_) => todo!(),
                    syn::Pat::Type(_) => todo!(),
                    syn::Pat::Verbatim(_) => todo!(),
                    syn::Pat::Wild(_) => todo!(),
                    _ => todo!(),
                }
            }
            syn::Stmt::Item(_) => panic!("item"),
            syn::Stmt::Macro(_) => panic!("Macros not supported."),
        };

        if !skipped {
            body.extend(quote::quote! {
                #statement
            });
        } else {
            body.extend(statement_gen);
        }
    }

    let code = quote::quote! {
        #func

        #signature {
            #body
        }
    }
    .into();

    // panic!("{code}");
    code
}

fn expand_sig(sig: &syn::Signature) -> proc_macro2::TokenStream {
    let mut inputs = quote::quote!();

    for input in &sig.inputs {
        match input {
            syn::FnArg::Typed(pat) => {
                let ty = &pat.ty;
                let ident = pat.pat.clone();

                inputs.extend(quote::quote! {
                    #ident: <#ty as burn_cube::CubeVariable>::Variable,
                });
            }
            _ => todo!(),
        }
    }

    let mut output = quote::quote!();

    match &sig.output {
        syn::ReturnType::Default => output.extend(quote::quote! { ()}),
        syn::ReturnType::Type(_, ty) => {
            output.extend(quote::quote! {
                <#ty as burn_cube::CubeVariable>::Variable
            });
        }
    }

    let ident = &sig.ident;
    let ident = syn::Ident::new(format!("{ident}_expand").as_str(), ident.span());

    quote::quote! {
        pub fn #ident(context: &mut burn_cube::CodegenContext<'_>, #inputs) -> #output
    }
    .into()
}
