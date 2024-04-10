mod statement;
use statement::parse_statement;

use proc_macro::TokenStream;

/// Derive macro for the module.
#[proc_macro_attribute]
pub fn cube(_attr: TokenStream, tokens: TokenStream) -> TokenStream {
    let func: syn::ItemFn = syn::parse(tokens).unwrap();
    let signature = expand_sig(&func.sig);
    let mut body = quote::quote! {};

    for statement in func.block.stmts.iter() {
        let tokens = parse_statement(statement);
        body.extend(tokens);
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
                    #ident: <#ty as burn_cube::RuntimeType>::ExpandType,
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
                <#ty as burn_cube::RuntimeType>::ExpandType
            });
        }
    }

    let ident = &sig.ident;
    let ident = syn::Ident::new(format!("{ident}_expand").as_str(), ident.span());

    quote::quote! {
        pub fn #ident(context: &mut burn_cube::CubeContext, #inputs) -> #output
    }
    .into()
}
