mod analysis;
mod codegen;
mod prelude;

use analysis::CodeAnalysis;
use codegen::codegen_statement;
use prelude::get_prelude;
use proc_macro::TokenStream;
use quote::ToTokens;
use syn::{parse_macro_input, punctuated::Punctuated, Meta};

/// Derive macro for the module.
#[proc_macro_attribute]
pub fn cube(attr: TokenStream, tokens: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr with Punctuated::<Meta, syn::Token![,]>::parse_terminated);

    let mut panic_mode = false;
    if let Some(arg) = args.first() {
        match arg {
            Meta::Path(path) => {
                if let Some(ident) = path.get_ident().map(|id| id.to_string()) {
                    match ident.as_str() {
                        "panic" => {
                            panic_mode = true;
                        }
                        _ => panic!("Attribute {ident} is not supported"),
                    }
                } else {
                    panic!("Only ident attribute supported");
                }
            }
            Meta::List(_) => panic!("No List attribute supported"),
            Meta::NameValue(_) => panic!("No NameValue attribute supported"),
        }
    }

    let func: syn::ItemFn = syn::parse(tokens).unwrap();
    let mut variable_analyses = CodeAnalysis::create(&func);

    let code = codegen_cube(&func, &mut variable_analyses);
    match panic_mode {
        true => panic!("{code}"),
        false => code,
    }
}

#[derive(Hash, PartialEq, Eq, Debug, Clone)]
struct VariableKey {
    name: String,
}

impl From<&syn::Ident> for VariableKey {
    fn from(value: &syn::Ident) -> Self {
        VariableKey {
            name: value.to_string(),
        }
    }
}

/// Generate the expanded version of a function marked with the cube macro
fn codegen_cube(func: &syn::ItemFn, code_analysis: &mut CodeAnalysis) -> TokenStream {
    let prelude = get_prelude(&code_analysis.needed_functions);
    let mod_name = get_name(&func.sig);
    let signature = expand_sig(&func.sig);
    let mut body = quote::quote! {};

    for statement in func.block.stmts.iter() {
        let tokens = codegen_statement(statement, 0, code_analysis);
        body.extend(tokens);
    }

    let code = quote::quote! {
        mod #mod_name {
            #prelude

            #[allow(dead_code)]
            #func

            #[allow(unused_mut)]
            #signature {
                #body
            }
        }
    }
    .into();

    code
}

fn get_name(sig: &syn::Signature) -> proc_macro2::TokenStream {
    let ident = &sig.ident;

    quote::quote! {
        #ident
    }
    .into()
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
    let ident = syn::Ident::new("expand", ident.span());

    let generics = sig.generics.clone().into_token_stream();

    quote::quote! {
        pub fn #ident #generics (context: &mut burn_cube::CubeContext, #inputs) -> #output
    }
    .into()
}
