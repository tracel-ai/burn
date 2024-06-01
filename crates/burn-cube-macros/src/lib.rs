mod analysis;
mod codegen;

use analysis::CodeAnalysis;
use codegen::{codegen_launch, codegen_statement};
use proc_macro::TokenStream;
use quote::ToTokens;
use syn::{parse_macro_input, punctuated::Punctuated, token::Comma, Meta};

enum CubeMode {
    /// Generates the expanded version of the function
    Default,
    /// Panics and prints the generated code, useful when debugging
    /// Use by writing #[cube(panic)]
    Debug,
}

/// Derive macro for the module.
#[proc_macro_attribute]
pub fn cube(attr: TokenStream, tokens: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr with Punctuated::<Meta, syn::Token![,]>::parse_terminated);
    let (mode, launch) = parse_attributes(&args);

    let func: syn::ItemFn = syn::parse(tokens).unwrap();
    let mut variable_analyses = CodeAnalysis::create(&func);

    let cube = codegen_cube(&func, &mut variable_analyses);
    let code: TokenStream = if launch {
        let launch = codegen_launch(&func.sig);

        quote::quote! {
            #cube
            #launch
        }
        .into()
    } else {
        cube.into()
    };

    match mode {
        CubeMode::Default => code,
        CubeMode::Debug => panic!("{code}"),
    }
}

fn parse_attributes(args: &Punctuated<Meta, Comma>) -> (CubeMode, bool) {
    let mut mode = CubeMode::Default;
    let mut launch = false;

    for arg in args.iter() {
        match arg {
            Meta::Path(path) => {
                if let Some(ident) = path.get_ident().map(|id| id.to_string()) {
                    match ident.as_str() {
                        "debug" => {
                            mode = CubeMode::Debug;
                        }
                        "launch" => {
                            launch = true;
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

    (mode, launch)
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
fn codegen_cube(func: &syn::ItemFn, code_analysis: &mut CodeAnalysis) -> proc_macro2::TokenStream {
    let signature = expand_sig(&func.sig);
    let mut body = quote::quote! {};

    for statement in func.block.stmts.iter() {
        let tokens = codegen_statement(statement, 0, code_analysis);
        body.extend(tokens);
    }

    quote::quote! {
        #[allow(dead_code)]
        #[allow(clippy::too_many_arguments)] // TODO support structs in Cube
        #func

        #[allow(unused_mut)]
        #[allow(clippy::too_many_arguments)] // TODO support structs in Cube
        #signature {
            #body
        }
    }
}

fn expand_sig(sig: &syn::Signature) -> proc_macro2::TokenStream {
    let mut inputs = quote::quote!();

    for input in &sig.inputs {
        match input {
            syn::FnArg::Typed(pat) => {
                let ty = &pat.ty;
                let ident = pat.pat.clone();

                inputs.extend(quote::quote! {
                    #ident: <#ty as burn_cube::frontend::CubeType>::ExpandType,
                });
            }
            _ => todo!("Only Typed inputs are supported"),
        }
    }

    let mut output = quote::quote!();

    match &sig.output {
        syn::ReturnType::Default => output.extend(quote::quote! { ()}),
        syn::ReturnType::Type(_, ty) => {
            output.extend(quote::quote! {
                <#ty as burn_cube::frontend::CubeType>::ExpandType
            });
        }
    }

    let ident = &sig.ident;
    let ident = syn::Ident::new(format!("{ident}_expand").as_str(), ident.span());

    let generics = sig.generics.clone().into_token_stream();

    quote::quote! {
        pub fn #ident #generics (context: &mut burn_cube::frontend::CubeContext, #inputs) -> #output
    }
}
