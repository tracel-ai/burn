#[macro_use]
extern crate derive_new;

mod analyzer;
mod codegen_function;
mod codegen_type;
mod tracker;

use analyzer::VariableAnalyzer;
use codegen_function::{codegen_launch, codegen_statement};
use codegen_type::generate_cube_type;
use proc_macro::TokenStream;
use quote::ToTokens;
use syn::{parse_macro_input, punctuated::Punctuated, token::Comma, Meta};
use tracker::VariableTracker;

enum CubeMode {
    /// Generates the expanded version of the function
    Default,
    /// Panics and prints the generated code, useful when debugging
    /// Use by writing #[cube(panic)]
    Debug,
}

// Derive macro to define a cube type that is launched with a kernel
#[proc_macro_derive(CubeLaunch)]
pub fn module_derive_cube_launch(input: TokenStream) -> TokenStream {
    let input = syn::parse(input).unwrap();

    generate_cube_type(&input, true)
}

// Derive macro to define a cube type that is not launched
#[proc_macro_derive(CubeType)]
pub fn module_derive_cube_type(input: TokenStream) -> TokenStream {
    let input = syn::parse(input).unwrap();

    generate_cube_type(&input, false)
}

/// Derive macro for the module.
#[proc_macro_attribute]
pub fn cube(attr: TokenStream, tokens: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr with Punctuated::<Meta, syn::Token![,]>::parse_terminated);
    let (mode, launch) = parse_attributes(&args);

    let func: syn::ItemFn =
        syn::parse(tokens).expect("Cube annotations only supported for functions");

    let mut variable_tracker = VariableAnalyzer::create_tracker(&func);

    let code: TokenStream = match codegen_cube(&func, &mut variable_tracker) {
        Ok(code) => {
            if launch {
                let launch = codegen_launch(&func.sig);

                quote::quote! {
                    #code
                    #launch
                }
                .into()
            } else {
                code.into()
            }
        }
        Err(err) => err.into(),
    };

    match mode {
        CubeMode::Default => code,
        CubeMode::Debug => panic!("State\n:{variable_tracker:?}\nCode:\n{code}"),
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

/// Generate the expanded version of a function marked with the cube macro
fn codegen_cube(
    func: &syn::ItemFn,
    variable_tracker: &mut VariableTracker,
) -> Result<proc_macro2::TokenStream, proc_macro2::TokenStream> {
    let signature = expand_sig(&func.sig, variable_tracker);
    let mut body = quote::quote! {};

    for statement in func.block.stmts.iter() {
        let tokens = codegen_statement(statement, 0, variable_tracker);
        body.extend(tokens);
    }

    let is_in_error = !variable_tracker.errors.is_empty();

    if is_in_error {
        // When there is an error, we don't generate the expand method, since it's only going to
        // create more errors that won't help fixing the issue.

        let mut code = quote::quote! {
            #[allow(dead_code)]
            #[allow(clippy::too_many_arguments)]
            #func
        };

        for err in variable_tracker.errors.drain(..) {
            code.extend(err.into_compile_error());
        }

        return Err(code);
    }

    Ok(quote::quote! {
        #[allow(dead_code)]
        #[allow(clippy::too_many_arguments)]
        #func

        #[allow(unused_mut)]
        #[allow(clippy::too_many_arguments)]
        #signature {
            #body
        }
    })
}

fn expand_sig(
    sig: &syn::Signature,
    variable_tracker: &mut VariableTracker,
) -> proc_macro2::TokenStream {
    let mut inputs = quote::quote!();

    for input in &sig.inputs {
        match input {
            syn::FnArg::Typed(pat) => {
                let ty = &pat.ty;
                let ident = pat.pat.clone();

                if let syn::Pat::Ident(ident) = ident.as_ref() {
                    variable_tracker.codegen_declare(ident.ident.to_string(), 0);
                }

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
        /// Expanded Cube function
        pub fn #ident #generics (context: &mut burn_cube::frontend::CubeContext, #inputs) -> #output
    }
}
