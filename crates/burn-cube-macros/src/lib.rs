#[macro_use]
extern crate derive_new;

mod analyzer;
mod codegen_function;
mod codegen_trait;
mod codegen_type;
mod tracker;

pub(crate) mod codegen_common;

use analyzer::VariableAnalyzer;
use codegen_common::signature::{expand_sig, ExpandMode};
use codegen_function::{codegen_launch, codegen_statement};
use codegen_trait::{expand_trait_def, expand_trait_impl};
use codegen_type::generate_cube_type;
use proc_macro::TokenStream;
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

struct SupportedAttributes {
    mode: CubeMode,
    launch: bool,
}

/// Derive macro for the module.
#[proc_macro_attribute]
pub fn cube(attr: TokenStream, tokens: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr with Punctuated::<Meta, syn::Token![,]>::parse_terminated);
    let attrs = parse_attributes(&args);

    let code: TokenStream = match syn::parse::<syn::Item>(tokens).unwrap() {
        syn::Item::Fn(func) => cube_fn(func, &attrs),
        syn::Item::Impl(item) => expand_trait_impl(item).into(),
        syn::Item::Trait(item) => expand_trait_def(item).into(),
        _ => panic!("Cube annotations only supported for functions"),
    };

    match attrs.mode {
        CubeMode::Default => code,
        CubeMode::Debug => panic!("{code}"),
    }
}

fn cube_fn(func: syn::ItemFn, attrs: &SupportedAttributes) -> TokenStream {
    let mut variable_tracker = VariableAnalyzer::create_tracker(&func);

    match codegen_cube(&func, &mut variable_tracker, attrs.launch) {
        Ok(code) => code.into(),
        Err(err) => err.into(),
    }
}

fn parse_attributes(args: &Punctuated<Meta, Comma>) -> SupportedAttributes {
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

    SupportedAttributes { mode, launch }
}

/// Generate the expanded version of a function marked with the cube macro
fn codegen_cube(
    func: &syn::ItemFn,
    variable_tracker: &mut VariableTracker,
    launch: bool,
) -> Result<proc_macro2::TokenStream, proc_macro2::TokenStream> {
    let signature = expand_sig(
        &func.sig,
        &syn::Visibility::Public(Default::default()), // Always public, otherwise we can't import
        // it from an outside module.
        Some(variable_tracker),
        ExpandMode::FuncImpl,
    );
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

    let launch_doc = if launch {
        "and launch functions "
    } else {
        "function "
    };

    let launch = if launch {
        codegen_launch(&func.sig)
    } else {
        quote::quote! {}
    };

    let mod_name = &func.sig.ident;
    let vis = &func.vis;
    let doc = format!("Module containing the expand {launch_doc}of {mod_name}.");

    Ok(quote::quote! {
        #[allow(dead_code)]
        #[allow(clippy::too_many_arguments)]
        #func


        #[doc = #doc]
        #vis mod #mod_name {
            use super::*;

            #launch

            #[allow(unused_mut)]
            #[allow(clippy::too_many_arguments)]
            #signature {
                #body
            }

        }
    })
}
