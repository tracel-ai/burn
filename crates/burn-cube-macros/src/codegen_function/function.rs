use proc_macro2::{Span, TokenStream};
use quote::quote_spanned;
use syn::{
    punctuated::Punctuated, spanned::Spanned, AngleBracketedGenericArguments, Expr, Ident,
    PathArguments, Token,
};

use crate::{codegen_function::expr::codegen_expr, tracker::VariableTracker};

use super::base::Codegen;

/// Codegen for method call
/// Supports [expr].method(args)
pub(crate) fn codegen_expr_method_call(
    call: &syn::ExprMethodCall,
    loop_level: usize,
    variable_tracker: &mut VariableTracker,
) -> TokenStream {
    let receiver = codegen_expr(&call.receiver, loop_level, variable_tracker);
    let method_expand = syn::Ident::new(
        format!("{}_expand", call.method).as_str(),
        proc_macro2::Span::call_site(),
    );
    let (expansion, variables) = codegen_args(&call.args, loop_level, variable_tracker);

    quote::quote!( {
        #expansion
        #receiver . #method_expand ( #variables )
    })
}

/// Codegen for a closure
pub(crate) fn codegen_closure(
    closure: &syn::ExprClosure,
    loop_level: usize,
    variable_tracker: &mut VariableTracker,
) -> TokenStream {
    let mut inputs = quote::quote! {};
    for input in closure.inputs.iter() {
        let (ident, ty) = match input {
            syn::Pat::Ident(ident) => (&ident.ident, None),
            syn::Pat::Type(pat_type) => (
                if let syn::Pat::Ident(ident) = &*pat_type.pat {
                    &ident.ident
                } else {
                    return syn::Error::new_spanned(pat_type, "Unsupported input")
                        .into_compile_error();
                },
                Some(pat_type.ty.clone()),
            ),
            _ => return syn::Error::new_spanned(input, "Unsupported input").into_compile_error(),
        };

        if let Some(ty) = ty {
            inputs.extend(quote::quote! {
                #ident : #ty,
            });
        } else {
            inputs.extend(quote::quote! {
                #ident,
            });
        }
    }

    let body = codegen_expr(closure.body.as_ref(), loop_level, variable_tracker);

    quote::quote! {
        |context: &mut CubeContext, #inputs| #body
    }
}

/// Maps
/// [A[::<...>]?::]^* func[::<...>] (args)
/// to
/// [A[::<...>]?::]^* func_expand[::<...>] (context, args)
///
/// Also returns a bool that is true if it's comptime
pub(crate) fn codegen_call(
    call: &syn::ExprCall,
    loop_level: usize,
    variable_tracker: &mut VariableTracker,
) -> Codegen {
    // We start with parsing the function path
    let path: Vec<(&Ident, Option<&AngleBracketedGenericArguments>)> = match call.func.as_ref() {
        syn::Expr::Path(expr_path) => {
            let mut path = Vec::new();
            for segment in expr_path.path.segments.iter() {
                let generics = if let PathArguments::AngleBracketed(arguments) = &segment.arguments
                {
                    Some(arguments)
                } else {
                    None
                };
                path.push((&segment.ident, generics));
            }
            path
        }
        _ => {
            return Codegen::new(
                syn::Error::new_spanned(&call.func, "Unsupported").into_compile_error(),
                false,
            )
        }
    };

    // Path
    let mut path_tokens = TokenStream::new();
    let mut is_comptime = false;
    let mut comptime_func: Option<String> = None;

    for (i, (ident, generics)) in path.iter().enumerate() {
        if *ident == "Comptime" {
            is_comptime = true;
            continue;
        }
        if i == path.len() - 1 {
            if is_comptime {
                comptime_func = Some(ident.to_string());
                break;
            }
            let func_name_expand = syn::Ident::new(
                format!("{ident}_expand").as_str(),
                proc_macro2::Span::call_site(),
            );
            path_tokens.extend(quote_spanned! {func_name_expand.span() => #func_name_expand });
            if let Some(generics) = generics {
                path_tokens.extend(quote_spanned! {generics.span() => #generics });
            }
        } else if let Some(generics) = generics {
            path_tokens.extend(quote_spanned! {ident.span() => #ident });
            path_tokens.extend(quote_spanned! {generics.span() => #generics :: });
        } else {
            path_tokens.extend(quote_spanned! {ident.span() => #ident :: });
        }
    }

    // Arguments
    if let Some(func_name) = comptime_func {
        let tokens = match func_name.as_str() {
            "get" | "new" => {
                let code = call.args.first().unwrap();
                quote::quote! {#code}
            }
            "map" => {
                let args = &call.args;

                // Codegen
                quote::quote! {
                    {
                        Comptime::map_expand(#args)
                    }
                }
            }
            "unwrap_or_else" => {
                let (expansion, variables) = codegen_args(&call.args, loop_level, variable_tracker);

                // Codegen
                quote::quote! {{
                    #expansion
                    Comptime::unwrap_or_else_expand(#variables)
                }}
            }
            "is_some" => {
                let code = call.args.first().unwrap();
                quote::quote! { #code.is_some() }
            }
            "vectorization" => {
                let (expansion, variables) = codegen_args(&call.args, loop_level, variable_tracker);

                // Codegen
                quote::quote! {{
                    #expansion
                    Comptime::vectorization_expand(#variables)
                }}
            }
            "vectorize" => {
                let (expansion, variables) = codegen_args(&call.args, loop_level, variable_tracker);

                // Codegen
                quote::quote! {{
                    #expansion
                    Comptime::vectorize_expand(#variables)
                }}
            }
            "runtime" => {
                let (expansion, variables) = codegen_args(&call.args, loop_level, variable_tracker);

                // Codegen
                quote::quote! {{
                    #expansion
                    Comptime::runtime_expand(#variables)
                }}
            }

            _ => panic!("Codegen: Comptime function {:?} does not exist", func_name),
        };

        Codegen::new(tokens, true)
    } else {
        let (expansion, variables) = codegen_args(&call.args, loop_level, variable_tracker);

        // Codegen
        let tokens = quote::quote! {{
            #expansion
            #path_tokens (#variables)
        }};

        Codegen::new(tokens, false)
    }
}

fn codegen_args(
    args: &Punctuated<Expr, Token![,]>,
    loop_level: usize,
    variable_tracker: &mut VariableTracker,
) -> (TokenStream, TokenStream) {
    let mut expansion = quote::quote! {};
    let mut variables = quote::quote! {};

    variables.extend(quote::quote! { context, });

    for (i, argument) in args.iter().enumerate() {
        let ident = Ident::new(format!("_var_{i}").as_str(), Span::call_site());
        let arg_token = codegen_expr(argument, loop_level, variable_tracker);
        expansion.extend(quote::quote! { let #ident = #arg_token; });
        variables.extend(quote::quote! { #ident, });
    }

    (expansion, variables)
}
