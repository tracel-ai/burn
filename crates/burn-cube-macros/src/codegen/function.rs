use proc_macro2::TokenStream;
use quote::quote_spanned;
use syn::{
    punctuated::Punctuated, spanned::Spanned, AngleBracketedGenericArguments, Expr, Ident,
    PathArguments, Token,
};

use crate::{analysis::CodeAnalysis, codegen::base::codegen_expr};

/// Codegen for method call
/// Supports [expr].method(args)
pub(crate) fn codegen_expr_method_call(
    call: &syn::ExprMethodCall,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    let receiver = codegen_expr(&call.receiver, loop_level, variable_analyses);
    let method_expand = syn::Ident::new(
        format!("{}_expand", call.method).as_str(),
        proc_macro2::Span::call_site(),
    );
    let args = codegen_args(&call.args, loop_level, variable_analyses);

    quote::quote!( #receiver . #method_expand ( #args ))
}

/// Codegen for a closure
pub(crate) fn codegen_closure(
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

/// Codegen for a function call
pub(crate) fn codegen_call(
    call: &syn::ExprCall,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    parse_function_call(call, loop_level, variable_analyses).0
}

/// Maps
/// [A[::<...>]?::]^* func[::<...>] (args)
/// to
/// [A[::<...>]?::]^* func_expand[::<...>] (context, args)
///
/// Also returns a bool that is true if it's comptime
pub(crate) fn parse_function_call(
    call: &syn::ExprCall,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> (TokenStream, bool) {
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
        _ => todo!("Codegen: func call {:?} not supported", call.func),
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
            "unwrap_or_else" => {
                let args = codegen_args(&call.args, loop_level, variable_analyses);

                // Codegen
                quote::quote! {
                    Comptime::unwrap_or_else_expand(#args)
                }
            }
            "is_some" => {
                let code = call.args.first().unwrap();
                quote::quote! { #code.is_some() }
            }
            _ => panic!("Codegen: Comptime function {:?} does not exist", func_name),
        };

        (tokens, true)
    } else {
        let args = codegen_args(&call.args, loop_level, variable_analyses);

        // Codegen
        let tokens = quote::quote! {
            #path_tokens (#args)
        };

        (tokens, false)
    }
}

fn codegen_args(
    args: &Punctuated<Expr, Token![,]>,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    let mut arg_tokens = quote::quote! {};
    arg_tokens.extend(quote::quote! { context, });
    for argument in args.iter() {
        let arg_token = codegen_expr(argument, loop_level, variable_analyses);
        arg_tokens.extend(quote::quote! { #arg_token, });
    }
    arg_tokens
}
