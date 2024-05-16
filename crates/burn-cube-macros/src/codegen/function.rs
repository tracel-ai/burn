use proc_macro2::TokenStream;
use quote::quote_spanned;
use syn::PathArguments;

use crate::{analysis::CodeAnalysis, codegen::base::codegen_expr};

/// Codegen for method call
pub(crate) fn codegen_expr_method_call(call: &syn::ExprMethodCall) -> TokenStream {
    quote::quote!( #call )
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
/// Supports:
/// func()
/// func::<T>()
/// T::func()
pub(crate) fn codegen_call(
    call: &syn::ExprCall,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    // We start with parsing the function path
    let (mut idents, generics) = match call.func.as_ref() {
        syn::Expr::Path(expr_path) => {
            let mut idents = Vec::new();
            let mut generics = None;
            for (index, segment) in expr_path.path.segments.iter().enumerate() {
                idents.push(&segment.ident);

                if index == expr_path.path.segments.len() - 1 {
                    if let PathArguments::AngleBracketed(arguments) = &segment.arguments {
                        generics = Some(arguments)
                    }
                }
            }
            (idents, generics)
        }
        _ => todo!("Codegen: func call {:?} not supported", call.func),
    };

    // Function name with support for longer path
    let func_name = idents
        .pop()
        .expect("Codegen: Func should have at least one ident");

    let mut previous_tokens = TokenStream::new();
    for ident in idents.iter() {
        previous_tokens.extend(quote_spanned! {ident.span() => #ident :: });
    }
    let func_name_expand = syn::Ident::new(
        format!("{func_name}_expand").as_str(),
        proc_macro2::Span::call_site(),
    );

    // Generics
    let generics = match generics {
        Some(generics) => quote::quote! { #generics },
        None => quote::quote! {},
    };

    // Arguments
    let mut args = quote::quote! {
        context,
    };
    for argument in call.args.iter() {
        let arg = codegen_expr(argument, loop_level, variable_analyses);
        args.extend(quote::quote! { #arg, });
    }

    // Codegen
    quote::quote! {
        #previous_tokens #func_name_expand #generics (#args)
    }
}
