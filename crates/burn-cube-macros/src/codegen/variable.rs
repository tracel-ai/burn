use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::Lit;

use crate::{analysis::CodeAnalysis, codegen::base::codegen_expr};

/// Codegen for literals
pub(crate) fn codegen_lit(lit: &syn::ExprLit) -> TokenStream {
    match lit.lit {
        // We treat floats differently to avoid getting 4..into() for instance
        Lit::Float(_) => {
            let lit_str = lit.lit.to_token_stream().to_string();
            let float_lit = lit_str.parse::<f32>().unwrap();
            quote::quote! { #float_lit.into() }
        }
        _ => {
            quote::quote! { #lit.into() }
        }
    }
}

/// Codegen for a local declaration (let ...)
/// Supports:
/// let x = ...
/// let x: T = ...
/// let _ = ...
pub(crate) fn codegen_local(
    local: &syn::Local,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    let let_tok = local.let_token;

    let ident = match &local.pat {
        syn::Pat::Ident(ident) => ident.to_token_stream(),
        syn::Pat::Type(pat_type) => match &*pat_type.pat {
            syn::Pat::Ident(pat_ident) => pat_ident.to_token_stream(),
            _ => todo!("Codegen: Unsupported typed path {:?}", pat_type.pat),
        },
        syn::Pat::Wild(wild) => wild.underscore_token.to_token_stream(),
        _ => todo!("Codegen: Declaration {:?} is unsupported.", local.pat),
    };

    match local.init.as_ref() {
        Some(init) => {
            let init = codegen_expr(&init.expr, loop_level, variable_analyses);

            quote::quote! {
                #let_tok #ident = #init;
            }
        }
        None => {
            quote::quote! {
                #let_tok #ident;
            }
        }
    }
}

/// Codegen for indexed access
pub(crate) fn codegen_index(
    index: &syn::ExprIndex,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    let array = codegen_expr(&index.expr, loop_level, variable_analyses);
    let index = codegen_expr(&index.index, loop_level, variable_analyses);

    quote::quote! {
        {
            let _array = #array;
            let _index = #index;
            burn_cube::index::expand(context, _array, _index)
        }
    }
}

/// Codegen for assignation
/// Supports:
/// - scalar
/// - indexed array
pub(crate) fn codegen_assign(
    assign: &syn::ExprAssign,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    match assign.left.as_ref() {
        syn::Expr::Index(index) => {
            let array = codegen_expr(&index.expr, loop_level, variable_analyses);
            let index = codegen_expr(&index.index, loop_level, variable_analyses);
            let value = codegen_expr(&assign.right, loop_level, variable_analyses);

            quote::quote! {
                {
                let _array = #array;
                let _index = #index;
                let _value = #value;
                burn_cube::index_assign::expand(context, _array, _index, _value)
                }
            }
        }
        syn::Expr::Path(_) => {
            let lhs = codegen_expr(&assign.left, loop_level, variable_analyses);
            let rhs = codegen_expr(&assign.right, loop_level, variable_analyses);

            quote::quote! {
                {
                    let _assign_lhs = #lhs;
                    let _assign_rhs = #rhs;
                    burn_cube::assign::expand(context, _assign_rhs, _assign_lhs)
                }
            }
        }
        _ => todo!("Assign of expr {:?} unsupported", assign.left),
    }
}

/// Codegen for a variable used in rhs of a statement
/// This function adds cloning when necessary
pub(crate) fn codegen_path_rhs(
    path: &syn::ExprPath,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    let ident = path
        .path
        .get_ident()
        .expect("Codegen: Only ident path are supported.");

    let will_be_used_again = variable_analyses.should_clone(ident, loop_level);

    if will_be_used_again {
        quote::quote! {
            #ident.clone()
        }
    } else {
        quote::quote! {
            #ident
        }
    }
}
