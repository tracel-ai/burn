use proc_macro2::TokenStream;

use crate::analysis::CodeAnalysis;

use super::base::codegen_expr;

/// Codegen for binary operations (+, -, *, etc.)
pub(crate) fn codegen_binary(
    binary: &syn::ExprBinary,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    let lhs = codegen_expr(&binary.left, loop_level, variable_analyses);
    let rhs = codegen_expr(&binary.right, loop_level, variable_analyses);

    match binary.op {
        syn::BinOp::Add(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::add::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::Sub(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::sub::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::Mul(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::mul::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::Div(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::div::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::Rem(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::rem::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::Ne(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::ne::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::Gt(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::gt::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::Lt(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::lt::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::AddAssign(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::add_assign_op::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::BitAnd(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::and::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::And(_) => unimplemented!("Logical and (&&) not overridable in Rust due to its short circuiting nature. Use bitwise instead (&). "),
        syn::BinOp::BitOr(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::or::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::Or(_) => unimplemented!("Logical or (||) not overridable in Rust due to its short circuiting nature. Use bitwise instead (|). "),
        _ => todo!("Codegen: unsupported op {:?}", binary.op),
    }
}
