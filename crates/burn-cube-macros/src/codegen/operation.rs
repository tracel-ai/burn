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
        syn::BinOp::Ge(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::ge::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::Lt(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::lt::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::Le(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::le::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::Eq(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::eq::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::AddAssign(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::add_assign_op::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::SubAssign(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::sub_assign_op::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::MulAssign(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::mul_assign_op::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::DivAssign(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::div_assign_op::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::And(_) => quote::quote! {
            {

                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::and::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::Or(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::or::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::BitAnd(_) => quote::quote! {
            {

                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::bitand::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::BitXor(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::bitxor::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::Shl(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::shl::expand(context, _lhs, _rhs)
            }
        },
        syn::BinOp::Shr(_) => quote::quote! {
            {
                let _lhs = #lhs;
                let _rhs = #rhs;
                burn_cube::shr::expand(context, _lhs, _rhs)
            }
        },
        _ => todo!("Codegen: unsupported op {:?}", binary.op),
    }
}

/// Codegen for unary operations
pub(crate) fn codegen_unary(
    unary: &syn::ExprUnary,
    loop_level: usize,
    variable_analyses: &mut CodeAnalysis,
) -> TokenStream {
    let inner = codegen_expr(&unary.expr, loop_level, variable_analyses);

    match unary.op {
        syn::UnOp::Not(_) => quote::quote! {
            {
                let _inner = #inner;
                burn_cube::not::expand(context, _inner)
            }
        },
        _ => todo!("Codegen: unsupported op {:?}", unary.op),
    }
}
