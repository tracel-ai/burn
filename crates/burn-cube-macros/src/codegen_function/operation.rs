use crate::tracker::VariableTracker;

use super::{base::Codegen, expr::codegen_expr};

/// Codegen for binary operations (+, -, *, etc.)
pub(crate) fn codegen_binary(
    binary: &syn::ExprBinary,
    loop_level: usize,
    variable_tracker: &mut VariableTracker,
) -> Codegen {
    let lhs = codegen_expr(&binary.left, loop_level, variable_tracker);
    let (lhs, is_comptime_lhs, lhs_array) = (lhs.tokens, lhs.is_comptime, lhs.array_indexing);
    let (rhs, is_comptime_rhs) = codegen_expr(&binary.right, loop_level, variable_tracker).split();

    if is_comptime_lhs && is_comptime_rhs {
        return Codegen::new(
            quote::quote! {
                #binary
            },
            true,
        );
    }

    Codegen::new(
        match binary.op {
            syn::BinOp::Add(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    burn_cube::frontend::add::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Sub(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    burn_cube::frontend::sub::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Mul(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    burn_cube::frontend::mul::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Div(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    burn_cube::frontend::div::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Rem(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    burn_cube::frontend::rem::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Ne(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    burn_cube::frontend::ne::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Gt(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    burn_cube::frontend::gt::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Ge(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    burn_cube::frontend::ge::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Lt(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    burn_cube::frontend::lt::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Le(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    burn_cube::frontend::le::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Eq(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    burn_cube::frontend::eq::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::AddAssign(_) => {
                if let Some(array) = lhs_array {
                    let (array, index) = (array.array, array.index);

                    quote::quote! {
                        {
                            let _array = #array;
                            let _index = #index;
                            let _value = #rhs;
                            burn_cube::frontend::add_assign_array_op::expand(context, _array, _index, _value)
                        }
                    }
                } else {
                    quote::quote! {
                        {
                            let _lhs = #lhs;
                            let _rhs = #rhs;
                            burn_cube::frontend::add_assign_op::expand(context, _lhs, _rhs)
                        }
                    }
                }
            }
            syn::BinOp::SubAssign(_) => {
                if let Some(array) = lhs_array {
                    let (array, index) = (array.array, array.index);

                    quote::quote! {
                        {
                            let _array = #array;
                            let _index = #index;
                            let _value = #rhs;
                            burn_cube::frontend::sub_assign_array_op::expand(context, _array, _index, _value)
                        }
                    }
                } else {
                    quote::quote! {
                        {
                            let _lhs = #lhs;
                            let _rhs = #rhs;
                            burn_cube::frontend::sub_assign_op::expand(context, _lhs, _rhs)
                        }
                    }
                }
            }
            syn::BinOp::MulAssign(_) => {
                if let Some(array) = lhs_array {
                    let (array, index) = (array.array, array.index);

                    quote::quote! {
                        {
                            let _array = #array;
                            let _index = #index;
                            let _value = #rhs;
                            burn_cube::frontend::mul_assign_array_op::expand(context, _array, _index, _value)
                        }
                    }
                } else {
                    quote::quote! {
                        {
                            let _lhs = #lhs;
                            let _rhs = #rhs;
                            burn_cube::frontend::mul_assign_op::expand(context, _lhs, _rhs)
                        }
                    }
                }
            }
            syn::BinOp::DivAssign(_) => {
                if let Some(array) = lhs_array {
                    let (array, index) = (array.array, array.index);

                    quote::quote! {
                        {
                            let _array = #array;
                            let _index = #index;
                            let _value = #rhs;
                            burn_cube::frontend::div_assign_array_op::expand(context, _array, _index, _value)
                        }
                    }
                } else {
                    quote::quote! {
                        {
                            let _lhs = #lhs;
                            let _rhs = #rhs;
                            burn_cube::frontend::div_assign_op::expand(context, _lhs, _rhs)
                        }
                    }
                }
            }
            syn::BinOp::And(_) => quote::quote! {
                {

                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    burn_cube::frontend::and::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Or(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    burn_cube::frontend::or::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::BitAnd(_) => quote::quote! {
                {

                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    burn_cube::frontend::bitand::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::BitXor(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    burn_cube::frontend::bitxor::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Shl(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    burn_cube::frontend::shl::expand(context, _lhs, _rhs)
                }
            },
            syn::BinOp::Shr(_) => quote::quote! {
                {
                    let _lhs = #lhs;
                    let _rhs = #rhs;
                    burn_cube::frontend::shr::expand(context, _lhs, _rhs)
                }
            },
            _ => todo!("Codegen: unsupported op {:?}", binary.op),
        },
        false,
    )
}

/// Codegen for unary operations
pub(crate) fn codegen_unary(
    unary: &syn::ExprUnary,
    loop_level: usize,
    variable_tracker: &mut VariableTracker,
) -> Codegen {
    let (inner, is_comptime) = codegen_expr(&unary.expr, loop_level, variable_tracker).split();

    if is_comptime {
        return Codegen::new(
            quote::quote! {
                #unary
            },
            true,
        );
    }

    Codegen::new(
        match unary.op {
            syn::UnOp::Not(_) => quote::quote! {
                {
                    let _inner = #inner;
                    burn_cube::frontend::not::expand(context, _inner)
                }
            },
            _ => todo!("Codegen: unsupported op {:?}", unary.op),
        },
        false,
    )
}
