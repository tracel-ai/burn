use burn_cube::{cube, Float, Numeric};

#[cube]
fn add_op<T: Numeric>(a: T, b: T) -> T {
    a + b
}

#[cube]
fn sub_op<T: Numeric>(a: T, b: T) -> T {
    a - b
}

#[cube]
fn mul_op<T: Numeric>(a: T, b: T) -> T {
    a * b
}

#[cube]
fn div_op<T: Numeric>(a: T, b: T) -> T {
    a / b
}

#[cube]
fn abs_op<T: Numeric>(a: T) -> T {
    T::abs(a)
}

#[cube]
fn exp_op<F: Float>(a: F) -> F {
    F::exp(a)
}

#[cube]
fn log_op<F: Float>(a: F) -> F {
    F::log(a)
}

#[cube]
fn log1p_op<F: Float>(a: F) -> F {
    F::log1p(a)
}

#[cube]
fn cos_op<F: Float>(a: F) -> F {
    F::cos(a)
}

#[cube]
fn sin_op<F: Float>(a: F) -> F {
    F::sin(a)
}

#[cube]
fn tanh_op<F: Float>(a: F) -> F {
    F::tanh(a)
}

#[cube]
fn powf_op<F: Float>(a: F, b: F) -> F {
    F::powf(a, b)
}

// #[cube]
// fn sqrt_op<F: Float>(a: F) -> F {
//     F::sqrt(a)
// }

// #[cube]
// fn floor_op<F: Float>(a: F) -> F {
//     F::floor(a)
// }

// #[cube]
// fn ceil_op<F: Float>(a: F) -> F {
//     F::ceil(a)
// }

// #[cube]
// fn erf_op<F: Float>(a: F) -> F {
//     F::erf(a)
// }

// #[cube]
// fn recip_op<F: Float>(a: F) -> F {
//     F::recip(a)
// }

// #[cube]
// fn equal_op<T: PrimitiveVariable>(a: T, b: T) -> Bool {
//     a == b
// }

// #[cube]
// fn not_equal_op<T: PrimitiveVariable>(a: T, b: T) -> Bool {
//     a != b
// }

// #[cube]
// fn lower_op<T: Numeric>(a: T, b: T) -> Bool {
//     a < b
// }

// #[cube]
// fn greater_op<T: Numeric>(a: T, b: T) -> Bool {
//     a > b
// }

// #[cube]
// fn lower_equal_op<T: Numeric>(a: T, b: T) -> Bool {
//     a <= b
// }

// #[cube]
// fn greater_equal_op<T: Numeric>(a: T, b: T) -> Bool {
//     a >= b
// }

// #[cube]
// fn clamp_op<T: Numeric>(a: T, l: T, u: T) -> T {
//     clamp(a, l, u)
// }

// #[cube]
// fn modulo_op(a: UInt, b: UInt) -> UInt {
//     a % b
// }

// #[cube]
// fn remainder_op<T: Numeric>(a: T, b: T) -> T {
//     rem(a, b)
// }

// #[cube]
// fn max_op<T: Numeric>(a: T, b: T) -> T {
//     max(a, b)
// }

// #[cube]
// fn min_op<T: Numeric>(a: T, b: T) -> T {
//     min(a, b)
// }

// #[cube]
// fn and_op(a: Bool, b: Bool) -> Bool {
//     a & b
// }

// #[cube]
// fn or_op(a: Bool, b: Bool) -> Bool {
//     a | b
// }

// #[cube]
// fn not_op(a: Bool) -> Bool {
//     !a
// }

// #[cube]
// fn bit_and_op(a: UInt, b: UInt) -> UInt {
//     a & b
// }

// #[cube]
// fn bit_or_op(a: UInt, b: UInt) -> UInt {
//     a | b
// }

// #[cube]
// fn shift_left_op(a: UInt, b: UInt) -> UInt {
//     a << b
// }

// #[cube]
// fn shift_left_op(a: UInt, b: UInt) -> UInt {
//     a >> b
// }

mod tests {
    use super::*;
    use burn_cube::{
        dialect::{Elem, FloatKind, Item},
        CubeContext, F32,
    };

    type ElemType = F32;

    macro_rules! define_binary_test {
        ($test_name:ident, $op_expand:ident, $op_name:expr) => {
            #[test]
            fn $test_name() {
                let mut context = CubeContext::root();
                let x = context.create_local(Item::new(Elem::Float(FloatKind::F32)));
                let y = context.create_local(Item::new(Elem::Float(FloatKind::F32)));

                $op_expand::<ElemType>(&mut context, x, y);

                assert_eq!(
                    format!("{:?}", context.into_scope().operations),
                    ref_ops_binary($op_name)
                );
            }
        };
    }

    macro_rules! define_unary_test {
        ($test_name:ident, $op_expand:ident, $op_name:expr) => {
            #[test]
            fn $test_name() {
                let mut context = CubeContext::root();
                let x = context.create_local(Item::new(Elem::Float(FloatKind::F32)));

                $op_expand::<ElemType>(&mut context, x);

                assert_eq!(
                    format!("{:?}", context.into_scope().operations),
                    ref_ops_unary($op_name)
                );
            }
        };
    }

    define_binary_test!(cube_can_add, add_op_expand, "Add");
    define_binary_test!(cube_can_sub, sub_op_expand, "Sub");
    define_binary_test!(cube_can_mul, mul_op_expand, "Mul");
    define_binary_test!(cube_can_div, div_op_expand, "Div");
    define_unary_test!(cube_can_abs, abs_op_expand, "Abs");
    define_unary_test!(cube_can_exp, exp_op_expand, "Exp");
    define_unary_test!(cube_can_log, log_op_expand, "Log");
    define_unary_test!(cube_can_log1p, log1p_op_expand, "Log1p");
    define_unary_test!(cube_can_cos, cos_op_expand, "Cos");
    define_unary_test!(cube_can_sin, sin_op_expand, "Sin");
    define_unary_test!(cube_can_tanh, tanh_op_expand, "Tanh");
    define_binary_test!(cube_can_powf, powf_op_expand, "Powf");

    fn ref_ops_binary(ops_name: &str) -> String {
        format!(
            "[Operator({}(BinaryOperator {{ \
            lhs: Local(0, Item {{ \
                elem: Float(F32), \
                vectorization: 1 \
            }}, 0), \
            rhs: Local(1, Item {{ \
                elem: Float(F32), \
                vectorization: 1 \
            }}, 0), \
            out: Local(0, Item {{ \
                elem: Float(F32), \
                vectorization: 1 \
            }}, 0) \
        }}))]",
            ops_name
        )
    }

    fn ref_ops_unary(ops_name: &str) -> String {
        format!(
            "[Operator({}(UnaryOperator {{ \
        input: Local(0, Item {{ \
            elem: Float(F32), \
            vectorization: 1 \
        }}, 0), \
        out: Local(0, Item {{ \
            elem: Float(F32), \
            vectorization: 1 \
        }}, 0) \
    }}))]",
            ops_name
        )
    }
}
