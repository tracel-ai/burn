use burn_cube::prelude::*;

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

#[cube]
fn sqrt_op<F: Float>(a: F) -> F {
    F::sqrt(a)
}

#[cube]
fn floor_op<F: Float>(a: F) -> F {
    F::floor(a)
}

#[cube]
fn ceil_op<F: Float>(a: F) -> F {
    F::ceil(a)
}

#[cube]
fn erf_op<F: Float>(a: F) -> F {
    F::erf(a)
}

#[cube]
fn recip_op<F: Float>(a: F) -> F {
    F::recip(a)
}

#[cube]
fn equal_op<T: CubeElem>(a: T, b: T) -> bool {
    a == b
}

#[cube]
fn not_equal_op<T: CubeElem>(a: T, b: T) -> bool {
    a != b
}

#[cube]
fn lower_op<T: Numeric>(a: T, b: T) -> bool {
    a < b
}

#[cube]
fn greater_op<T: Numeric>(a: T, b: T) -> bool {
    a > b
}

#[cube]
fn lower_equal_op<T: Numeric>(a: T, b: T) -> bool {
    a <= b
}

#[cube]
fn greater_equal_op<T: Numeric>(a: T, b: T) -> bool {
    a >= b
}

#[cube]
fn modulo_op(a: UInt, b: UInt) -> UInt {
    a % b
}

#[cube]
fn remainder_op<T: Numeric>(a: T, b: T) -> T {
    T::rem(a, b)
}

#[cube]
fn max_op<T: Numeric>(a: T, b: T) -> T {
    T::max(a, b)
}

#[cube]
fn min_op<T: Numeric>(a: T, b: T) -> T {
    T::min(a, b)
}

#[cube]
fn and_op(a: bool, b: bool) -> bool {
    a && b
}

#[cube]
fn or_op(a: bool, b: bool) -> bool {
    a || b
}

#[cube]
fn not_op(a: bool) -> bool {
    !a
}

#[cube]
fn bitand_op(a: UInt, b: UInt) -> UInt {
    a & b
}

#[cube]
fn bitxor_op(a: UInt, b: UInt) -> UInt {
    a ^ b
}

#[cube]
fn shl_op(a: UInt, b: UInt) -> UInt {
    a << b
}

#[cube]
fn shr_op(a: UInt, b: UInt) -> UInt {
    a >> b
}

#[cube]
fn add_assign_op<T: Numeric>(mut a: T, b: T) {
    a += b;
}

#[cube]
fn sub_assign_op<T: Numeric>(mut a: T, b: T) {
    a -= b;
}

#[cube]
fn mul_assign_op<T: Numeric>(mut a: T, b: T) {
    a *= b;
}

#[cube]
fn div_assign_op<T: Numeric>(mut a: T, b: T) {
    a /= b;
}

mod tests {
    use super::*;
    use burn_cube::ir::{Elem, FloatKind, Item};

    macro_rules! binary_test {
        ($test_name:ident, $op_expand:ident, $op_name:expr, $func:ident) => {
            #[test]
            fn $test_name() {
                let mut context = CubeContext::root();
                let x = context.create_local(Item::new(Elem::Float(FloatKind::F32)));
                let y = context.create_local(Item::new(Elem::Float(FloatKind::F32)));

                $op_expand::<F32>(&mut context, x, y);

                assert_eq!(
                    format!("{:?}", context.into_scope().operations),
                    $func($op_name)
                );
            }
        };
    }

    macro_rules! unary_test {
        ($test_name:ident, $op_expand:ident, $op_name:expr) => {
            #[test]
            fn $test_name() {
                let mut context = CubeContext::root();
                let x = context.create_local(Item::new(Elem::Float(FloatKind::F32)));

                $op_expand::<F32>(&mut context, x);

                assert_eq!(
                    format!("{:?}", context.into_scope().operations),
                    ref_ops_unary($op_name)
                );
            }
        };
    }

    macro_rules! binary_boolean_test {
        ($test_name:ident, $op_expand:ident, $op_name:expr) => {
            #[test]
            fn $test_name() {
                let mut context = CubeContext::root();
                let x = context.create_local(Item::new(Elem::Bool));
                let y = context.create_local(Item::new(Elem::Bool));

                $op_expand(&mut context, x, y);

                assert_eq!(
                    format!("{:?}", context.into_scope().operations),
                    ref_ops_binary_boolean($op_name)
                );
            }
        };
    }

    macro_rules! binary_uint_test {
        ($test_name:ident, $op_expand:ident, $op_name:expr) => {
            #[test]
            fn $test_name() {
                let mut context = CubeContext::root();
                let x = context.create_local(Item::new(Elem::UInt));
                let y = context.create_local(Item::new(Elem::UInt));

                $op_expand(&mut context, x, y);

                assert_eq!(
                    format!("{:?}", context.into_scope().operations),
                    ref_ops_binary_uint($op_name)
                );
            }
        };
    }

    binary_test!(cube_can_add, add_op_expand, "Add", ref_ops_binary);
    binary_test!(cube_can_sub, sub_op_expand, "Sub", ref_ops_binary);
    binary_test!(cube_can_mul, mul_op_expand, "Mul", ref_ops_binary);
    binary_test!(cube_can_div, div_op_expand, "Div", ref_ops_binary);
    unary_test!(cube_can_abs, abs_op_expand, "Abs");
    unary_test!(cube_can_exp, exp_op_expand, "Exp");
    unary_test!(cube_can_log, log_op_expand, "Log");
    unary_test!(cube_can_log1p, log1p_op_expand, "Log1p");
    unary_test!(cube_can_cos, cos_op_expand, "Cos");
    unary_test!(cube_can_sin, sin_op_expand, "Sin");
    unary_test!(cube_can_tanh, tanh_op_expand, "Tanh");
    binary_test!(cube_can_powf, powf_op_expand, "Powf", ref_ops_binary);
    unary_test!(cube_can_sqrt, sqrt_op_expand, "Sqrt");
    unary_test!(cube_can_erf, erf_op_expand, "Erf");
    unary_test!(cube_can_recip, recip_op_expand, "Recip");
    unary_test!(cube_can_floor, floor_op_expand, "Floor");
    unary_test!(cube_can_ceil, ceil_op_expand, "Ceil");
    binary_test!(cube_can_eq, equal_op_expand, "Equal", ref_ops_cmp);
    binary_test!(cube_can_ne, not_equal_op_expand, "NotEqual", ref_ops_cmp);
    binary_test!(cube_can_lt, lower_op_expand, "Lower", ref_ops_cmp);
    binary_test!(
        cube_can_le,
        lower_equal_op_expand,
        "LowerEqual",
        ref_ops_cmp
    );
    binary_test!(
        cube_can_ge,
        greater_equal_op_expand,
        "GreaterEqual",
        ref_ops_cmp
    );
    binary_test!(cube_can_gt, greater_op_expand, "Greater", ref_ops_cmp);
    binary_test!(cube_can_max, max_op_expand, "Max", ref_ops_binary);
    binary_test!(cube_can_min, min_op_expand, "Min", ref_ops_binary);
    binary_test!(
        cube_can_add_assign,
        add_assign_op_expand,
        "Add",
        ref_ops_binary
    );
    binary_test!(
        cube_can_sub_assign,
        sub_assign_op_expand,
        "Sub",
        ref_ops_binary
    );
    binary_test!(
        cube_can_mul_assign,
        mul_assign_op_expand,
        "Mul",
        ref_ops_binary
    );
    binary_test!(
        cube_can_div_assign,
        div_assign_op_expand,
        "Div",
        ref_ops_binary
    );
    binary_boolean_test!(cube_can_and, and_op_expand, "And");
    binary_boolean_test!(cube_can_or, or_op_expand, "Or");
    binary_uint_test!(cube_can_bitand, bitand_op_expand, "BitwiseAnd");
    binary_uint_test!(cube_can_bitxor, bitxor_op_expand, "BitwiseXor");
    binary_uint_test!(cube_can_shl, shl_op_expand, "ShiftLeft");
    binary_uint_test!(cube_can_shr, shr_op_expand, "ShiftRight");
    binary_uint_test!(cube_can_mod, modulo_op_expand, "Modulo");
    binary_test!(
        cube_can_rem,
        remainder_op_expand,
        "Remainder",
        ref_ops_binary
    );

    #[test]
    fn cube_can_not() {
        let mut context = CubeContext::root();
        let x = context.create_local(Item::new(Elem::Bool));

        not_op_expand(&mut context, x);

        assert_eq!(
            format!("{:?}", context.into_scope().operations),
            ref_ops_unary_boolean("Not")
        );
    }

    fn ref_ops_binary(ops_name: &str) -> String {
        ref_ops_template(ops_name, "Float(F32)", "Float(F32)", true)
    }

    fn ref_ops_unary(ops_name: &str) -> String {
        ref_ops_template(ops_name, "Float(F32)", "Float(F32)", false)
    }

    fn ref_ops_cmp(ops_name: &str) -> String {
        ref_ops_template(ops_name, "Float(F32)", "Bool", true)
    }

    fn ref_ops_unary_boolean(ops_name: &str) -> String {
        ref_ops_template(ops_name, "Bool", "Bool", false)
    }

    fn ref_ops_binary_boolean(ops_name: &str) -> String {
        ref_ops_template(ops_name, "Bool", "Bool", true)
    }

    fn ref_ops_binary_uint(ops_name: &str) -> String {
        ref_ops_template(ops_name, "UInt", "UInt", true)
    }

    fn ref_ops_template(ops_name: &str, in_type: &str, out_type: &str, binary: bool) -> String {
        if binary {
            let out_number = if in_type == out_type { 0 } else { 2 };
            format!(
                "[Operator({ops_name}(BinaryOperator {{ \
                lhs: Local(0, Item {{ \
                    elem: {in_type}, \
                    vectorization: 1 \
                }}, 0), \
                rhs: Local(1, Item {{ \
                    elem: {in_type}, \
                    vectorization: 1 \
                }}, 0), \
                out: Local({out_number}, Item {{ \
                    elem: {out_type}, \
                    vectorization: 1 \
                }}, 0) \
            }}))]"
            )
        } else {
            format!(
                "[Operator({ops_name}(UnaryOperator {{ \
                input: Local(0, Item {{ \
                    elem: {in_type}, \
                    vectorization: 1 \
                }}, 0), \
                out: Local(0, Item {{ \
                    elem: {out_type}, \
                    vectorization: 1 \
                }}, 0) \
            }}))]"
            )
        }
    }
}
