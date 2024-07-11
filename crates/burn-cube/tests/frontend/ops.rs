use burn_cube::prelude::*;

#[cube]
pub fn add_op<T: Numeric>(a: T, b: T) -> T {
    a + b
}

#[cube]
pub fn sub_op<T: Numeric>(a: T, b: T) -> T {
    a - b
}

#[cube]
pub fn mul_op<T: Numeric>(a: T, b: T) -> T {
    a * b
}

#[cube]
pub fn div_op<T: Numeric>(a: T, b: T) -> T {
    a / b
}

#[cube]
pub fn abs_op<T: Numeric>(a: T) -> T {
    T::abs(a)
}

#[cube]
pub fn exp_op<F: Float>(a: F) -> F {
    F::exp(a)
}

#[cube]
pub fn log_op<F: Float>(a: F) -> F {
    F::log(a)
}

#[cube]
pub fn log1p_op<F: Float>(a: F) -> F {
    F::log1p(a)
}

#[cube]
pub fn cos_op<F: Float>(a: F) -> F {
    F::cos(a)
}

#[cube]
pub fn sin_op<F: Float>(a: F) -> F {
    F::sin(a)
}

#[cube]
pub fn tanh_op<F: Float>(a: F) -> F {
    F::tanh(a)
}

#[cube]
pub fn powf_op<F: Float>(a: F, b: F) -> F {
    F::powf(a, b)
}

#[cube]
pub fn sqrt_op<F: Float>(a: F) -> F {
    F::sqrt(a)
}

#[cube]
pub fn floor_op<F: Float>(a: F) -> F {
    F::floor(a)
}

#[cube]
pub fn ceil_op<F: Float>(a: F) -> F {
    F::ceil(a)
}

#[cube]
pub fn erf_op<F: Float>(a: F) -> F {
    F::erf(a)
}

#[cube]
pub fn recip_op<F: Float>(a: F) -> F {
    F::recip(a)
}

#[cube]
pub fn equal_op<T: CubePrimitive>(a: T, b: T) -> bool {
    a == b
}

#[cube]
pub fn not_equal_op<T: CubePrimitive>(a: T, b: T) -> bool {
    a != b
}

#[cube]
pub fn lower_op<T: Numeric>(a: T, b: T) -> bool {
    a < b
}

#[cube]
pub fn greater_op<T: Numeric>(a: T, b: T) -> bool {
    a > b
}

#[cube]
pub fn lower_equal_op<T: Numeric>(a: T, b: T) -> bool {
    a <= b
}

#[cube]
pub fn greater_equal_op<T: Numeric>(a: T, b: T) -> bool {
    a >= b
}

#[cube]
pub fn modulo_op(a: UInt, b: UInt) -> UInt {
    a % b
}

#[cube]
pub fn remainder_op<T: Numeric>(a: T, b: T) -> T {
    T::rem(a, b)
}

#[cube]
pub fn max_op<T: Numeric>(a: T, b: T) -> T {
    T::max(a, b)
}

#[cube]
pub fn min_op<T: Numeric>(a: T, b: T) -> T {
    T::min(a, b)
}

#[cube]
pub fn and_op(a: bool, b: bool) -> bool {
    a && b
}

#[cube]
pub fn or_op(a: bool, b: bool) -> bool {
    a || b
}

#[cube]
pub fn not_op(a: bool) -> bool {
    !a
}

#[cube]
pub fn bitand_op(a: UInt, b: UInt) -> UInt {
    a & b
}

#[cube]
pub fn bitxor_op(a: UInt, b: UInt) -> UInt {
    a ^ b
}

#[cube]
pub fn shl_op(a: UInt, b: UInt) -> UInt {
    a << b
}

#[cube]
pub fn shr_op(a: UInt, b: UInt) -> UInt {
    a >> b
}

#[cube]
pub fn add_assign_op<T: Numeric>(mut a: T, b: T) {
    a += b;
}

#[cube]
pub fn sub_assign_op<T: Numeric>(mut a: T, b: T) {
    a -= b;
}

#[cube]
pub fn mul_assign_op<T: Numeric>(mut a: T, b: T) {
    a *= b;
}

#[cube]
pub fn div_assign_op<T: Numeric>(mut a: T, b: T) {
    a /= b;
}

mod tests {
    use super::*;
    use burn_cube::ir::{Elem, FloatKind, Item};

    macro_rules! binary_test {
        ($test_name:ident, $op_expand:expr, $op_name:expr, $func:ident) => {
            #[test]
            fn $test_name() {
                let mut context = CubeContext::root();
                let x = context.create_local(Item::new(Elem::Float(FloatKind::F32)));
                let y = context.create_local(Item::new(Elem::Float(FloatKind::F32)));

                $op_expand(&mut context, x, y);

                assert_eq!(
                    format!("{:?}", context.into_scope().operations),
                    $func($op_name)
                );
            }
        };
    }

    macro_rules! unary_test {
        ($test_name:ident, $op_expand:expr, $op_name:expr) => {
            #[test]
            fn $test_name() {
                let mut context = CubeContext::root();
                let x = context.create_local(Item::new(Elem::Float(FloatKind::F32)));

                $op_expand(&mut context, x);

                assert_eq!(
                    format!("{:?}", context.into_scope().operations),
                    ref_ops_unary($op_name)
                );
            }
        };
    }

    macro_rules! binary_boolean_test {
        ($test_name:ident, $op_expand:expr, $op_name:expr) => {
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
        ($test_name:ident, $op_expand:expr, $op_name:expr) => {
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

    binary_test!(cube_can_add, add_op::__expand::<F32>, "Add", ref_ops_binary);
    binary_test!(cube_can_sub, sub_op::__expand::<F32>, "Sub", ref_ops_binary);
    binary_test!(cube_can_mul, mul_op::__expand::<F32>, "Mul", ref_ops_binary);
    binary_test!(cube_can_div, div_op::__expand::<F32>, "Div", ref_ops_binary);
    unary_test!(cube_can_abs, abs_op::__expand::<F32>, "Abs");
    unary_test!(cube_can_exp, exp_op::__expand::<F32>, "Exp");
    unary_test!(cube_can_log, log_op::__expand::<F32>, "Log");
    unary_test!(cube_can_log1p, log1p_op::__expand::<F32>, "Log1p");
    unary_test!(cube_can_cos, cos_op::__expand::<F32>, "Cos");
    unary_test!(cube_can_sin, sin_op::__expand::<F32>, "Sin");
    unary_test!(cube_can_tanh, tanh_op::__expand::<F32>, "Tanh");
    binary_test!(
        cube_can_powf,
        powf_op::__expand::<F32>,
        "Powf",
        ref_ops_binary
    );
    unary_test!(cube_can_sqrt, sqrt_op::__expand::<F32>, "Sqrt");
    unary_test!(cube_can_erf, erf_op::__expand::<F32>, "Erf");
    unary_test!(cube_can_recip, recip_op::__expand::<F32>, "Recip");
    unary_test!(cube_can_floor, floor_op::__expand::<F32>, "Floor");
    unary_test!(cube_can_ceil, ceil_op::__expand::<F32>, "Ceil");
    binary_test!(cube_can_eq, equal_op::__expand::<F32>, "Equal", ref_ops_cmp);
    binary_test!(
        cube_can_ne,
        not_equal_op::__expand::<F32>,
        "NotEqual",
        ref_ops_cmp
    );
    binary_test!(cube_can_lt, lower_op::__expand::<F32>, "Lower", ref_ops_cmp);
    binary_test!(
        cube_can_le,
        lower_equal_op::__expand::<F32>,
        "LowerEqual",
        ref_ops_cmp
    );
    binary_test!(
        cube_can_ge,
        greater_equal_op::__expand::<F32>,
        "GreaterEqual",
        ref_ops_cmp
    );
    binary_test!(
        cube_can_gt,
        greater_op::__expand::<F32>,
        "Greater",
        ref_ops_cmp
    );
    binary_test!(cube_can_max, max_op::__expand::<F32>, "Max", ref_ops_binary);
    binary_test!(cube_can_min, min_op::__expand::<F32>, "Min", ref_ops_binary);
    binary_test!(
        cube_can_add_assign,
        add_assign_op::__expand::<F32>,
        "Add",
        ref_ops_binary
    );
    binary_test!(
        cube_can_sub_assign,
        sub_assign_op::__expand::<F32>,
        "Sub",
        ref_ops_binary
    );
    binary_test!(
        cube_can_mul_assign,
        mul_assign_op::__expand::<F32>,
        "Mul",
        ref_ops_binary
    );
    binary_test!(
        cube_can_div_assign,
        div_assign_op::__expand::<F32>,
        "Div",
        ref_ops_binary
    );
    binary_boolean_test!(cube_can_and, and_op::__expand, "And");
    binary_boolean_test!(cube_can_or, or_op::__expand, "Or");
    binary_uint_test!(cube_can_bitand, bitand_op::__expand, "BitwiseAnd");
    binary_uint_test!(cube_can_bitxor, bitxor_op::__expand, "BitwiseXor");
    binary_uint_test!(cube_can_shl, shl_op::__expand, "ShiftLeft");
    binary_uint_test!(cube_can_shr, shr_op::__expand, "ShiftRight");
    binary_uint_test!(cube_can_mod, modulo_op::__expand, "Modulo");
    binary_test!(
        cube_can_rem,
        remainder_op::__expand::<F32>,
        "Remainder",
        ref_ops_binary
    );

    #[test]
    fn cube_can_not() {
        let mut context = CubeContext::root();
        let x = context.create_local(Item::new(Elem::Bool));

        not_op::__expand(&mut context, x);

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
                lhs: Local {{ id: 0, item: Item {{ \
                    elem: {in_type}, \
                    vectorization: 1 \
                }}, depth: 0 }}, \
                rhs: Local {{ id: 1, item: Item {{ \
                    elem: {in_type}, \
                    vectorization: 1 \
                }}, depth: 0 }}, \
                out: Local {{ id: {out_number}, item: Item {{ \
                    elem: {out_type}, \
                    vectorization: 1 \
                }}, depth: 0 }} \
            }}))]"
            )
        } else {
            format!(
                "[Operator({ops_name}(UnaryOperator {{ \
                input: Local {{ id: 0, item: Item {{ \
                    elem: {in_type}, \
                    vectorization: 1 \
                }}, depth: 0 }}, \
                out: Local {{ id: 0, item: Item {{ \
                    elem: {out_type}, \
                    vectorization: 1 \
                }}, depth: 0 }} \
            }}))]"
            )
        }
    }
}
