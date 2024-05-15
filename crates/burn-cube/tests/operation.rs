use burn_cube::{branch::*, cube, Bool, CubeContext, Float, Numeric, PrimitiveVariable, UInt, F32};
use burn_jit::{
    gpu,
    gpu::{Elem, Item, Variable},
};

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
    abs(a)
}

#[cube]
fn exp_op<F: Float>(a: F) -> F {
    exp(a)
}

#[cube]
fn log_op<F: Float>(a: F) -> F {
    log(a)
}

#[cube]
fn log1p_op<F: Float>(a: F) -> F {
    log1p(a)
}

#[cube]
fn cos_op<F: Float>(a: F) -> F {
    cos(a)
}

#[cube]
fn sin_op<F: Float>(a: F) -> F {
    sin(a)
}

#[cube]
fn tanh_op<F: Float>(a: F) -> F {
    tanh(a)
}

#[cube]
fn powf_op<F: Float>(a: F, b: F) -> F {
    powf(a, b)
}

#[cube]
fn sqrt_op<F: Float>(a: F) -> F {
    sqrt(a)
}

#[cube]
fn floor_op<F: Float>(a: F) -> F {
    floor(a)
}

#[cube]
fn ceil_op<F: Float>(a: F) -> F {
    ceil(a)
}

#[cube]
fn erf_op<F: Float>(a: F) -> F {
    erf(a)
}

#[cube]
fn recip_op<F: Float>(a: F) -> F {
    recip(a)
}

#[cube]
fn equal_op<T: PrimitiveVariable>(a: T, b: T) -> Bool {
    a == b
}

#[cube]
fn not_equal_op<T: PrimitiveVariable>(a: T, b: T) -> Bool {
    a != b
}

#[cube]
fn lower_op<T: Numeric>(a: T, b: T) -> Bool {
    a < b
}

#[cube]
fn greater_op<T: Numeric>(a: T, b: T) -> Bool {
    a > b
}

#[cube]
fn lower_equal_op<T: Numeric>(a: T, b: T) -> Bool {
    a <= b
}

#[cube]
fn greater_equal_op<T: Numeric>(a: T, b: T) -> Bool {
    a >= b
}

#[cube]
fn clamp_op<T: Numeric>(a: T, l: T, u: T) -> T {
    clamp(a, l, u)
}

#[cube]
fn modulo_op(a: UInt, b: UInt) -> UInt {
    a % b
}

#[cube]
fn remainder_op<T: Numeric>(a: T, b: T) -> T {
    rem(a, b)
}

#[cube]
fn max_op<T: Numeric>(a: T, b: T) -> T {
    max(a, b)
}

#[cube]
fn min_op<T: Numeric>(a: T, b: T) -> T {
    min(a, b)
}

#[cube]
fn and_op(a: Bool, b: Bool) -> Bool {
    a & b
}

#[cube]
fn or_op(a: Bool, b: Bool) -> Bool {
    a | b
}

#[cube]
fn not_op(a: Bool) -> Bool {
    !a
}

#[cube]
fn bit_and_op(a: UInt, b: UInt) -> UInt {
    a & b
}

#[cube]
fn bit_or_op(a: UInt, b: UInt) -> UInt {
    a | b
}

#[cube]
fn shift_left_op(a: UInt, b: UInt) -> UInt {
    a << b
}

#[cube]
fn shift_left_op(a: UInt, b: UInt) -> UInt {
    a >> b
}
