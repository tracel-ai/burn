use core::{
    hash::{Hash, Hasher},
    marker::PhantomData,
};

use alloc::vec::Vec;
use num_traits::{Float, PrimInt};
use serde::{Deserialize, Serialize};

/// Quantization scheme/strategy.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationStrategy {
    /// Per-tensor `int8` affine/asymmetric quantization.
    PerTensorAffineInt8(AffineQuantization<f32, i8, i32>),
    /// Per-tensor `int8` symmetric quantization.
    PerTensorSymmetricInt8(SymmetricQuantization<f32, i8>),
}

/// Quantization scheme to convert elements of a higher precision data type `E` to a lower precision
/// data type `Q` and vice-versa.
pub trait Quantization<E: Float, Q: PrimInt> {
    /// Create a new quantization scheme for an input range `[alpha, beta]`.
    fn new(alpha: E, beta: E) -> Self;
    /// Convert the values to a lower precision data type.
    fn quantize(&self, values: &[E]) -> Vec<Q>;
    /// Convert the values back to a higher precision data type.
    fn dequantize(&self, values: &[Q]) -> Vec<E>;
}

/// Affine quantization scheme.
///
/// Note that the accumulation type `A` should have a bigger range than quantized type `Q`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AffineQuantization<E: Float, Q: PrimInt, A: PrimInt> {
    /// The scaling factor.
    pub scale: E,
    /// The zero-point offset.
    pub offset: Q,
    /// Accumulation type.
    _a: PhantomData<A>,
}

impl<E: Float, Q: PrimInt, A: PrimInt> Quantization<E, Q> for AffineQuantization<E, Q, A> {
    fn new(alpha: E, beta: E) -> Self {
        // Q range `[a, b]`
        let a = E::from(Q::min_value()).unwrap();
        let b = E::from(Q::max_value()).unwrap();

        // Compute scale and offset to convert a floating point value in range `[alpha, beta]` to the quantized range
        let range = beta - alpha;
        Self {
            scale: range / (b - a),
            offset: Q::from(E::round(((beta * a) - (alpha * b)) / range)).unwrap(),
            _a: PhantomData,
        }
    }

    fn quantize(&self, values: &[E]) -> Vec<Q> {
        // Quantized range `[a, b]`
        let a = E::from(Q::min_value()).unwrap();
        let b = E::from(Q::max_value()).unwrap();

        // x_q = clamp(round(x / scale + offset), a, b)
        let z = E::from(self.offset).unwrap();
        values
            .iter()
            .map(|x| Q::from(x.div(self.scale).add(z).round().clamp(a, b)).unwrap())
            .collect()
    }

    fn dequantize(&self, values: &[Q]) -> Vec<E> {
        // x = scale * (x_q - offset)
        values
            .iter()
            .map(|x_q| {
                self.scale
                    * (E::from(
                        A::from(*x_q)
                            .unwrap()
                            .saturating_sub(A::from(self.offset).unwrap()),
                    )
                    .unwrap())
            })
            .collect()
    }
}

/// Symmetric quantization scheme.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SymmetricQuantization<E: Float, Q: PrimInt> {
    /// The scaling factor.
    pub scale: E,
    /// The quantized type.
    _q: PhantomData<Q>,
}

impl<E: Float, Q: PrimInt> Quantization<E, Q> for SymmetricQuantization<E, Q> {
    fn new(alpha: E, beta: E) -> Self {
        assert!(
            !Q::min_value().is_zero(),
            "Symmetric quantization is only valid for signed integers."
        );

        // Quantized range `[a, b]`
        let b = E::from(Q::max_value()).unwrap();
        let a = b.neg();

        // Compute scale to convert a floating point value in range `[-alpha, alpha]` to the quantized range
        let alpha = alpha.abs().max(beta.abs());
        Self {
            scale: (alpha + alpha) / (b - a),
            _q: PhantomData,
        }
    }

    fn quantize(&self, values: &[E]) -> Vec<Q> {
        // Quantized range [a, b]
        let b = E::from(Q::max_value()).unwrap();
        let a = b.neg();

        // x_q = clamp(round(x / scale), a, b)
        values
            .iter()
            .map(|x| Q::from(x.div(self.scale).round().clamp(a, b)).unwrap())
            .collect()
    }

    fn dequantize(&self, values: &[Q]) -> Vec<E> {
        // x = scale * x_q
        values
            .iter()
            .map(|x_q| self.scale * E::from(*x_q).unwrap())
            .collect()
    }
}

// Masks for the parts of the IEEE 754 float
const SIGN_MASK: u64 = 0x8000000000000000u64;
const EXP_MASK: u64 = 0x7ff0000000000000u64;
const MAN_MASK: u64 = 0x000fffffffffffffu64;

#[inline]
/// Used for hashing. Input must not be zero or NaN.
/// Adapted from: https://github.com/reem/rust-ordered-float/blob/master/src/lib.rs
fn raw_double_bits<F: Float>(f: &F) -> u64 {
    let (man, exp, sign) = f.integer_decode();
    let exp_u64 = exp as u16 as u64;
    let sign_u64 = (sign > 0) as u64;
    (man & MAN_MASK) | ((exp_u64 << 52) & EXP_MASK) | ((sign_u64 << 63) & SIGN_MASK)
}

#[inline(always)]
fn canonicalize_signed_zero<T: Float>(x: T) -> T {
    // -0.0 + 0.0 == +0.0 under IEEE754 roundTiesToEven rounding mode,
    // which Rust guarantees. Thus by adding a positive zero we
    // canonicalize signed zero without any branches in one instruction.
    x + T::zero()
}

impl<E: Float, Q: PrimInt + Hash, A: PrimInt> Hash for AffineQuantization<E, Q, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash raw bits.
        let bits = raw_double_bits(&canonicalize_signed_zero(self.scale));
        bits.hash(state);
        self.offset.hash(state);
    }
}

impl<E: Float, Q: PrimInt, A: PrimInt> PartialEq for AffineQuantization<E, Q, A> {
    fn eq(&self, other: &Self) -> bool {
        self.scale == other.scale && self.offset == other.offset
    }
}

impl<E: Float, Q: PrimInt, A: PrimInt> Eq for AffineQuantization<E, Q, A> {}

impl<E: Float, Q: PrimInt> Hash for SymmetricQuantization<E, Q> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash raw bits.
        let bits = raw_double_bits(&canonicalize_signed_zero(self.scale));
        bits.hash(state);
    }
}

impl<E: Float, Q: PrimInt> PartialEq for SymmetricQuantization<E, Q> {
    fn eq(&self, other: &Self) -> bool {
        self.scale == other.scale
    }
}

impl<E: Float, Q: PrimInt> Eq for SymmetricQuantization<E, Q> {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_int8_affine_quantization() {
        let x: [f32; 4] = [-1.8, -1.0, 0.0, 0.5];
        let expected_q = vec![-128, -39, 72, 127];
        let expected_d = vec![-1.8039216, -1.0011765, 0.0, 0.49607843];

        let affine = AffineQuantization::<f32, i8, i32>::new(-1.8, 0.5);

        let q = affine.quantize(&x);
        assert_eq!(q, expected_q);

        let d = affine.dequantize(&expected_q);

        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_int8_symmetric_quantization() {
        let x: [f32; 4] = [-1.8, -1.0, 0.0, 0.5];
        let expected_q = vec![-127, -71, 0, 35];
        let expected_d = vec![-1.8, -1.0062993, 0.0, 0.496063];

        let symmetric = SymmetricQuantization::<f32, i8>::new(-1.8, 0.5);

        let q: Vec<i8> = symmetric.quantize(&x);
        assert_eq!(q, expected_q);

        let d = symmetric.dequantize(&expected_q);

        assert_eq!(d, expected_d);
    }
}
