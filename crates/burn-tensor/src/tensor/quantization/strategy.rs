use core::{
    hash::{Hash, Hasher},
    marker::PhantomData,
};

use alloc::vec::Vec;
use burn_common::{iter_slice_par, run_par};
use num_traits::{Float, PrimInt};
use serde::{Deserialize, Serialize};

use super::{QuantizationMode, QuantizationScheme, QuantizationType};

/// Quantization strategy.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationStrategy {
    /// Per-tensor `int8` affine/asymmetric quantization.
    PerTensorAffineInt8(AffineQuantization<f32, i8, i32>),
    /// Per-tensor `int8` symmetric quantization.
    PerTensorSymmetricInt8(SymmetricQuantization<f32, i8>),
}

impl QuantizationStrategy {
    /// Returns the corresponding quantization scheme.
    pub fn scheme(&self) -> QuantizationScheme {
        match self {
            QuantizationStrategy::PerTensorAffineInt8(_) => {
                QuantizationScheme::PerTensor(QuantizationMode::Affine, QuantizationType::QInt8)
            }
            QuantizationStrategy::PerTensorSymmetricInt8(_) => {
                QuantizationScheme::PerTensor(QuantizationMode::Symmetric, QuantizationType::QInt8)
            }
        }
    }
}

/// Quantization scheme to convert elements of a higher precision data type `E` to a lower precision
/// data type `Q` and vice-versa.
pub trait Quantization<E: Float + Send + Sync, Q: PrimInt + Send + Sync> {
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
pub struct AffineQuantization<E: Float + Send + Sync, Q: PrimInt + Send + Sync, A: PrimInt> {
    /// The scaling factor.
    pub scale: E,
    /// The zero-point offset.
    pub offset: Q,
    /// Accumulation type.
    _a: PhantomData<A>,
}

fn valid_scale<E: Float>(mut scale: E) -> E {
    // If scale is 0 (most likely due to a tensor full of zeros), we arbitrarily adjust the
    // scale to 0.1 to avoid division by zero.
    if scale.eq(&E::zero()) {
        scale = E::from(0.1).unwrap();
    }
    scale
}

impl<E: Float + Send + Sync, Q: PrimInt + Send + Sync, A: PrimInt> AffineQuantization<E, Q, A> {
    /// Initialize an affine quantization scheme with the given parameters.
    pub fn init(scale: E, offset: Q) -> Self {
        Self {
            scale: valid_scale(scale),
            offset,
            _a: PhantomData,
        }
    }
}

impl<E: Float + Send + Sync, Q: PrimInt + Send + Sync, A: PrimInt + Send + Sync> Quantization<E, Q>
    for AffineQuantization<E, Q, A>
{
    fn new(alpha: E, beta: E) -> Self {
        // Q range `[a, b]`
        let a = E::from(Q::min_value()).unwrap();
        let b = E::from(Q::max_value()).unwrap();

        // We extend the `[alpha, beta]` interval to ensure that it contains 0.
        // Otherwise, we would not meet the requirement that 0 be an exactly
        // representable value (zero-point).
        let alpha = E::min(alpha, E::zero());
        let beta = E::max(beta, E::zero());

        // Compute scale and offset to convert a floating point value in range `[alpha, beta]` to the quantized range
        let scale = valid_scale((beta - alpha) / (b - a));
        let z = -(alpha / scale - a);
        Self {
            scale,
            offset: Q::from(z).unwrap(),
            _a: PhantomData,
        }
    }

    fn quantize(&self, values: &[E]) -> Vec<Q> {
        // Quantized range `[a, b]`
        let a = E::from(Q::min_value()).unwrap();
        let b = E::from(Q::max_value()).unwrap();

        // x_q = clamp(round(x / scale + offset), a, b)
        let z = E::from(self.offset).unwrap();
        run_par!(|| {
            iter_slice_par!(values)
                .map(|x| Q::from(x.div(self.scale).add(z).round().clamp(a, b)).unwrap())
                .collect()
        })
    }

    fn dequantize(&self, values: &[Q]) -> Vec<E> {
        // x = scale * (x_q - offset)
        run_par!(|| {
            iter_slice_par!(values)
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
        })
    }
}

/// Symmetric quantization scheme.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SymmetricQuantization<E: Float + Send + Sync, Q: PrimInt + Send + Sync> {
    /// The scaling factor.
    pub scale: E,
    /// The quantized type.
    _q: PhantomData<Q>,
}

impl<E: Float + Send + Sync, Q: PrimInt + Send + Sync> SymmetricQuantization<E, Q> {
    /// Initialize a symmetric quantization scheme with the given parameters.
    pub fn init(scale: E) -> Self {
        Self {
            scale: valid_scale(scale),
            _q: PhantomData,
        }
    }
}

impl<E: Float + Send + Sync, Q: PrimInt + Send + Sync> Quantization<E, Q>
    for SymmetricQuantization<E, Q>
{
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
        let scale = valid_scale((alpha + alpha) / (b - a));
        Self {
            scale,
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

impl<E: Float + Send + Sync, Q: PrimInt + Hash + Send + Sync, A: PrimInt> Hash
    for AffineQuantization<E, Q, A>
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash raw bits.
        let bits = raw_double_bits(&canonicalize_signed_zero(self.scale));
        bits.hash(state);
        self.offset.hash(state);
    }
}

impl<E: Float + Send + Sync, Q: PrimInt + Send + Sync, A: PrimInt> PartialEq
    for AffineQuantization<E, Q, A>
{
    fn eq(&self, other: &Self) -> bool {
        self.scale == other.scale && self.offset == other.offset
    }
}

impl<E: Float + Send + Sync, Q: PrimInt + Send + Sync, A: PrimInt> Eq
    for AffineQuantization<E, Q, A>
{
}

impl<E: Float + Send + Sync, Q: PrimInt + Send + Sync> Hash for SymmetricQuantization<E, Q> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash raw bits.
        let bits = raw_double_bits(&canonicalize_signed_zero(self.scale));
        bits.hash(state);
    }
}

impl<E: Float + Send + Sync, Q: PrimInt + Send + Sync> PartialEq for SymmetricQuantization<E, Q> {
    fn eq(&self, other: &Self) -> bool {
        self.scale == other.scale
    }
}

impl<E: Float + Send + Sync, Q: PrimInt + Send + Sync> Eq for SymmetricQuantization<E, Q> {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_int8_affine_quantization() {
        let x: [f32; 4] = [-1.8, -1.0, 0.0, 0.5];
        let expected_q = vec![-128, -40, 71, 126];
        let expected_d = vec![-1.794902, -1.0011765, 0.0, 0.49607843];

        let affine = AffineQuantization::<f32, i8, i32>::new(-1.8, 0.5);

        let q = affine.quantize(&x);
        assert_eq!(q, expected_q);

        let d = affine.dequantize(&expected_q);

        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_affine_should_ensure_zero_point() {
        let x: [f32; 6] = [2.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let expected_q = vec![-26, -77, -26, 25, 76, 127];
        let expected_d = x.to_vec();

        let affine = AffineQuantization::<f32, i8, i32>::new(1.0, 5.0);

        assert_eq!(affine.offset, -128);
        assert_eq!(affine.scale, 0.019607844);

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
