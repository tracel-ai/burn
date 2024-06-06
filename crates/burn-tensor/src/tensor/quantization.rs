use core::{cmp::Ordering, marker::PhantomData};

use alloc::vec::Vec;
use num_traits::{Float, PrimInt};
use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffineQuantization<E: Float, Q: PrimInt, A: PrimInt> {
    /// The scaling factor.
    pub scale: E,
    /// The zero-point offset.
    pub offset: Q,
    /// Accumulation type.
    _a: PhantomData<A>,
}

impl<E: Float, Q: PrimInt, A: PrimInt> Eq for AffineQuantization<E, Q, A> {}

impl<E: Float, Q: PrimInt, A: PrimInt> PartialEq for AffineQuantization<E, Q, A> {
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (
                self.scale.partial_cmp(&other.scale),
                self.offset.cmp(&other.offset)
            ),
            (Some(Ordering::Equal), Ordering::Equal)
        )
    }
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
///
/// Note that symmetric quantization is only valid for signed integers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymmetricQuantization<E: Float, Q: PrimInt> {
    /// The scaling factor.
    pub scale: E,
    _q: PhantomData<Q>,
}

impl<E: Float, Q: PrimInt> Eq for SymmetricQuantization<E, Q> {}

impl<E: Float, Q: PrimInt> PartialEq for SymmetricQuantization<E, Q> {
    fn eq(&self, other: &Self) -> bool {
        matches!(self.scale.partial_cmp(&other.scale), Some(Ordering::Equal))
    }
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

        // Compute scale to convert a floating point value in range `[alpha, beta]` to the quantized range
        Self {
            scale: (beta - alpha) / (b - a),
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

/// Quantization scheme/strategy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationStrategy {
    /// `int8` affine/asymmetric quantization.
    Int8Affine(AffineQuantization<f32, i8, i32>),
    /// `int8` symmetric quantization.
    Int8Symmetric(SymmetricQuantization<f32, i8>),
}

impl QuantizationStrategy {
    /// Convert the values to a lower precision data type.
    pub fn quantize(&self, values: &[f32]) -> Vec<i8> {
        match self {
            Self::Int8Affine(m) => m.quantize(values),
            Self::Int8Symmetric(m) => m.quantize(values),
        }
    }

    /// Convert the values back to a higher precision data type.
    pub fn dequantize(&self, values: &[i8]) -> Vec<f32> {
        match self {
            Self::Int8Affine(m) => m.dequantize(values),
            Self::Int8Symmetric(m) => m.dequantize(values),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_affine_quantization() {
        let x: [f32; 4] = [-1.8, -1.0, 0.0, 0.5];
        let expected_q = vec![-128, -39, 72, 127];
        let expected_d = vec![-1.8039216, -1.0011765, 0.0, 0.49607843];

        let affine = QuantizationStrategy::Int8Affine(AffineQuantization::new(-1.8, 0.5));

        let q = affine.quantize(&x);
        assert_eq!(q, expected_q);

        let d = affine.dequantize(&expected_q);

        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_int8_symmetric_quantization() {
        let x: [f32; 4] = [-1.8, -1.0, 0.0, 0.5];
        let expected_q = vec![-127, -110, 0, 55];
        let expected_d = vec![-1.15, -0.996063, 0.0, 0.4980315];

        let symmetric = QuantizationStrategy::Int8Symmetric(SymmetricQuantization::new(-1.8, 0.5));

        let q = symmetric.quantize(&x);
        assert_eq!(q, expected_q);

        let d = symmetric.dequantize(&expected_q);

        assert_eq!(d, expected_d);
    }
}
