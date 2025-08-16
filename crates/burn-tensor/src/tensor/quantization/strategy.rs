use alloc::vec::Vec;
use core::marker::PhantomData;
use num_traits::{Float, PrimInt, Signed};
use serde::{Deserialize, Serialize};

use super::{QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue};

/// Quantization strategy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationStrategy {
    /// Per-tensor `int8` symmetric quantization.
    PerTensorSymmetricInt8(SymmetricQuantization<f32, i8>),
    /// Per-block `int8` symmetric quantization.
    PerBlockSymmetricInt8(Vec<SymmetricQuantization<f32, i8>>, usize),
}

impl QuantizationStrategy {
    /// Quantize the values to a lower precision data type.
    pub fn quantize(&self, values: &[f32]) -> Vec<i8> {
        match self {
            QuantizationStrategy::PerTensorSymmetricInt8(strategy) => strategy.quantize(values),
            QuantizationStrategy::PerBlockSymmetricInt8(strategy, block_size) => {
                let num_blocks = strategy.len();
                let numel = values.len();
                assert_eq!(
                    numel / block_size,
                    num_blocks,
                    "Invalid per-block quantization with num blocks {num_blocks} and {numel} values"
                );
                values
                    .chunks(*block_size)
                    .enumerate()
                    .flat_map(|(block_id, block)| strategy[block_id].quantize(block))
                    .collect()
            }
        }
    }

    /// Dequantize the values to a higher precision data type.
    pub fn dequantize(&self, values: &[i8]) -> Vec<f32> {
        match self {
            QuantizationStrategy::PerTensorSymmetricInt8(strategy) => strategy.dequantize(values),
            QuantizationStrategy::PerBlockSymmetricInt8(strategy, block_size) => {
                let num_blocks = strategy.len();
                let numel = values.len();
                assert_eq!(
                    numel / block_size,
                    num_blocks,
                    "Invalid per-block quantization with block size {block_size}, num blocks {num_blocks} and {numel} values"
                );
                values
                    .chunks(*block_size)
                    .enumerate()
                    .flat_map(|(block_id, block)| strategy[block_id].dequantize(block))
                    .collect()
            }
        }
    }
}

impl QuantizationStrategy {
    /// Returns the corresponding quantization scheme.
    pub fn scheme(&self) -> QuantScheme {
        match self {
            QuantizationStrategy::PerTensorSymmetricInt8(_) => QuantScheme {
                level: QuantLevel::Tensor,
                mode: QuantMode::Symmetric,
                value: QuantValue::QInt8,
                store: QuantStore::U32,
                param: QuantParam::F32,
            },
            QuantizationStrategy::PerBlockSymmetricInt8(_blocks, block_size) => QuantScheme {
                level: QuantLevel::Block(*block_size),
                mode: QuantMode::Symmetric,
                value: QuantValue::QInt8,
                store: QuantStore::U32,
                param: QuantParam::F32,
            },
        }
    }
}

/// Quantization scheme to convert elements of a higher precision data type `E` to a lower precision
/// data type `Q` and vice-versa.
pub trait Quantization<E: Float + Send + Sync, Q: PrimInt + Send + Sync> {
    /// Returns the quantization range `[a, b]`.
    fn range() -> (Q, Q);
    /// Create a new quantization scheme for an input range `[alpha, beta]`.
    fn new(alpha: E, beta: E) -> Self;
    /// Convert the values to a lower precision data type.
    fn quantize(&self, values: &[E]) -> Vec<Q>;
    /// Convert a single value to a lower precision data type.
    fn quantize_one(&self, value: E) -> Q;
    /// Convert the values back to a higher precision data type.
    fn dequantize(&self, values: &[Q]) -> Vec<E>;
    /// Convert a single value back to a higher precision data type.
    fn dequantize_one(&self, value: Q) -> E;
}

fn valid_scale<E: Float>(mut scale: E) -> E {
    // If scale is 0 (most likely due to a tensor full of zeros), we arbitrarily adjust the
    // scale to 0.1 to avoid division by zero.
    if scale.eq(&E::zero()) {
        scale = E::from(0.1).unwrap();
    }
    scale
}

/// Symmetric quantization scheme.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SymmetricQuantization<E: Float + Send + Sync, Q: PrimInt + Signed + Send + Sync> {
    /// The scaling factor.
    pub scale: E,
    /// The quantized type.
    _q: PhantomData<Q>,
}

impl<E: Float + Send + Sync, Q: PrimInt + Signed + Send + Sync> SymmetricQuantization<E, Q> {
    /// Initialize a symmetric quantization scheme with the given parameters.
    pub fn init(scale: E) -> Self {
        Self {
            scale: valid_scale(scale),
            _q: PhantomData,
        }
    }
}

impl<E: Float + Send + Sync, Q: PrimInt + Signed + Send + Sync> Quantization<E, Q>
    for SymmetricQuantization<E, Q>
{
    fn new(alpha: E, beta: E) -> Self {
        let (a, b) = Self::range();
        let a = E::from(a).unwrap();
        let b = E::from(b).unwrap();

        // Compute scale to convert a floating point value in range `[-alpha, alpha]` to the quantized range
        let alpha = alpha.abs().max(beta.abs());
        let scale = valid_scale((alpha + alpha) / (b - a));
        Self {
            scale,
            _q: PhantomData,
        }
    }

    fn quantize(&self, values: &[E]) -> Vec<Q> {
        values.iter().map(|x| self.quantize_one(*x)).collect()
    }

    fn dequantize(&self, values: &[Q]) -> Vec<E> {
        values.iter().map(|x_q| self.dequantize_one(*x_q)).collect()
    }

    fn quantize_one(&self, value: E) -> Q {
        let (a, b) = Self::range();
        let a = E::from(a).unwrap();
        let b = E::from(b).unwrap();

        // x_q = clamp(round(x / scale), a, b)
        Q::from(value.div(self.scale).round().clamp(a, b)).unwrap()
    }

    fn dequantize_one(&self, value: Q) -> E {
        // x = scale * x_q
        self.scale * E::from(value).unwrap()
    }

    fn range() -> (Q, Q) {
        // Only implemented for symmetric *signed* at this time
        let b = Q::max_value();
        (b.neg(), b)
    }
}

impl<E: Float + Send + Sync, Q: PrimInt + Signed + Send + Sync> PartialEq
    for SymmetricQuantization<E, Q>
{
    fn eq(&self, other: &Self) -> bool {
        self.scale == other.scale
    }
}

impl<E: Float + Send + Sync, Q: PrimInt + Signed + Send + Sync> Eq for SymmetricQuantization<E, Q> {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

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

    #[test]
    fn test_int8_symmetric_quantization_per_block() {
        let x: [f32; 8] = [-1.8, -1.0, 0.0, 0.5, -1.8, -1.0, 0.0, 0.5];
        let expected_q = vec![-127, -71, 0, 35, -127, -71, 0, 35];
        let expected_d = vec![
            -1.8, -1.0062993, 0.0, 0.496063, -1.8, -1.0062993, 0.0, 0.496063,
        ];

        let symmetric = SymmetricQuantization::<f32, i8>::new(-1.8, 0.5);
        let strategy = QuantizationStrategy::PerBlockSymmetricInt8(vec![symmetric, symmetric], 4);

        let q: Vec<i8> = strategy.quantize(&x);
        assert_eq!(q, expected_q);

        let d = symmetric.dequantize(&expected_q);

        assert_eq!(d, expected_d);
    }
}
