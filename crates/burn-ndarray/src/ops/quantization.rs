use alloc::vec::Vec;
use num_traits::{Float, PrimInt};

use burn_backend::quantization::{BlockSize, QuantValue};

// NOTE: this mainly serves as a simple reference implementation.
// The de/quantization ops should be refactored to use ndarray.

/// Quantization strategy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantizationStrategy {
    /// Per-tensor symmetric quantization.
    PerTensorSymmetric(SymmetricQuantization<f32>),
    /// Per-block symmetric quantization.
    PerBlockSymmetric(Vec<SymmetricQuantization<f32>>, BlockSize),
}

impl QuantizationStrategy {
    /// Quantize the values to a lower precision data type.
    pub fn quantize(&self, values: &[f32]) -> Vec<i8> {
        match self {
            QuantizationStrategy::PerTensorSymmetric(strategy) => strategy.quantize(values),
            QuantizationStrategy::PerBlockSymmetric(strategy, block_size) => {
                let block_elems = block_size.num_elements();
                let num_blocks = strategy.len();
                let numel = values.len();
                assert_eq!(
                    numel / block_elems,
                    num_blocks,
                    "Invalid per-block quantization with num blocks {num_blocks} and {numel} values"
                );
                values
                    .chunks(block_elems)
                    .enumerate()
                    .flat_map(|(block_id, block)| strategy[block_id].quantize(block))
                    .collect()
            }
        }
    }

    /// Dequantize the values to a higher precision data type.
    pub fn dequantize(&self, values: &[i8]) -> Vec<f32> {
        match self {
            QuantizationStrategy::PerTensorSymmetric(strategy) => strategy.dequantize(values),
            QuantizationStrategy::PerBlockSymmetric(strategy, block_size) => {
                let block_elems = block_size.num_elements();
                let num_blocks = strategy.len();
                let numel = values.len();
                assert_eq!(
                    numel / block_elems,
                    num_blocks,
                    "Invalid per-block quantization with block size {block_elems}, num blocks {num_blocks} and {numel} values"
                );
                values
                    .chunks(block_elems)
                    .enumerate()
                    .flat_map(|(block_id, block)| strategy[block_id].dequantize(block))
                    .collect()
            }
        }
    }
}

/// Quantization scheme to convert elements of a higher precision data type `E` to a lower precision
/// data type `Q` and vice-versa.
pub trait Quantization<E: Float + Send + Sync> {
    /// Returns the quantization range `[a, b]`.
    fn range(&self) -> (E, E);
    /// Convert the values to a lower precision data type.
    fn quantize<Q: PrimInt>(&self, values: &[E]) -> Vec<Q>;
    /// Convert a single value to a lower precision data type.
    fn quantize_one<Q: PrimInt>(&self, value: E) -> Q;
    /// Convert the values back to a higher precision data type.
    fn dequantize<Q: PrimInt>(&self, values: &[Q]) -> Vec<E>;
    /// Convert a single value back to a higher precision data type.
    fn dequantize_one<Q: PrimInt>(&self, value: Q) -> E;
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
#[derive(Debug, Clone, Copy)]
pub struct SymmetricQuantization<E: Float + Send + Sync> {
    /// The scaling factor.
    pub scale: E,
    // The quantization value data type.
    value: QuantValue,
}

impl<E: Float + Send + Sync> SymmetricQuantization<E> {
    /// Initialize a symmetric quantization scheme with the given parameters.
    pub fn init(scale: E, value: QuantValue) -> Self {
        Self {
            scale: valid_scale(scale),
            value,
        }
    }

    #[allow(dead_code)]
    /// Create a new quantization scheme for an input range `[alpha, beta]`.
    fn new(alpha: E, beta: E, value: QuantValue) -> Self {
        let (a, b) = value.range();
        let a = E::from(a).unwrap();
        let b = E::from(b).unwrap();

        // Compute scale to convert a floating point value in range `[-alpha, alpha]` to the quantized range
        let alpha = alpha.abs().max(beta.abs());
        let scale = valid_scale((alpha + alpha) / (b - a));
        Self { scale, value }
    }
}

impl<E: Float + Send + Sync> Quantization<E> for SymmetricQuantization<E> {
    fn quantize<Q: PrimInt>(&self, values: &[E]) -> Vec<Q> {
        values.iter().map(|x| self.quantize_one(*x)).collect()
    }

    fn dequantize<Q: PrimInt>(&self, values: &[Q]) -> Vec<E> {
        values.iter().map(|x_q| self.dequantize_one(*x_q)).collect()
    }

    fn quantize_one<Q: PrimInt>(&self, value: E) -> Q {
        let (a, b) = self.range();

        // x_q = clamp(round(x / scale), a, b)
        Q::from(value.div(self.scale).round().clamp(a, b)).unwrap()
    }

    fn dequantize_one<Q: PrimInt>(&self, value: Q) -> E {
        // x = scale * x_q
        self.scale * E::from(value).unwrap()
    }

    fn range(&self) -> (E, E) {
        let (a, b) = self.value.range();
        let a = E::from(a).unwrap();
        let b = E::from(b).unwrap();
        (a, b)
    }
}

impl<E: Float + Send + Sync> PartialEq for SymmetricQuantization<E> {
    fn eq(&self, other: &Self) -> bool {
        self.scale == other.scale
    }
}

impl<E: Float + Send + Sync> Eq for SymmetricQuantization<E> {}

#[cfg(test)]
mod tests {
    use burn_backend::TensorData;

    use super::*;
    use alloc::vec;

    #[test]
    fn test_int8_symmetric_quantization() {
        let x: [f32; 4] = [-1.8, -1.0, 0.0, 0.5];
        let expected_q = vec![-127, -71, 0, 35];
        let expected_d = vec![-1.8, -1.0062993, 0.0, 0.496063];

        let symmetric = SymmetricQuantization::<f32>::new(-1.8, 0.5, QuantValue::Q8S);

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

        let symmetric = SymmetricQuantization::<f32>::new(-1.8, 0.5, QuantValue::Q8S);
        let strategy = QuantizationStrategy::PerBlockSymmetric(
            vec![symmetric, symmetric],
            BlockSize::new([4]),
        );

        let q: Vec<i8> = strategy.quantize(&x);
        assert_eq!(q, expected_q);

        let d = symmetric.dequantize(&expected_q);

        assert_eq!(d, expected_d);
    }

    #[test]
    fn should_support_dequantize() {
        let strategy = QuantizationStrategy::PerTensorSymmetric(SymmetricQuantization {
            scale: 0.1,
            value: QuantValue::Q8S,
        });

        let output = strategy.dequantize(&[-127i8, -77, -26, 25, 76, 127]);

        let output = TensorData::new(output, [2, 3]);

        output.assert_approx_eq::<f32>(
            &TensorData::from([[-12.7, -7.7, -2.6], [2.5, 7.6, 12.7]]),
            Default::default(),
        );
    }
}
