use alloc::{vec, vec::Vec};
use burn_common::{iter_slice_par, run_par};
use core::marker::PhantomData;
use num_traits::{Float, PrimInt, Signed};
use serde::{Deserialize, Serialize};

use crate::{Element, ElementConversion};

use super::{BlockLayout, QuantizationMode, QuantizationScheme, QuantizationType};

/// Quantization strategy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationStrategy {
    /// Per-tensor `int8` affine/asymmetric quantization.
    PerTensorAffineInt8(AffineQuantization<f32, i8, i32>),
    /// Per-tensor `int8` symmetric quantization.
    PerTensorSymmetricInt8(SymmetricQuantization<f32, i8>),
    /// Per-block `int8` affine/asymmetric quantization.
    PerBlockAffineInt8(Vec<AffineQuantization<f32, i8, i32>>, BlockLayout),
    /// Per-block `int8` symmetric quantization.
    PerBlockSymmetricInt8(Vec<SymmetricQuantization<f32, i8>>, BlockLayout),
}

impl QuantizationStrategy {
    /// Quantize the values to a lower precision data type.
    pub fn quantize(&self, values: &[f32], shape: &[usize]) -> Vec<i8> {
        match self {
            QuantizationStrategy::PerTensorAffineInt8(strategy) => strategy.quantize(values),
            QuantizationStrategy::PerTensorSymmetricInt8(strategy) => strategy.quantize(values),
            QuantizationStrategy::PerBlockAffineInt8(strategy, layout) => match layout {
                BlockLayout::Flat(block_size) => {
                    apply_per_block(values, *block_size as usize, |block_id, v| {
                        strategy[block_id].quantize(v)
                    })
                }
                BlockLayout::Grid(m, n) => {
                    apply_per_block_grid(values, shape, *m as usize, *n as usize, |block_id, v| {
                        strategy[block_id].quantize_one(v)
                    })
                }
            },
            QuantizationStrategy::PerBlockSymmetricInt8(strategy, layout) => match layout {
                BlockLayout::Flat(block_size) => {
                    apply_per_block(values, *block_size as usize, |block_id, v| {
                        strategy[block_id].quantize(v)
                    })
                }
                BlockLayout::Grid(m, n) => {
                    apply_per_block_grid(values, shape, *m as usize, *n as usize, |block_id, v| {
                        strategy[block_id].quantize_one(v)
                    })
                }
            },
        }
    }

    /// Dequantize the values to a higher precision data type.
    pub fn dequantize(&self, values: &[i8], shape: &[usize]) -> Vec<f32> {
        match self {
            QuantizationStrategy::PerTensorAffineInt8(strategy) => strategy.dequantize(values),
            QuantizationStrategy::PerTensorSymmetricInt8(strategy) => strategy.dequantize(values),
            QuantizationStrategy::PerBlockAffineInt8(strategy, layout) => match layout {
                BlockLayout::Flat(block_size) => {
                    apply_per_block(values, *block_size as usize, |block_id, v| {
                        strategy[block_id].dequantize(v)
                    })
                }
                BlockLayout::Grid(m, n) => {
                    apply_per_block_grid(values, shape, *m as usize, *n as usize, |block_id, v| {
                        strategy[block_id].dequantize_one(v)
                    })
                }
            },
            QuantizationStrategy::PerBlockSymmetricInt8(strategy, layout) => match layout {
                BlockLayout::Flat(block_size) => {
                    apply_per_block(values, *block_size as usize, |block_id, v| {
                        strategy[block_id].dequantize(v)
                    })
                }
                BlockLayout::Grid(m, n) => {
                    apply_per_block_grid(values, shape, *m as usize, *n as usize, |block_id, v| {
                        strategy[block_id].dequantize_one(v)
                    })
                }
            },
        }
    }
}

fn apply_per_block_grid<I: Element, O: Element, F: Fn(usize, I) -> O>(
    values: &[I],
    shape: &[usize],
    m: usize,
    n: usize,
    transform: F,
) -> Vec<O> {
    let (b, height, width) = match shape.len() {
        2 => (1, shape[0], shape[1]),
        3 => (shape[0], shape[1], shape[2]),
        _ => unimplemented!("Per-block grid quantization is only supported for 2D or 3D tensors"),
    };
    assert!(
        height % m == 0 && width % n == 0,
        "Invalid per-block quantization with block grid [{m}, {n}] and tensor of shape {shape:?}"
    );
    let mut output = vec![0.elem::<O>(); values.len()];

    let mut block_id = 0;
    // TODO: parallel
    for ih in (0..b * height).step_by(m) {
        for iw in (0..width).step_by(n) {
            // block height
            for bh in 0..m {
                let start_idx = (ih + bh) * width + iw;
                // block width
                for bw in 0..n {
                    let elem_idx = start_idx + bw;
                    let x_q = transform(block_id, values[elem_idx]);
                    output[elem_idx] = x_q;
                }
            }
            block_id += 1;
        }
    }
    output
}

fn apply_per_block<I: Element, O: Element, F: Fn(usize, &[I]) -> Vec<O>>(
    values: &[I],
    block_size: usize,
    transform: F,
) -> Vec<O> {
    let numel = values.len();
    assert_eq!(
        numel % block_size,
        0,
        "Invalid per-block quantization with block size {block_size} and {numel} values"
    );
    // TODO: parallel chunks
    values
        .chunks(block_size)
        .enumerate()
        .flat_map(|(block_id, block)| transform(block_id, block))
        .collect()
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
            QuantizationStrategy::PerBlockSymmetricInt8(_, layout) => QuantizationScheme::PerBlock(
                QuantizationMode::Symmetric,
                QuantizationType::QInt8,
                *layout,
            ),
            QuantizationStrategy::PerBlockAffineInt8(_, layout) => QuantizationScheme::PerBlock(
                QuantizationMode::Affine,
                QuantizationType::QInt8,
                *layout,
            ),
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
        let (a, b) = Self::range();
        let a = E::from(a).unwrap();
        let b = E::from(b).unwrap();

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
        run_par!(|| {
            iter_slice_par!(values)
                .map(|x| self.quantize_one(*x))
                .collect()
        })
    }

    fn dequantize(&self, values: &[Q]) -> Vec<E> {
        run_par!(|| {
            iter_slice_par!(values)
                .map(|x_q| self.dequantize_one(*x_q))
                .collect()
        })
    }

    fn quantize_one(&self, value: E) -> Q {
        let (a, b) = Self::range();
        let a = E::from(a).unwrap();
        let b = E::from(b).unwrap();

        // x_q = clamp(round(x / scale + offset), a, b)
        let z = E::from(self.offset).unwrap();
        Q::from(value.div(self.scale).add(z).round().clamp(a, b)).unwrap()
    }

    fn dequantize_one(&self, value: Q) -> E {
        // x = scale * (x_q - offset)
        self.scale
            * (E::from(
                A::from(value)
                    .unwrap()
                    .saturating_sub(A::from(self.offset).unwrap()),
            )
            .unwrap())
    }

    fn range() -> (Q, Q) {
        (Q::min_value(), Q::max_value())
    }
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

    #[test]
    fn test_int8_symmetric_quantization_per_block_flat() {
        let x: [f32; 8] = [-1.8, -1.0, 0.0, 0.5, -1.8, -1.0, 0.0, 0.5];
        let shape = &[2, 4];
        let expected_q = vec![-127, -71, 0, 35, -127, -71, 0, 35];
        let expected_d = vec![
            -1.8, -1.0062993, 0.0, 0.496063, -1.8, -1.0062993, 0.0, 0.496063,
        ];

        let symmetric = SymmetricQuantization::<f32, i8>::new(-1.8, 0.5);
        let strategy = QuantizationStrategy::PerBlockSymmetricInt8(
            vec![symmetric, symmetric],
            BlockLayout::Flat(4),
        );

        let q: Vec<i8> = strategy.quantize(&x, shape);
        assert_eq!(q, expected_q);

        let d = symmetric.dequantize(&expected_q);

        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_int8_affine_quantization_per_block_flat() {
        let x = [
            [-1.8, -1.0, 0.0, 0.5],
            [-0.8, 1.2, 0.25, 0.5],
            [-8., 12., 2.5, 5.],
            [0.2, 0.3, 0.4, 0.5],
        ]
        .concat();
        let shape = &[2, 8];
        let expected_q = [
            [-128i8, -40, 71, 126],
            [-128, 127, 6, 38],
            [-128, 127, 6, 38],
            [-26, 25, 76, 127],
        ]
        .concat();
        let expected_d = [
            [-1.794902, -1.0011765, 0.0, 0.49607843],
            [-0.8000001, 1.2, 0.2509804, 0.5019608],
            [-8.0, 12.0, 2.509804, 5.019608],
            [0.20000002, 0.3, 0.40000004, 0.5],
        ]
        .concat();

        // Affine quantization for each block with range min/max
        let per_block_strategy = vec![
            AffineQuantization::<f32, i8, i32>::new(-1.8, 0.5),
            AffineQuantization::<f32, i8, i32>::new(-0.8, 1.2),
            AffineQuantization::<f32, i8, i32>::new(-8., 12.),
            AffineQuantization::<f32, i8, i32>::new(0.2, 0.5),
        ];
        let strategy =
            QuantizationStrategy::PerBlockAffineInt8(per_block_strategy, BlockLayout::Flat(4));

        let q: Vec<i8> = strategy.quantize(&x, shape);
        assert_eq!(q, expected_q);

        let d = strategy.dequantize(&expected_q, shape);

        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_int8_symmetric_quantization_per_block_grid() {
        let x: [f32; 8] = [-1.8, -1.0, 0.0, 0.5, 0.5, 0.0, -1.0, -1.8];
        let shape = &[2, 4];
        let expected_q = vec![-127, -71, 0, 35, 35, 0, -71, -127];
        let expected_d = vec![
            -1.8, -1.0062993, 0.0, 0.496063, 0.496063, 0.0, -1.0062993, -1.8,
        ];

        let symmetric = SymmetricQuantization::<f32, i8>::new(-1.8, 0.5);
        let strategy = QuantizationStrategy::PerBlockSymmetricInt8(
            vec![symmetric, symmetric],
            BlockLayout::Grid(2, 2),
        );

        let q: Vec<i8> = strategy.quantize(&x, shape);
        assert_eq!(q, expected_q);

        let d = strategy.dequantize(&expected_q, shape);

        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_int8_symmetric_quantization_per_block_grid_3d() {
        let shape = &[2, 4, 4];
        let x = [
            // 2x2 blocks: [[-1.8, -1.0, 0.0, 0.5], [-0.8, 1.2, 0.25, 0.5]]
            [-1.8, -1.0, -0.8, 1.2],
            [0.0, 0.5, 0.25, 0.5],
            // 2x2 blocks: [[-0.08, 0.12, 0.025, 0.05], [0.2, 0.3, 0.4, 0.5]]
            [-0.08, 0.12, 0.2, 0.3],
            [0.025, 0.05, 0.4, 0.5],
            // 2x2 blocks: [[0.01, 0.03, 0.02, 0.06], [4.0, 3.0, 2.0, 1.0]]
            [0.01, 0.03, 4.0, 3.0],
            [0.02, 0.06, 2.0, 1.0],
            // 2x2 blocks: [[0.4, 0.3, 0.2, 0.1], [0.5, 0.0, -1.0, -1.8]]
            [0.4, 0.3, 0.5, 0.0],
            [0.2, 0.1, -1.0, -1.8],
        ]
        .concat(); // easier to visualize with a vec of rows
        let expected_q = [
            [-127, -71, -85, 127],
            [0, 35, 26, 53],
            [-85, 127, 51, 76],
            [26, 53, 102, 127],
            [21, 64, 127, 95],
            [42, 127, 64, 32],
            [127, 95, 35, 0],
            [64, 32, -71, -127],
        ]
        .concat();
        let expected_d = [
            [-1.8, -1.0062993, -0.8031496, 1.2],
            [0.0, 0.496063, 0.24566929, 0.5007874],
            [-0.08031496, 0.12, 0.2007874, 0.2992126],
            [0.024566928, 0.05007874, 0.4015748, 0.5],
            [0.009921259, 0.03023622, 4.0, 2.992126],
            [0.019842518, 0.06, 2.015748, 1.007874],
            [0.4, 0.2992126, 0.496063, 0.0],
            [0.2015748, 0.1007874, -1.0062993, -1.8],
        ]
        .concat();

        // Symmetric quantization for each block with range min/max
        let per_block_strategy = vec![
            SymmetricQuantization::<f32, i8>::new(-1.8, 0.5),
            SymmetricQuantization::<f32, i8>::new(-0.8, 1.2),
            SymmetricQuantization::<f32, i8>::new(-0.08, 0.12),
            SymmetricQuantization::<f32, i8>::new(0.2, 0.5),
            SymmetricQuantization::<f32, i8>::new(0.01, 0.06),
            SymmetricQuantization::<f32, i8>::new(1.0, 4.0),
            SymmetricQuantization::<f32, i8>::new(0.1, 0.4),
            SymmetricQuantization::<f32, i8>::new(-1.8, 0.5),
        ];
        let strategy = QuantizationStrategy::PerBlockSymmetricInt8(
            per_block_strategy,
            BlockLayout::Grid(2, 2),
        );

        let q: Vec<i8> = strategy.quantize(&x, shape);
        assert_eq!(q, expected_q);

        let d = strategy.dequantize(&expected_q, shape);

        assert_eq!(d, expected_d);
    }
}
