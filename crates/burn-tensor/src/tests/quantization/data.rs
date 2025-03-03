#[burn_tensor_testgen::testgen(q_data)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};
    use burn_tensor::quantization::{
        AffineQuantization, BlockLayout, QuantizationStrategy, SymmetricQuantization,
    };
    use burn_tensor::{Tensor, TensorData};

    // NOTE: we mark the per-block tests as `might_panic` since backends are not strictly
    // required to support this quantization scheme.
    // Also std feature gated (until `catch_unwind` is stable in core).
    #[cfg(feature = "std")]
    use burn_tensor::might_panic;

    #[test]
    fn should_support_per_tensor_affine_int8() {
        let data = TensorData::quantized(
            vec![-128i8, -39, 72, 127],
            [4],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.009_019_608, 72)),
        );
        let tensor = TestTensor::<1>::from_data(data.clone(), &Default::default());

        tensor.into_data().assert_eq(&data, true);
    }

    #[test]
    fn should_support_per_tensor_symmetric_int8() {
        let data = TensorData::quantized(
            vec![-127i8, -71, 0, 35],
            [4],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(
                0.014_173_228,
            )),
        );
        let tensor = TestTensor::<1>::from_data(data.clone(), &Default::default());

        tensor.into_data().assert_eq(&data, true);
    }

    #[cfg(feature = "std")]
    #[might_panic(reason = "Per-block quantization is not supported")]
    #[test]
    fn should_support_per_block_flat() {
        // Per-block qparams
        let data = TensorData::quantized(
            vec![
                [-127i8, -71, 0, 35, -56, 85, 18, 35],
                [-20, 30, 6, 13, 51, 76, 102, 127],
            ]
            .concat(),
            [2, 8],
            QuantizationStrategy::PerBlockAffineInt8(
                vec![
                    AffineQuantization::init(0.009019608, 71),
                    AffineQuantization::init(0.007843138, -26),
                    AffineQuantization::init(0.00078431366, -25),
                    AffineQuantization::init(0.0019607844, -128),
                ],
                BlockLayout::Flat(4),
            ),
        );
        let tensor = TestTensor::<2>::from_data(data.clone(), &Default::default());

        tensor.into_data().assert_eq(&data, true);
    }

    #[cfg(feature = "std")]
    #[might_panic(reason = "Per-block quantization is not supported")]
    #[test]
    fn should_support_per_block_grid() {
        // Per-block qparams
        let scales: [f32; 8] = [
            0.014173228,
            0.009448819,
            0.0009448819,
            0.003937008,
            0.00047244094,
            0.031496063,
            0.0031496063,
            0.014173228,
        ];
        let data = TensorData::quantized(
            vec![
                [-127i8, -71, -85, 127],
                [0, 35, 26, 53],
                [-85, 127, 51, 76],
                [26, 53, 102, 127],
                [21, 64, 127, 95],
                [42, 127, 64, 32],
                [127, 95, 35, 0],
                [64, 32, -71, -127],
            ]
            .concat(),
            [8, 4],
            QuantizationStrategy::PerBlockSymmetricInt8(
                scales
                    .iter()
                    .map(|&s| SymmetricQuantization::init(s))
                    .collect(),
                BlockLayout::Grid(2, 2),
            ),
        );
        let tensor = TestTensor::<2>::from_data(data.clone(), &Default::default());

        tensor.into_data().assert_eq(&data, true);
    }
}
