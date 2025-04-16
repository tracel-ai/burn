#[burn_tensor_testgen::testgen(q_data)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};
    use burn_tensor::quantization::{
        QuantizationStrategy, SymmetricQuantization,
    };
    use burn_tensor::{Tensor, TensorData};

    // NOTE: we mark the per-block tests as `might_panic` since backends are not strictly
    // required to support this quantization scheme.
    // Also std feature gated (until `catch_unwind` is stable in core).
    #[cfg(feature = "std")]
    use burn_tensor::might_panic;

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
}
