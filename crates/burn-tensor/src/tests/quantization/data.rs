#[burn_tensor_testgen::testgen(q_data)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};
    use burn_tensor::quantization::{QuantizationStrategy, SymmetricQuantization};
    use burn_tensor::{Tensor, TensorData};

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

        let q_data = tensor.into_data();
        q_data.assert_eq(&data, true);

        let tensor = TestTensor::<1>::from_data(q_data.clone(), &Default::default());

        tensor.into_data().assert_eq(&q_data, true);
    }

    #[test]
    fn should_support_per_block_symmetric_int8() {
        let data = TensorData::quantized(
            vec![
                -127i8, -71, 0, 35, -127i8, -71, 0, 35, -32, -63, -95, -127, -32, -63, -95, -127,
            ],
            [16],
            QuantizationStrategy::PerBlockSymmetricInt8(
                vec![
                    SymmetricQuantization::init(0.014_173_228),
                    SymmetricQuantization::init(0.000_314_96),
                ],
                8,
            ),
        );
        let tensor = TestTensor::<1>::from_data(data.clone(), &Default::default());

        tensor.into_data().assert_eq(&data, true);
    }
}
