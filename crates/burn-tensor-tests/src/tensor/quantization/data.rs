use super::*;
use alloc::vec;
use burn_tensor::quantization::{QTensorPrimitive, QuantLevel, QuantValue};
use burn_tensor::{TensorData, ops::QuantizedTensor};

#[test]
fn should_support_per_tensor_symmetric_int8() {
    let data = TensorData::quantized(
        vec![-127i8, -71, 0, 35],
        [4],
        QuantizedTensor::<TestBackend>::default_scheme().with_value(QuantValue::Q8S),
        &[0.014_173_228],
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
        QuantizedTensor::<TestBackend>::default_scheme()
            .with_value(QuantValue::Q8S)
            .with_level(QuantLevel::block([8])),
        &[0.014_173_228, 0.000_314_96],
    );
    let tensor = TestTensor::<1>::from_data(data.clone(), &Default::default());

    tensor.into_data().assert_eq(&data, true);
}
