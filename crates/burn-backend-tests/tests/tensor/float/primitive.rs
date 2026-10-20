use super::*;
use burn_tensor::{DType, Device, Element, Shape, TensorData, flex32};

#[test]
fn should_support_float_dtype() {
    let tensor = TestTensor::<2>::from([[0.0, -1.0, 2.0], [3.0, 4.0, -5.0]])/*.into_primitive()*/;

    assert_eq!(tensor.shape(), Shape::new([2, 3]));
    assert_eq!(
        tensor.dtype(),
        FloatElem::dtype() // default float elem type
    );
}

#[test]
fn explicit_dtype_is_preserved_after_default_dtype_is_locked() {
    let device = Device::default();
    // Tensor creation locks device settings to the configured dtype
    let _default = TestTensor::<1>::zeros([1], &device);
    let settings = device.settings();
    assert_eq!(DType::from(settings.float_dtype), FloatElem::dtype());

    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        if !device.supports_dtype(dtype) {
            continue;
        }

        let explicit = TestTensor::<1>::from_data(TensorData::from([1.0f64]), (&device, dtype));

        assert_eq!(explicit.dtype(), dtype);
    }
}

#[test]
fn should_support_into_data_from_data() {
    let device = Default::default();
    let data =
        TestTensor::<2>::from_data([[0.0, -1.0, 2.0], [3.0, 4.0, -5.0]], &device).into_data();
    let tensor = TestTensor::<2>::from_data(data, &device).slice(0);

    // Regression test for `LazyDeviceController` from_data(tensor.into_data()) roundtrips
    // These unnecessary round-trips should be avoided, but should not panic
    tensor
        .into_data()
        .assert_eq(&TensorData::from([[0.0, -1.0, 2.0]]), false);
}

#[test]
fn flex32_from_data_should_round_trip_and_compute() {
    let device = Device::default();
    if !device.supports_dtype(DType::Flex32) {
        return;
    }

    let input = TensorData::from([1.0, 2.0, 3.0].map(flex32::from_f32));
    let lhs = TestTensor::<1>::from_data(input.clone(), (&device, DType::Flex32));

    assert_eq!(lhs.dtype(), DType::Flex32);
    let readback = lhs.to_data();
    assert_eq!(readback, input);

    let rhs = TestTensor::<1>::from_data(readback, (&device, DType::Flex32));
    let output = lhs + rhs;

    assert_eq!(output.dtype(), DType::Flex32);
    assert_eq!(
        output.into_data(),
        TensorData::from([2.0, 4.0, 6.0].map(flex32::from_f32))
    );
}
