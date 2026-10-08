use super::*;
use burn_tensor::{
    DType, Tolerance,
    quantization::{QuantStore, QuantValue, ScaleDtype},
};

#[test]
fn bytes_packed_along_an_outer_axis_load_on_flex_unchanged() {
    let device = burn_tensor::Device::default();
    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q4S)
        .with_store(QuantStore::PackedU32(0))
        .per_block([16], ScaleDtype::F16)
        .per_tensor(ScaleDtype::F32);
    if !device.supports_dtype(DType::QFloat(scheme)) {
        return;
    }
    let input: TestTensor<2> = TestTensorInt::arange(0..512, &device)
        .float()
        .div_scalar(512.)
        .sub_scalar(0.5)
        .reshape([32, 16]);
    let quantized = input
        .swap_dims(0, 1)
        .quantize_dynamic(&scheme)
        .swap_dims(0, 1);
    let written = quantized.clone().into_data();

    let on_flex = TestTensor::<2>::from_data(written.clone(), &ReferenceDevice::new());

    assert_eq!(on_flex.dtype(), written.dtype());
    assert_eq!(on_flex.clone().into_data().as_bytes(), written.as_bytes());
    on_flex
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&quantized.dequantize().into_data(), Tolerance::default());
}
