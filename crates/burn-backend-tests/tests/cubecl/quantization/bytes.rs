use super::*;
use burn_tensor::{
    DType, Device, Tolerance,
    quantization::{QuantScheme, QuantStore, QuantValue, ScaleDtype},
};

fn input(device: &Device) -> TestTensor<2> {
    TestTensorInt::arange(0..512, device)
        .float()
        .div_scalar(512.)
        .sub_scalar(0.5)
        .reshape([32, 16])
}

fn packed(device: &Device) -> QuantScheme {
    device
        .settings()
        .quantization
        .scheme
        .with_store(QuantStore::PackedU32(0))
}

#[test]
fn bytes_packed_along_an_outer_axis_load_on_flex_unchanged() {
    let device = Device::default();
    let packed = packed(&device);
    for scheme in [
        packed
            .with_value(QuantValue::Q4S)
            .per_block([16], ScaleDtype::F16)
            .per_tensor(ScaleDtype::F32),
        packed
            .with_value(QuantValue::E4M3)
            .per_block([16], ScaleDtype::F32),
        packed
            .with_value(QuantValue::E2M1)
            .per_block([16], ScaleDtype::UE8M0),
    ] {
        if !device.supports_dtype(DType::QFloat(scheme)) {
            continue;
        }
        let quantized = input(&device)
            .swap_dims(0, 1)
            .quantize_dynamic(&scheme)
            .swap_dims(0, 1);
        let written = quantized.clone().into_data();

        let on_flex = TestTensor::<2>::from_data(written.clone(), &ReferenceDevice::new());

        assert_eq!(on_flex.dtype(), written.dtype(), "{scheme:?}");
        assert_eq!(
            on_flex.clone().into_data().as_bytes(),
            written.as_bytes(),
            "{scheme:?}"
        );
        on_flex
            .dequantize()
            .into_data()
            .assert_approx_eq::<FloatElem>(
                &quantized.dequantize().into_data(),
                Tolerance::default(),
            );
    }
}

// Not compared with the bytes the device writes: a device's division need not round correctly, so
// its scales can differ from Flex's in the last bit.
#[test]
fn bytes_flex_writes_read_on_a_device_as_on_flex() {
    let device = Device::default();
    let reference = ReferenceDevice::new();
    let packed = packed(&device);
    for scheme in [
        packed.with_value(QuantValue::Q8S),
        packed
            .with_value(QuantValue::Q4S)
            .per_block([16], ScaleDtype::F32),
        packed.with_value(QuantValue::E4M3),
        packed.with_value(QuantValue::E5M2),
        packed
            .with_value(QuantValue::E2M1)
            .per_block([16], ScaleDtype::UE8M0),
    ] {
        if !device.supports_dtype(DType::QFloat(scheme)) {
            continue;
        }
        let written = TestTensor::<2>::from_data(input(&device).into_data(), &reference)
            .quantize_dynamic(&scheme)
            .into_data();

        let on_device = TestTensor::<2>::from_data(written.clone(), &device);
        let on_flex = TestTensor::<2>::from_data(written, &reference);

        on_device
            .dequantize()
            .into_data()
            .assert_approx_eq::<FloatElem>(&on_flex.dequantize().into_data(), Tolerance::default());
    }
}
