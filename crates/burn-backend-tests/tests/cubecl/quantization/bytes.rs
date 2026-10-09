use super::*;
use burn_tensor::{
    DType, Device, TensorData, Tolerance,
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

/// Every row reaches `value`'s largest magnitude times a power of two, so each scale is a power of
/// two that any device's division yields exactly; the other values are ninths, never a rounding tie.
fn exact_scale_input(value: QuantValue) -> TensorData {
    let (_, max) = value.range();
    let values: Vec<f32> = (0..32)
        .flat_map(|row| {
            let magnitude = max / (1 << (row % 4)) as f32;
            (0..16).map(move |col| match col {
                0 => -magnitude,
                col => magnitude * (col as f32 - 8.0) / 9.0,
            })
        })
        .collect();
    TensorData::new(values, [32, 16])
}

#[test]
fn flex_quantizes_to_the_bytes_a_device_writes() {
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
        let input = exact_scale_input(scheme.value);

        let on_device = TestTensor::<2>::from_data(input.clone(), &device)
            .quantize_dynamic(&scheme)
            .into_data();
        let on_flex = TestTensor::<2>::from_data(input, &reference)
            .quantize_dynamic(&scheme)
            .into_data();

        assert_eq!(on_flex.as_bytes(), on_device.as_bytes(), "{scheme:?}");
    }
}
