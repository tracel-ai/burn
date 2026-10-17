use burn_tensor::{
    Device, TensorData,
    quantization::{QuantScheme, QuantStore, QuantValue, ScaleDtype},
};

/// The scheme `device` quantizes with by default, with 8-bit symmetric values.
pub fn int8_scheme(device: &Device) -> QuantScheme {
    device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
}

/// Every layout a quantized tensor crosses the wire in: one scale, block scales, values packed
/// eight to a word, and block scales under a per-tensor scale.
pub fn wire_schemes(device: &Device) -> [QuantScheme; 4] {
    let q8 = int8_scheme(device);
    [
        q8,
        q8.per_block([32], ScaleDtype::F32),
        q8.with_value(QuantValue::Q4S)
            .with_store(QuantStore::PackedU32(0))
            .per_block([32], ScaleDtype::F32),
        q8.per_block([2, 16], ScaleDtype::UE4M3)
            .per_tensor(ScaleDtype::F32),
    ]
}

/// Values whose magnitude changes along the tensor, so every block gets a scale of its own.
pub fn wire_floats() -> TensorData {
    let values: Vec<f32> = (0..64 * 64)
        .map(|i| ((i * 37 % 211) as f32 - 105.0) * (1 + i / 512) as f32 * 0.01)
        .collect();
    TensorData::new(values, [64, 64])
}

/// Values, scales and scheme alike: the bytes a quantized tensor is sent as.
#[track_caller]
pub fn assert_same_bytes(actual: &TensorData, expected: &TensorData) {
    assert_eq!(actual.dtype(), expected.dtype());
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(actual.as_bytes(), expected.as_bytes());
}
