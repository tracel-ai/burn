/// Quantization data type.
pub enum QuantizationType {
    /// 8-bit signed integer.
    QInt8,
}

/// Quantization scheme.
pub enum QuantizationScheme {
    /// Per-tensor affine/asymmetric quantization.
    PerTensorAffine(QuantizationType),
    /// Per-tensor symmetric quantization.
    PerTensorSymmetric(QuantizationType),
    // /// Per-channel affine/asymmetric quantization.
    // PerChannelAffine,
    // /// Per-channel symmetric quantization.
    // PerChannelSymmetric,
}
