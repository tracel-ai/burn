use serde::{Deserialize, Serialize};

use crate::quantization::QuantizationScheme;

use super::TensorDescription;

/// A quantized tensor description represents a snapshot of a quantized tensor when it was used.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizedTensorDescription {
    /// The quantized tensor.
    pub tensor: TensorDescription,
    /// The quantization parameters.
    pub qparams: QuantizationParametersDescription,
    /// The quantization scheme
    pub scheme: QuantizationScheme,
}

/// Quantization parameters description.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationParametersDescription {
    /// The scaling factor.
    pub scale: TensorDescription,
    /// The zero-point offset.
    pub offset: Option<TensorDescription>,
}
