use alloc::vec::Vec;

use burn_backend::{DType, TensorMetadata};
use burn_std::{QuantScheme, Shape, quantization::validate_levels};

use crate::{FlexDevice, tensor::FlexTensor};

/// Quantized tensor for the Flex backend.
///
/// Stores quantized i8 values in the tensor and keeps scales separately
/// for efficient dequantization without reparsing bytes.
#[derive(Clone, Debug)]
pub struct FlexQTensor {
    /// The underlying quantized data (stored as i8).
    pub(crate) tensor: FlexTensor,
    /// Quantization scheme.
    pub(crate) scheme: QuantScheme,
    /// Per-tensor or per-block scale factors.
    pub(crate) scales: Vec<f32>,
    /// The per-tensor scale that [`scales`](Self::scales) are expressed relative to, for a
    /// two-level scheme.
    pub(crate) global: Option<f32>,
}

impl FlexQTensor {
    /// Create a new quantized tensor.
    ///
    /// The tensor must store i8 data and scales must be non-empty. A two-level scheme takes a
    /// per-tensor scale, and no other level does.
    pub fn new(
        tensor: FlexTensor,
        scheme: QuantScheme,
        scales: Vec<f32>,
        global: Option<f32>,
    ) -> Self {
        assert_eq!(
            tensor.dtype(),
            DType::I8,
            "quantized tensor must store i8 data, got {:?}",
            tensor.dtype()
        );
        assert!(
            !scales.is_empty(),
            "quantized tensor must have at least one scale factor"
        );
        validate_levels(&scheme);
        assert_eq!(
            scheme.level.global_param().is_some(),
            global.is_some(),
            "{:?} does not match a per-tensor scale of {global:?}",
            scheme.level
        );
        Self {
            tensor,
            scheme,
            scales,
            global,
        }
    }

    /// Get the underlying tensor.
    pub fn tensor(&self) -> &FlexTensor {
        &self.tensor
    }

    /// Get the quantization scales.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }
}

impl TensorMetadata for FlexQTensor {
    type Device = FlexDevice;

    fn dtype(&self) -> DType {
        DType::QFloat(self.scheme)
    }

    fn shape(&self) -> Shape {
        self.tensor.shape()
    }

    fn rank(&self) -> usize {
        self.tensor.rank()
    }

    fn device(&self) -> Self::Device {
        FlexDevice
    }

    fn can_mut(&self) -> bool {
        self.tensor.is_unique()
    }
}
