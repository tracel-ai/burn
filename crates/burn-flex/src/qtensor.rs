use alloc::vec::Vec;

use burn_backend::{DType, QTensorPrimitive, TensorMetadata, quantization::QuantStore};
use burn_std::{QuantScheme, Shape};

use crate::tensor::FlexTensor;

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
}

impl FlexQTensor {
    /// Create a new quantized tensor.
    ///
    /// The tensor must store i8 data and scales must be non-empty.
    pub fn new(tensor: FlexTensor, scheme: QuantScheme, scales: Vec<f32>) -> Self {
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
        Self {
            tensor,
            scheme,
            scales,
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

impl QTensorPrimitive for FlexQTensor {
    fn scheme(&self) -> &QuantScheme {
        &self.scheme
    }

    fn default_scheme() -> QuantScheme {
        QuantScheme::default().with_store(QuantStore::Native)
    }
}

impl TensorMetadata for FlexQTensor {
    fn dtype(&self) -> DType {
        DType::QFloat(self.scheme)
    }

    fn shape(&self) -> Shape {
        self.tensor.shape()
    }

    fn rank(&self) -> usize {
        self.tensor.rank()
    }
}
