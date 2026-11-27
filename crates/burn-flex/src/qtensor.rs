use alloc::vec::Vec;

use burn_backend::{DType, TensorMetadata};
use burn_std::{QuantScheme, Shape};

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
    /// The coarser level's scales that [`scales`](Self::scales) are expressed relative to, for a
    /// two-level scheme: one per block of the coarser level, row-major over its grid.
    pub(crate) coarser: Option<Vec<f32>>,
}

impl FlexQTensor {
    /// Create a new quantized tensor.
    ///
    /// The tensor must store i8 data and scales must be non-empty. A two-level scheme takes
    /// coarser scales, and no other level does.
    pub fn new(
        tensor: FlexTensor,
        scheme: QuantScheme,
        scales: Vec<f32>,
        coarser: Option<Vec<f32>>,
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
        assert_eq!(
            scheme.levels().nth(1).is_some(),
            coarser.is_some(),
            "{scheme:?} does not match coarser scales of {coarser:?}"
        );
        Self {
            tensor,
            scheme,
            scales,
            coarser,
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
