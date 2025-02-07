use burn_tensor::{
    quantization::{QTensorPrimitive, QuantizationScheme, QuantizationStrategy},
    DType, Element, Shape, TensorData, TensorMetadata,
};

use crate::{element::CandleElement, CandleDevice};

/// A tensor that uses the candle backend.
#[derive(Debug, Clone)]
pub struct CandleTensor {
    pub(crate) tensor: candle_core::Tensor,
}

impl TensorMetadata for CandleTensor {
    fn dtype(&self) -> DType {
        match self.tensor.dtype() {
            candle_core::DType::U8 => DType::U8,
            candle_core::DType::U32 => DType::U32,
            candle_core::DType::I64 => DType::I64,
            candle_core::DType::BF16 => DType::BF16,
            candle_core::DType::F16 => DType::F16,
            candle_core::DType::F32 => DType::F32,
            candle_core::DType::F64 => DType::F64,
        }
    }

    fn shape(&self) -> Shape {
        Shape::from(self.tensor.dims().to_vec())
    }
}

impl CandleTensor {
    /// Create a new tensor.
    pub fn new(tensor: candle_core::Tensor) -> Self {
        Self { tensor }
    }

    /// Creates a new tensor from data and a device.
    ///
    /// # Arguments
    ///
    /// * `data` - The tensor's data.
    /// * `device` - The device on which the tensor will be allocated.
    ///
    /// # Returns
    ///
    /// A new tensor.
    pub fn from_data<E: CandleElement>(data: TensorData, device: CandleDevice) -> Self {
        let candle_shape: candle_core::Shape = data.shape.clone().into();
        let tensor = candle_core::Tensor::from_slice(
            data.as_slice::<E>().unwrap(),
            candle_shape,
            &device.into(),
        );
        Self::new(tensor.unwrap())
    }
}

/// A quantized tensor for the candle backend.
#[derive(Clone, Debug)]
pub struct CandleQTensor {
    /// The quantized tensor.
    // NOTE: candle  does not implement `WithDType` for i8
    pub qtensor: CandleTensor,
    /// The quantization scheme.
    pub scheme: QuantizationScheme,
}

impl QTensorPrimitive for CandleQTensor {
    fn scheme(&self) -> &QuantizationScheme {
        &self.scheme
    }
}

impl TensorMetadata for CandleQTensor {
    fn dtype(&self) -> DType {
        DType::QFloat(self.scheme)
    }

    fn shape(&self) -> Shape {
        self.qtensor.shape()
    }
}
