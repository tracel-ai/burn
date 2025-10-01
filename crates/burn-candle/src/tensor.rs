use burn_tensor::{
    DType, Element, FloatDType, IntDType, Shape, TensorData, TensorMetadata,
    quantization::{QTensorPrimitive, QuantScheme},
};

use crate::{CandleDevice, element::CandleElement};

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

    fn rank(&self) -> usize {
        self.tensor.dims().len()
    }
}

impl QTensorPrimitive for CandleTensor {
    fn scheme(&self) -> &QuantScheme {
        unimplemented!("Quantization is not supported")
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

pub(crate) trait IntoDType {
    fn into_dtype(self) -> candle_core::DType;
}

impl IntoDType for IntDType {
    fn into_dtype(self) -> candle_core::DType {
        match self {
            IntDType::I64 => candle_core::DType::I64,
            IntDType::U32 => candle_core::DType::U32,
            IntDType::U8 => candle_core::DType::U8,
            other => panic!("Unsupported dtype {other:?}"),
        }
    }
}

impl IntoDType for FloatDType {
    fn into_dtype(self) -> candle_core::DType {
        match self {
            FloatDType::F64 => candle_core::DType::F64,
            FloatDType::F32 => candle_core::DType::F32,
            FloatDType::Flex32 => candle_core::DType::F32,
            FloatDType::F16 => candle_core::DType::F16,
            FloatDType::BF16 => candle_core::DType::BF16,
        }
    }
}
