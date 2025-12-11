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
    fn try_into_dtype(self) -> Result<candle_core::DType, candle_core::Error>;

    fn into_dtype(self) -> candle_core::DType
    where
        Self: Sized,
    {
        self.try_into_dtype().unwrap()
    }
}

impl IntoDType for IntDType {
    fn try_into_dtype(self) -> Result<candle_core::DType, candle_core::Error> {
        let dtype: DType = self.into();
        dtype.try_into_dtype()
    }
}

impl IntoDType for FloatDType {
    fn try_into_dtype(self) -> Result<candle_core::DType, candle_core::Error> {
        let dtype: DType = self.into();
        dtype.try_into_dtype()
    }
}

impl IntoDType for DType {
    fn try_into_dtype(self) -> Result<candle_core::DType, candle_core::Error> {
        match self {
            DType::F64 => Ok(candle_core::DType::F64),
            DType::F32 => Ok(candle_core::DType::F32),
            DType::Flex32 => Ok(candle_core::DType::F32),
            DType::F16 => Ok(candle_core::DType::F16),
            DType::BF16 => Ok(candle_core::DType::BF16),
            DType::I64 => Ok(candle_core::DType::I64),
            DType::U32 => Ok(candle_core::DType::U32),
            DType::U8 => Ok(candle_core::DType::U8),
            // DType::Bool => Ok(candle_core::DType::U8),
            _ => Err(candle_core::Error::Msg(format!(
                "Unsupported dtype {self:?}"
            ))),
        }
    }
}
