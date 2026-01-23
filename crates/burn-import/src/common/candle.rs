use core::ops::Deref;
use std::collections::HashMap;

use burn::record::serde::{
    data::{NestedValue, Serializable},
    error,
    ser::Serializer,
};
use burn::{
    module::ParamId,
    record::PrecisionSettings,
    tensor::{Element, ElementConversion, TensorData, bf16, f16},
};

use candle_core::WithDType;
use serde::Serialize;

use burn::record::RecorderError;
use zip::result::ZipError;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Serde error: {0}")]
    Serde(#[from] error::Error),

    #[error("Candle Tensor error: {0}")]
    CandleTensor(#[from] candle_core::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Zip error: {0}")]
    Zip(#[from] ZipError),

    // Add other kinds of errors as needed
    #[error("other error: {0}")]
    Other(String),
}

// Implement From trait for Error to RecorderError
impl From<Error> for RecorderError {
    fn from(error: Error) -> Self {
        RecorderError::DeserializeError(error.to_string())
    }
}

/// Serializes a candle tensor.
///
/// Tensors are wrapped in a `Param` struct (learnable parameters) and serialized as a `TensorData` struct.
///
/// Values are serialized as `FloatElem` or `IntElem` depending on the precision settings.
impl Serializable for CandleTensor {
    fn serialize<PS>(&self, serializer: Serializer) -> Result<NestedValue, error::Error>
    where
        PS: PrecisionSettings,
    {
        let shape = self.shape().clone().into_dims();
        let flatten = CandleTensor(self.flatten_all().expect("Failed to flatten the tensor"));
        let param_id = ParamId::new();

        match self.dtype() {
            candle_core::DType::U8 => {
                serialize_data::<u8, PS::IntElem>(flatten, shape, param_id, serializer)
            }
            candle_core::DType::U32 => {
                serialize_data::<u32, PS::IntElem>(flatten, shape, param_id, serializer)
            }
            candle_core::DType::I64 => {
                serialize_data::<i64, PS::IntElem>(flatten, shape, param_id, serializer)
            }
            candle_core::DType::BF16 => {
                serialize_data::<bf16, PS::FloatElem>(flatten, shape, param_id, serializer)
            }
            candle_core::DType::F16 => {
                serialize_data::<f16, PS::FloatElem>(flatten, shape, param_id, serializer)
            }
            candle_core::DType::F32 => {
                serialize_data::<f32, PS::FloatElem>(flatten, shape, param_id, serializer)
            }
            candle_core::DType::F64 => {
                serialize_data::<f64, PS::FloatElem>(flatten, shape, param_id, serializer)
            }
        }
    }
}

/// Helper function to serialize a candle tensor data.
fn serialize_data<T, E>(
    tensor: CandleTensor,
    shape: Vec<usize>,
    param_id: ParamId,
    serializer: Serializer,
) -> Result<NestedValue, error::Error>
where
    E: Element + Serialize,
    T: WithDType + ElementConversion,
{
    let data: Vec<E> = tensor
        .to_vec1::<T>()
        .map_err(|err| error::Error::Other(format!("Candle to vec1 error: {err}")))?
        .into_iter()
        .map(ElementConversion::elem)
        .collect();

    let data = TensorData::new(data, shape.clone());
    let (dtype, bytes) = (data.dtype, data.into_bytes());

    // Manually serialize the tensor instead of using the `ParamSerde` struct, such as:
    // ParamSerde::new(param_id, TensorData::new(data, shape)).serialize(serializer)
    // Because serializer copies individual elements of TensorData `value` into a new Vec<u8>,
    // which is not necessary and inefficient.
    let mut tensor_data: HashMap<String, NestedValue> = HashMap::new();
    tensor_data.insert("bytes".into(), NestedValue::Bytes(bytes));
    tensor_data.insert("shape".into(), shape.serialize(serializer.clone())?);
    tensor_data.insert("dtype".into(), dtype.serialize(serializer)?);

    let mut param: HashMap<String, NestedValue> = HashMap::new();
    param.insert("id".into(), NestedValue::String(param_id.serialize()));
    param.insert("param".into(), NestedValue::Map(tensor_data));

    Ok(NestedValue::Map(param))
}

/// New type struct for Candle tensors because we need to implement the `Serializable` trait for it.
pub struct CandleTensor(pub candle_core::Tensor);

impl Deref for CandleTensor {
    type Target = candle_core::Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub fn print_debug_info(
    tensors: &HashMap<String, CandleTensor>,
    remapped_keys: Vec<(String, String)>,
) {
    let mut remapped_keys = remapped_keys;
    remapped_keys.sort();
    println!("Debug information of keys and tensor shapes:\n---");
    for (new_key, old_key) in remapped_keys {
        if old_key != new_key {
            println!("Original Key: {old_key}");
            println!("Remapped Key: {new_key}");
        } else {
            println!("Key: {new_key}");
        }

        let shape = tensors[&new_key].shape();
        let dtype = tensors[&new_key].dtype();
        println!("Shape: {shape:?}");
        println!("Dtype: {dtype:?}");
        println!("---");
    }
}
