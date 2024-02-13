use core::ops::Deref;
use std::collections::HashMap;
use std::path::Path;

use super::{adapter::PyTorchAdapter, error::Error};

use burn::{
    module::ParamId,
    record::{ParamSerde, PrecisionSettings},
    tensor::{DataSerialize, Element, ElementConversion},
};
use burn::{
    record::serde::{
        data::{remap, unflatten, NestedValue, Serializable},
        de::Deserializer,
        error,
        ser::Serializer,
    },
    tensor::backend::Backend,
};

use candle_core::{pickle, WithDType};
use half::{bf16, f16};
use regex::Regex;
use serde::{de::DeserializeOwned, Serialize};

/// Deserializes a PyTorch file.
///
/// # Arguments
///
/// * `path` - A string slice that holds the path of the file to read.
/// * `key_remap` - A vector of tuples containing a regular expression and a replacement string.
/// * `top_level_key` - An optional top-level key to load state_dict from a dictionary.
pub fn from_file<PS, D, B>(
    path: &Path,
    key_remap: Vec<(Regex, String)>,
    top_level_key: Option<&str>,
) -> Result<D, Error>
where
    D: DeserializeOwned,
    PS: PrecisionSettings,
    B: Backend,
{
    // Read the pickle file and return a vector of Candle tensors
    let tensors: HashMap<String, CandleTensor> = pickle::read_all_with_key(path, top_level_key)?
        .into_iter()
        .map(|(key, tensor)| (key, CandleTensor(tensor)))
        .collect();

    // Remap the keys (replace the keys in the map with the new keys)
    let tensors = remap(tensors, key_remap);

    // Convert the vector of Candle tensors to a nested value data structure
    let nested_value = unflatten::<PS, _>(tensors)?;

    // Create a deserializer with PyTorch adapter and nested value
    let deserializer = Deserializer::<PyTorchAdapter<PS, B>>::new(nested_value, true);

    // Deserialize the nested value into a record type
    let value = D::deserialize(deserializer)?;
    Ok(value)
}

/// Serializes a candle tensor.
///
/// Tensors are wrapped in a `Param` struct (learnable parameters) and serialized as a `DataSerialize` struct.
///
/// Values are serialized as `FloatElem` or `IntElem` depending on the precision settings.
impl Serializable for CandleTensor {
    fn serialize<PS>(&self, serializer: Serializer) -> Result<NestedValue, error::Error>
    where
        PS: PrecisionSettings,
    {
        let shape = self.shape().clone().into_dims();
        let flatten = CandleTensor(self.flatten_all().expect("Failed to flatten the tensor"));
        let param_id = ParamId::new().into_string();

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
    param_id: String,
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

    ParamSerde::new(param_id, DataSerialize::new(data, shape)).serialize(serializer)
}

/// New type struct for Candle tensors because we need to implement the `Serializable` trait for it.
struct CandleTensor(candle_core::Tensor);

impl Deref for CandleTensor {
    type Target = candle_core::Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
