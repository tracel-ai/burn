use std::collections::HashMap;
use std::path::Path;

use super::adapter::PyTorchAdapter;
use crate::record::{
    data::{remap, unflatten, Serializable},
    de::Deserializer,
    error::Error,
};

use burn::{
    module::ParamId,
    record::PrecisionSettings,
    tensor::{DataSerialize, Element, ElementConversion},
};

use candle_core::{pickle, Tensor as CandleTensor, WithDType};
use half::{bf16, f16};
use regex::Regex;
use serde::{de::DeserializeOwned, Serialize};

/// Deserializes a PyTorch file.
///
/// # Arguments
///
/// * `path` - A string slice that holds the path of the file to read.
/// * `key_remap` - A vector of tuples containing a regular expression and a replacement string.
pub fn from_file<PS, D>(path: &Path, key_remap: Vec<(Regex, String)>) -> Result<D, Error>
where
    D: DeserializeOwned,
    PS: PrecisionSettings,
{
    // Read the pickle file and return a vector of Candle tensors
    let tensors: HashMap<String, CandleTensor> =
        pickle::read_all(path).unwrap().into_iter().collect();

    // Remap the keys (replace the keys in the map with the new keys)
    let tensors = remap(tensors, key_remap);

    // Convert the vector of Candle tensors to a nested value data structure
    let nested_value = unflatten::<PS, _>(tensors);

    // Create a deserializer with PyTorch adapter and nested value
    let deserializer = Deserializer::<PyTorchAdapter<PS>>::new(nested_value);

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
    fn serialize<PS, S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
        PS: PrecisionSettings,
    {
        let shape = self.shape().clone().into_dims();
        let flatten = self.flatten_all().unwrap();
        let param_id = ParamId::new().into_string();

        match self.dtype() {
            candle_core::DType::U8 => {
                serialize_data::<u8, PS::IntElem, _>(flatten, shape, param_id, serializer)
            }
            candle_core::DType::U32 => {
                serialize_data::<u32, PS::IntElem, _>(flatten, shape, param_id, serializer)
            }
            candle_core::DType::I64 => {
                serialize_data::<i64, PS::IntElem, _>(flatten, shape, param_id, serializer)
            }
            candle_core::DType::BF16 => {
                serialize_data::<bf16, PS::FloatElem, _>(flatten, shape, param_id, serializer)
            }
            candle_core::DType::F16 => {
                serialize_data::<f16, PS::FloatElem, _>(flatten, shape, param_id, serializer)
            }
            candle_core::DType::F32 => {
                serialize_data::<f32, PS::FloatElem, _>(flatten, shape, param_id, serializer)
            }
            candle_core::DType::F64 => {
                serialize_data::<f64, PS::FloatElem, _>(flatten, shape, param_id, serializer)
            }
        }
    }
}

/// Redefine a Param struct so it can be serialized.
///
/// Note: This is a workaround for the fact that `Param` is not serializable.
#[derive(new, Debug, Clone, Serialize)]
struct Param<T> {
    id: String,
    param: T,
}

/// Helper function to serialize a candle tensor data.
fn serialize_data<T, E, S>(
    tensor: CandleTensor,
    shape: Vec<usize>,
    param_id: String,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    E: Element + Serialize,
    S: serde::Serializer,
    T: WithDType + ElementConversion,
{
    let data: Vec<E> = tensor
        .to_vec1::<T>()
        .unwrap()
        .into_iter()
        .map(ElementConversion::elem)
        .collect();

    Param::new(param_id, DataSerialize::new(data, shape)).serialize(serializer)
}
