// use std::{marker::PhantomData, path::PathBuf};

// use crate::pytorch::RecordType;

// use super::{reader::NestedValue, Converter};

use std::collections::HashMap;

use burn::{
    module::ParamId,
    record::PrecisionSettings,
    tensor::{DataSerialize, Element, ElementConversion},
};

use candle_core::{Tensor as CandleTensor, WithDType};
use half::{bf16, f16};
use serde::Serialize;

use crate::record::{reader::NestedValue, ser::Serializer};

///  Redefine a Param struct so it can be serialized.
#[derive(new, Debug, Clone, Serialize)]
pub struct Param<T> {
    pub id: String,
    pub param: T,
}

/// Serializes a candle tensor.
///
/// Tensors are wrapped in a `Param` struct (learnable parameters) and serialized as a `DataSerialize` struct.
///
/// Values are serialized as `FloatElem` or `IntElem` depending on the precision settings.
pub fn serialize_tensor<S, PS>(tensor: &CandleTensor, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
    PS: PrecisionSettings,
{
    let shape = tensor.shape().clone().into_dims();
    let flatten = tensor.flatten_all().unwrap();
    let param_id = ParamId::new().into_string();

    match tensor.dtype() {
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

/// Serializes a candle tensor data.
fn serialize_data<T, E, S>(
    flatten: CandleTensor,
    shape: Vec<usize>,
    param_id: String,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    E: Element + Serialize,
    S: serde::Serializer,
    T: WithDType + ElementConversion,
{
    let data: Vec<E> = flatten
        .to_vec1::<T>()
        .unwrap()
        .into_iter()
        .map(ElementConversion::elem)
        .collect();

    Param::new(param_id, DataSerialize::new(data, shape)).serialize(serializer)
}

/// Helper function to insert a value into a nested map/vector of tensors.
fn insert_nested_value(current: &mut NestedValue, keys: &[&str], value: NestedValue) {
    if keys.is_empty() {
        *current = value;
        return;
    }

    match current {
        NestedValue::Map(map) => {
            if !map.contains_key(keys[0]) {
                let next = if keys[1..]
                    .first()
                    .and_then(|k| k.parse::<usize>().ok())
                    .is_some()
                {
                    NestedValue::Vec(Vec::new())
                } else {
                    NestedValue::Map(HashMap::new())
                };
                map.insert(keys[0].to_string(), next);
            }
            insert_nested_value(map.get_mut(keys[0]).unwrap(), &keys[1..], value);
        }
        NestedValue::Vec(vec) => {
            let index = keys[0].parse::<usize>().unwrap();
            if index >= vec.len() {
                vec.resize_with(index + 1, || NestedValue::Map(HashMap::new()));
            }
            insert_nested_value(&mut vec[index], &keys[1..], value);
        }
        _ => panic!("Invalid structure encountered"),
    }
}

/// Convert a vector of Candle tensors to a nested map/vector of tensors.
pub fn reverse_flatten<PS: PrecisionSettings>(input: HashMap<String, CandleTensor>) -> NestedValue {
    let mut result = NestedValue::Map(HashMap::new());

    for (key, value) in input {
        let parts: Vec<&str> = key.split('.').collect();
        let st = serialize_tensor::<_, PS>(&value, Serializer::new()).unwrap();

        insert_nested_value(&mut result, &parts, st);
    }

    result
}
