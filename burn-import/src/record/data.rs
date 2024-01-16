use std::collections::HashMap;

use burn::record::{PrecisionSettings, Record};
use regex::Regex;
use serde::Deserialize;

use super::adapter::DefaultAdapter;
use super::de::Deserializer;
use super::error::Error;
use super::ser::Serializer;

/// A nested map/vector of tensors.
#[derive(Debug, Clone)]
pub enum NestedValue {
    /// The default value (typically for primitives like integers)
    Default,

    /// A string value
    String(String),

    /// Floating point 32-bit value
    F32(f32),

    /// Floating point 64-bit value
    F64(f64),

    /// Signed 16-bit integer value
    I16(i16),

    /// Signed 32-bit integer value
    I32(i32),

    /// Signed 64-bit integer value
    I64(i64),

    /// Unsigned 16-bit integer value used for bf16 and f16 serialization
    U16(u16),

    /// Unsigned 64-bit integer value
    U64(u64),

    /// A map of nested values (typically used for structs)
    Map(HashMap<String, NestedValue>),

    /// A vector of nested values (typically used for vector of structs)
    Vec(Vec<NestedValue>),
}

impl NestedValue {
    pub fn get_map(&self) -> Option<&HashMap<String, NestedValue>> {
        match self {
            NestedValue::Map(map) => Some(map),
            _ => None,
        }
    }

    pub fn get_string(&self) -> Option<&str> {
        match self {
            NestedValue::String(string) => Some(string),
            _ => None,
        }
    }

    pub fn get_f32(&self) -> Option<f32> {
        match self {
            NestedValue::F32(f32) => Some(*f32),
            _ => None,
        }
    }

    pub fn get_f64(&self) -> Option<f64> {
        match self {
            NestedValue::F64(f64) => Some(*f64),
            _ => None,
        }
    }

    pub fn get_i16(&self) -> Option<i16> {
        match self {
            NestedValue::I16(i16) => Some(*i16),
            _ => None,
        }
    }

    pub fn get_i32(&self) -> Option<i32> {
        match self {
            NestedValue::I32(i32) => Some(*i32),
            _ => None,
        }
    }

    pub fn get_i64(&self) -> Option<i64> {
        match self {
            NestedValue::I64(i64) => Some(*i64),
            _ => None,
        }
    }

    pub fn get_u16(&self) -> Option<u16> {
        match self {
            NestedValue::U16(u16) => Some(*u16),
            _ => None,
        }
    }

    pub fn get_u64(&self) -> Option<u64> {
        match self {
            NestedValue::U64(u64) => Some(*u64),
            _ => None,
        }
    }

    /// Deserialize a nested value into a record type.
    ///
    /// NOTE: Deserialization is done using the default adapter (see `DefaultAdapter`).
    pub fn de_into<T, PS>(self) -> Result<T, Error>
    where
        T: Record,
        PS: PrecisionSettings,
    {
        let deserializer = Deserializer::<DefaultAdapter>::new(self);

        let item = T::Item::deserialize(deserializer)?;

        // Convert the deserialized item into a Record instance
        Ok(T::from_item::<PS>(item))
    }
}

/// Remap the tensor locations according to the key remapping.
pub fn remap<T>(
    mut tensors: HashMap<String, T>,
    key_remap: Vec<(Regex, String)>,
) -> HashMap<String, T> {
    if key_remap.is_empty() {
        return tensors;
    }

    let mut remapped = HashMap::new();

    for (name, tensor) in tensors.drain() {
        let mut new_name = name.clone();
        for (pattern, replacement) in &key_remap {
            if pattern.is_match(&name) {
                new_name = pattern.replace_all(&name, replacement.as_str()).to_string();
                break;
            }
        }
        remapped.insert(new_name, tensor);
    }

    remapped
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

pub trait Serializable {
    fn serialize<PS, S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
        PS: PrecisionSettings;
}

/// Convert a vector of Candle tensors to a nested map/vector of tensors.
pub fn unflatten<PS, T>(input: HashMap<String, T>) -> NestedValue
where
    PS: PrecisionSettings,
    T: Serializable,
{
    let mut result = NestedValue::Map(HashMap::new());

    for (key, value) in input {
        let parts: Vec<&str> = key.split('.').collect();
        let st = value.serialize::<PS, _>(Serializer::new()).unwrap();

        insert_nested_value(&mut result, &parts, st);
    }

    result
}
