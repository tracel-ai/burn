use std::collections::HashMap;

use std::path::Path;

use super::{converter::reverse_flatten, error::Error, reader::NestedValue, PyTorchFileRecorder};

use candle_core::{pickle, Tensor as CandleTensor};
use serde::de::{
    DeserializeOwned, // DeserializeSeed, EnumAccess, IntoDeserializer, MapAccess,
    DeserializeSeed,
    IntoDeserializer,
    SeqAccess,
    // SeqAccess, VariantAccess,
    Visitor, self,
};

// use thiserror::Error;

// struct Error;

// // Bincode deserializer: https://github.com/bincode-org/bincode/blob/980e4029552e416c1fd2f0699a78d33562b33966/src/features/serde/de_owned.rs#L346

pub fn from_file<'de, D>(path: &Path) -> Result<D, Error>
where
    D: DeserializeOwned,
{
    // Read the pickle file and return a vector of Candle tensors
    let tensors: HashMap<String, CandleTensor> =
        pickle::read_all(path).unwrap().into_iter().collect();

    // Remap the keys (replace the keys in the map with the new keys)
    // let remapped_tensor = self.remap(tensors);

    // Convert the vector of Candle tensors to a nested map/vector of tensors
    let nested_value = reverse_flatten(tensors);

    let mut deserializer = Deserializer::new(nested_value);
    let value = D::deserialize(deserializer)?;
    Ok(value)
}

pub struct Deserializer {
    // This string starts with the input data and characters are truncated off
    // the beginning as data is parsed.
    value: Option<NestedValue>,
}

impl Deserializer {
    fn new(value: NestedValue) -> Self {
        Self { value: Some(value) }
    }
}

impl<'de> serde::Deserializer<'de> for Deserializer {
    type Error = Error;

    fn deserialize_any<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        // match self.value {
        //     Some(NestedValue::Map(map)) => visitor.visit_map(MyMapAccess::new(map)),
        //     Some(NestedValue::String(s)) => visitor.visit_string(s),
        //     Some(NestedValue::Tensor(t)) => visitor.visit_i32(i),
        //     Some(_) => Err(de::Error::custom("Expected struct, string or int")),
        //     // Some(Values::DefaultValue) => visitor.visit_i32(Default::default()),
        //     None => Err(de::Error::custom("No value to deserialize")),
        // }

        unimplemented!()
    }

    fn deserialize_struct<V>(
        self,
        name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        println!("deserialize_struct");
        println!("self.value: {:?}", self.value);
        println!("name: {:?}", name);
        println!("fields: {:?}", fields);

        // Add missing fields into the map

        match self.value {
            Some(NestedValue::Map(map)) => {
                // Add missing fields into the map with default value
                let mut map = map;
                for field in fields.iter().map(|s| s.to_string()) {
                    map.entry(field).or_insert(NestedValue::Default);
                }

                visitor.visit_map(HashMapAccess::new(map))
            }
            _ => Err(de::Error::custom("Expected struct")),
        }
    }

    fn deserialize_map<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_bool<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_i8<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_i16<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_i32<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_i64<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_u8<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_u16<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_u32<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_u64<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_f32<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_f64<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_char<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_str<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_string<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_bytes<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_byte_buf<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_option<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_unit<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_unit_struct<V>(
        self,
        _name: &'static str,
        _visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_newtype_struct<V>(
        self,
        _name: &'static str,
        _visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_seq<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_tuple<V>(self, _len: usize, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_tuple_struct<V>(
        self,
        _name: &'static str,
        _len: usize,
        _visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_enum<V>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        _visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_identifier<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_ignored_any<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        todo!()
    }
}

use serde::de::MapAccess;
use serde::forward_to_deserialize_any;

struct HashMapAccess {
    iter: std::collections::hash_map::IntoIter<String, NestedValue>,
    next_value: Option<NestedValue>,
}

impl HashMapAccess {
    fn new(map: HashMap<String, NestedValue>) -> Self {
        HashMapAccess {
            iter: map.into_iter(),
            next_value: None,
        }
    }
}

impl<'de> MapAccess<'de> for HashMapAccess
where
    String: IntoDeserializer<'de>,
    NestedValue: IntoDeserializer<'de> + Clone,
{
    type Error = Error;

    fn next_key_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        match self.iter.next() {
            Some((k, v)) => {
                // Keep the value for the next call to next_value_seed.
                self.next_value = Some(v);
                // Deserialize the key.
                seed.deserialize(k.into_deserializer()).map(Some)
            }
            None => Ok(None),
            // None => seed.deserialize(k.into_deserializer()).map(Some),
        }
    }

    fn next_value_seed<T>(&mut self, seed: T) -> Result<T::Value, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        match self.next_value.take() {
            Some(NestedValue::Default) => seed.deserialize(DefaultDeserializer),
            // Some(v) => seed.deserialize(v.into_deserializer()),
            // None => Err(serde::de::Error::custom("value missing")),
            None => seed.deserialize(DefaultDeserializer),
        }
    }
}

impl<'de> IntoDeserializer<'de, Error> for NestedValue {
    type Deserializer = Deserializer;

    fn into_deserializer(self) -> Self::Deserializer {
        Deserializer::new(self)
    }
}

struct DefaultDeserializer;

impl<'de> serde::Deserializer<'de> for DefaultDeserializer {
    type Error = Error;

    fn deserialize_any<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        unimplemented!()
    }

    fn deserialize_i32<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i32(Default::default())
    }

    fn deserialize_f32<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_f32(Default::default())
    }

    fn deserialize_string<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_string(Default::default())
    }

    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_none()
    }

    fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_seq(DefaultSeqAccess::new())
    }

    // TODO: Implement the rest of the methods.

    forward_to_deserialize_any! {
        bool i8 i16 i64 i128 u8 u16 u32 u64 u128 f64 char str
        bytes byte_buf unit unit_struct newtype_struct tuple
        tuple_struct map enum identifier ignored_any struct
    }
}

pub struct DefaultSeqAccess;

impl Default for DefaultSeqAccess {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultSeqAccess {
    pub fn new() -> Self {
        DefaultSeqAccess
    }
}

impl<'de> SeqAccess<'de> for DefaultSeqAccess {
    type Error = Error;

    fn next_element_seed<T>(&mut self, _seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        // Since this is a default implementation, we'll just return None.
        Ok(None)
    }

    fn size_hint(&self) -> Option<usize> {
        // Since this is a default implementation, we'll just return None.
        None
    }
}

// use serde::de::{Deserialize, MapAccess};
// use std::hash::Hash;

// pub struct MyMapAccess<K, V> {
//     iter: std::collections::hash_map::IntoIter<K, V>,
//     next_value: Option<V>,
// }

// impl<K, V> MyMapAccess<K, V> {
//     pub fn new(map: HashMap<K, V>) -> Self {
//         MyMapAccess {
//             iter: map.into_iter(),
//             next_value: None,
//         }
//     }
// }

// impl<'de, K, V> MapAccess<'de> for MyMapAccess<K, V>
// where
//     K: Deserialize<'de> + IntoDeserializer<'de> + Eq + Hash + Clone,
//     V: Deserialize<'de> + IntoDeserializer<'de> + Clone,
// {
//     type Error = serde::de::value::Error;

//     fn next_key_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
//     where
//         T: DeserializeSeed<'de>,
//     {
//         match self.iter.next() {
//             Some((k, v)) => {
//                 // Keep the value for the next call to next_value_seed.
//                 self.next_value = Some(v);
//                 // Deserialize the key.
//                 seed.deserialize(k.into_deserializer()).map(Some)
//             }
//             None => Ok(None),
//         }
//     }

//     fn next_value_seed<T>(&mut self, seed: T) -> Result<T::Value, Self::Error>
//     where
//         T: DeserializeSeed<'de>,
//     {
//         match self.next_value.take() {
//             Some(v) => seed.deserialize(v.into_deserializer()),
//             None => Err(serde::de::Error::custom("value missing")),
//         }
//     }
// }

// //--------
