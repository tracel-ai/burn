use std::collections::HashMap;

use burn::record::serde::error::Error;
use serde::{
    ser::{self, SerializeSeq, SerializeStruct, Serializer as SerializerTrait},
    Serialize,
};

#[derive(Debug)]
pub enum NestedTensor {
    Map(HashMap<String, NestedTensor>),
    TensorId(String),
    Nothing,
}

impl NestedTensor {
    pub fn flatten(&self, prefix: String) -> HashMap<String, String> {
        let mut map = HashMap::new();
        match self {
            NestedTensor::Map(inner) => {
                for (key, value) in inner {
                    let new_prefix = if prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{}.{}", prefix, key)
                    };
                    map.extend(value.flatten(new_prefix));
                }
            }
            NestedTensor::TensorId(tensor) => {
                map.insert(prefix, tensor.clone());
            }
            NestedTensor::Nothing => {}
        }
        map
    }
}

#[derive(Debug)]
pub struct Serializer {
    /// The state of the serialization process
    state: Option<NestedTensor>,
}

impl Serializer {
    /// Creates a new serializer.
    pub fn new() -> Self {
        Serializer { state: None }
    }
}

impl Default for Serializer {
    fn default() -> Self {
        Self::new()
    }
}

impl SerializerTrait for Serializer {
    type Ok = NestedTensor;
    type Error = Error;
    type SerializeSeq = Self;
    type SerializeTuple = ser::Impossible<NestedTensor, Self::Error>;
    type SerializeTupleStruct = ser::Impossible<NestedTensor, Self::Error>;
    type SerializeTupleVariant = ser::Impossible<NestedTensor, Self::Error>;
    type SerializeMap = ser::Impossible<NestedTensor, Self::Error>;
    type SerializeStruct = Self;
    type SerializeStructVariant = ser::Impossible<NestedTensor, Self::Error>;

    fn serialize_struct(
        self,
        name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        if name == "TensorData" {
            Ok(self)
        } else {
            Ok(Serializer::new())
        }
    }

    fn serialize_newtype_struct<T>(
        self,
        _name: &'static str,
        _value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: Serialize + ?Sized,
    {
        Ok(NestedTensor::Nothing)
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        Ok(self)
    }

    fn serialize_i32(self, _v: i32) -> Result<Self::Ok, Self::Error> {
        Ok(NestedTensor::Nothing)
    }

    fn serialize_str(self, v: &str) -> Result<Self::Ok, Self::Error> {
        Ok(NestedTensor::TensorId(v.into()))
    }

    fn serialize_i16(self, _v: i16) -> Result<Self::Ok, Self::Error> {
        Ok(NestedTensor::Nothing)
    }

    fn serialize_i64(self, _v: i64) -> Result<Self::Ok, Self::Error> {
        Ok(NestedTensor::Nothing)
    }

    fn serialize_u16(self, _v: u16) -> Result<Self::Ok, Self::Error> {
        Ok(NestedTensor::Nothing)
    }

    fn serialize_u64(self, _v: u64) -> Result<Self::Ok, Self::Error> {
        Ok(NestedTensor::Nothing)
    }

    fn serialize_f32(self, _v: f32) -> Result<Self::Ok, Self::Error> {
        Ok(NestedTensor::Nothing)
    }

    fn serialize_f64(self, _v: f64) -> Result<Self::Ok, Self::Error> {
        Ok(NestedTensor::Nothing)
    }

    // The following methods are not implemented because they are not needed for the
    // serialization of Param structs.

    fn serialize_char(self, _v: char) -> Result<Self::Ok, Self::Error> {
        Ok(NestedTensor::Nothing)
    }

    fn serialize_bytes(self, _v: &[u8]) -> Result<Self::Ok, Self::Error> {
        Ok(NestedTensor::Nothing)
    }

    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        Ok(NestedTensor::Nothing)
    }
    fn serialize_u32(self, _v: u32) -> Result<Self::Ok, Self::Error> {
        Ok(NestedTensor::Nothing)
    }
    fn serialize_bool(self, _v: bool) -> Result<Self::Ok, Self::Error> {
        Ok(NestedTensor::Nothing)
    }

    fn serialize_i8(self, _v: i8) -> Result<Self::Ok, Self::Error> {
        Ok(NestedTensor::Nothing)
    }

    fn serialize_u8(self, _v: u8) -> Result<Self::Ok, Self::Error> {
        Ok(NestedTensor::Nothing)
    }

    fn serialize_some<T>(self, value: &T) -> Result<Self::Ok, Self::Error>
    where
        T: Serialize + ?Sized,
    {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        Ok(NestedTensor::Nothing)
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<Self::Ok, Self::Error> {
        Ok(NestedTensor::Nothing)
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
    ) -> Result<Self::Ok, Self::Error> {
        Ok(NestedTensor::Nothing)
    }

    fn serialize_newtype_variant<T>(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: Serialize + ?Sized,
    {
        Ok(NestedTensor::Nothing)
    }

    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        unimplemented!()
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        unimplemented!()
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        unimplemented!()
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        unimplemented!()
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        unimplemented!()
    }
}

// Implementing the SerializeStruct trait for Serializer
impl SerializeStruct for Serializer {
    type Ok = NestedTensor;
    type Error = Error;

    fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<(), Self::Error>
    where
        T: Serialize + ?Sized,
    {
        let serialized_value = value.serialize(Serializer::new())?;
        match self.state {
            Some(NestedTensor::Map(ref mut map)) => {
                map.insert(key.to_string(), serialized_value); // Inserting into the state
            }
            Some(_) => {
                panic!("Invalid state encountered");
            }
            None => {
                let mut map = HashMap::new();
                map.insert(key.to_string(), serialized_value); // Inserting into the state
                self.state = Some(NestedTensor::Map(map));
            }
        }
        Ok(())
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        if self.state.is_none() {
            // If the state is empty, return an empty map
            Ok(NestedTensor::Map(HashMap::new()))
        } else {
            self.state.ok_or(Error::InvalidState)
        }
    }
}

impl SerializeSeq for Serializer {
    type Ok = NestedTensor;
    type Error = Error;

    fn serialize_element<T>(&mut self, _value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        Ok(())
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(NestedTensor::Nothing)
    }
}
