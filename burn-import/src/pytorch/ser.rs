use std::collections::HashMap;

use super::{error::Error, reader::NestedValue};

use serde::{
    ser::{self, SerializeStruct, Serializer},
    Serialize,
};


// Simple struct serializer that converts a struct into NestedValues
pub (crate) struct StructSerializer {
    // This will hold the state of the serialization process
    state: HashMap<String, NestedValue>,
}

impl StructSerializer {
    fn new() -> Self {
        StructSerializer {
            state: HashMap::new(),
        }
    }
}

impl Serializer for StructSerializer {
    type Ok = NestedValue;
    type Error = Error;
    type SerializeSeq = ser::Impossible<NestedValue, Self::Error>;
    type SerializeTuple = ser::Impossible<NestedValue, Self::Error>;
    type SerializeTupleStruct = ser::Impossible<NestedValue, Self::Error>;
    type SerializeTupleVariant = ser::Impossible<NestedValue, Self::Error>;
    type SerializeMap = ser::Impossible<NestedValue, Self::Error>;
    type SerializeStruct = Self;
    type SerializeStructVariant = ser::Impossible<NestedValue, Self::Error>;

    fn serialize_i32(self, _v: i32) -> Result<Self::Ok, Self::Error> {
        Ok(NestedValue::Default)
    }

    fn serialize_str(self, v: &str) -> Result<Self::Ok, Self::Error> {
        Ok(NestedValue::String(v.to_string()))
    }

    fn serialize_bool(self, _v: bool) -> Result<Self::Ok, Self::Error> {
        Ok(NestedValue::Default)
    }

    fn serialize_i8(self, _v: i8) -> Result<Self::Ok, Self::Error> {
        Ok(NestedValue::Default)
    }

    fn serialize_i16(self, _v: i16) -> Result<Self::Ok, Self::Error> {
        Ok(NestedValue::Default)
    }

    fn serialize_i64(self, _v: i64) -> Result<Self::Ok, Self::Error> {
        Ok(NestedValue::Default)
    }

    fn serialize_u8(self, _v: u8) -> Result<Self::Ok, Self::Error> {
        Ok(NestedValue::Default)
    }

    fn serialize_u16(self, _v: u16) -> Result<Self::Ok, Self::Error> {
        Ok(NestedValue::Default)
    }

    fn serialize_u32(self, _v: u32) -> Result<Self::Ok, Self::Error> {
        Ok(NestedValue::Default)
    }

    fn serialize_u64(self, _v: u64) -> Result<Self::Ok, Self::Error> {
        Ok(NestedValue::Default)
    }

    fn serialize_f32(self, _v: f32) -> Result<Self::Ok, Self::Error> {
        Ok(NestedValue::Default)
    }

    fn serialize_f64(self, _v: f64) -> Result<Self::Ok, Self::Error> {
        Ok(NestedValue::Default)
    }

    fn serialize_char(self, _v: char) -> Result<Self::Ok, Self::Error> {
        Ok(NestedValue::Default)
    }

    fn serialize_bytes(self, _v: &[u8]) -> Result<Self::Ok, Self::Error> {
        Ok(NestedValue::Default)
    }

    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        Ok(NestedValue::Default)
    }

    fn serialize_some<T: ?Sized>(self, value: &T) -> Result<Self::Ok, Self::Error>
    where
        T: Serialize,
    {
        value.serialize(StructSerializer::new())
    }

    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        unimplemented!("serialize_unit not implemented for StructSerializer")
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<Self::Ok, Self::Error> {
        unimplemented!("serialize_unit_struct not implemented for StructSerializer")
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
    ) -> Result<Self::Ok, Self::Error> {
        unimplemented!("serialize_unit_variant not implemented for StructSerializer")
    }

    fn serialize_newtype_struct<T: ?Sized>(
        self,
        _name: &'static str,
        _value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: Serialize,
    {
        unimplemented!("serialize_newtype_struct not implemented for StructSerializer")
    }

    fn serialize_newtype_variant<T: ?Sized>(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: Serialize,
    {
        unimplemented!("serialize_newtype_variant not implemented for StructSerializer")
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        todo!()
    }

    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        unimplemented!("serialize_tuple not implemented for StructSerializer")
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        unimplemented!("serialize_tuple_struct not implemented for StructSerializer")
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        unimplemented!("serialize_tuple_variant not implemented for StructSerializer")
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        unimplemented!("serialize_map not implemented for StructSerializer")
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        todo!()
    }

    fn serialize_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        Ok(self)
    }
}

// Implementing the SerializeStruct trait for StructSerializer
impl<'a> SerializeStruct for StructSerializer {
    type Ok = NestedValue;
    type Error = Error;

    fn serialize_field<T: ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Self::Error>
    where
        T: Serialize,
    {
        let serialized_value = value.serialize(StructSerializer::new())?;
        self.state.insert(key.to_string(), serialized_value); // Inserting into the state
        Ok(())
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        // Convert the accumulated state into a Values::StructValue
        Ok(NestedValue::Map(self.state))
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::*;

    #[derive(Serialize, Deserialize, Debug, Clone)]
    struct MyStruct1 {
        a: MyStruct3,
        b: MyStruct2,
    }

    #[derive(Serialize, Deserialize, Debug, Clone)]
    struct MyStruct2 {
        a: i32,
        b: Option<i32>,
        c: String,
        d: Option<String>,
    }

    #[derive(Serialize, Deserialize, Debug, Clone)]
    struct MyStruct3 {
        x: String,
        y: String,
    }

    #[test]
    fn test_serialize() {
        let my_struct = MyStruct1 {
            a: MyStruct3 {
                x: "Hello".to_owned(),
                y: "World".to_owned(),
            },
            b: MyStruct2 {
                a: 1,
                b: None,
                c: "Hello".to_owned(),
                d: Some("World".to_owned()),
            },
        };

        let serialized = my_struct.serialize(StructSerializer::new()).unwrap();

        let serialized_str = format!("{:?}", serialized);

        // Compare the lengths of expected and actual serialized strings because
        // the order of the fields is not guaranteed for HashMaps.
        assert_eq!(
            serialized_str.len(),
            concat!(
            r#"Map({"b": Map({"a": Default, "c": String("Hello"), "b": Default, "d": String("World")}),"#,
            r#" "a": Map({"x": String("Hello"), "y": String("World")})})"#
            ).len()
        );
    }
}
