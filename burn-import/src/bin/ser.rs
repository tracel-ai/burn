use serde::{
    de,
    ser::{self, SerializeStruct, Serializer},
    Deserialize, Serialize,
};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct MyStruct1 {
    a: MyStruct3,
    b: MyStruct2,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct MyStruct2 {
    a: i32,
    b: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct MyStruct3 {
    x: String,
    y: String,
}

#[derive(Debug, Clone)]
enum Values {
    StructValue(HashMap<String, Values>),
    StringValue(String),
    IntValue(i32),
}

struct MySerializer {
    // This will hold the state of the serialization process
    state: HashMap<String, Values>,
}

impl MySerializer {
    fn new() -> Self {
        MySerializer {
            state: HashMap::new(),
        }
    }
}

impl Serializer for MySerializer {
    type Ok = Values;
    type Error = de::value::Error;
    type SerializeSeq = ser::Impossible<Values, Self::Error>;
    type SerializeTuple = ser::Impossible<Values, Self::Error>;
    type SerializeTupleStruct = ser::Impossible<Values, Self::Error>;
    type SerializeTupleVariant = ser::Impossible<Values, Self::Error>;
    type SerializeMap = ser::Impossible<Values, Self::Error>;
    type SerializeStruct = Self;
    type SerializeStructVariant = ser::Impossible<Values, Self::Error>;

    // Here you'll need to implement methods for each type of serialization you want to support.
    // For your case, you are interested in serializing structs and primitive types like i32 and String.
    // Below is a simplistic implementation handling only some types:

    fn serialize_i32(self, v: i32) -> Result<Self::Ok, Self::Error> {
        Ok(Values::IntValue(v))
    }

    fn serialize_str(self, v: &str) -> Result<Self::Ok, Self::Error> {
        Ok(Values::StringValue(v.to_owned()))
    }

    fn serialize_bool(self, _v: bool) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_i8(self, _v: i8) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_i16(self, _v: i16) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_i64(self, _v: i64) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_u8(self, _v: u8) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_u16(self, _v: u16) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_u32(self, _v: u32) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_u64(self, _v: u64) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_f32(self, _v: f32) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_f64(self, _v: f64) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_char(self, _v: char) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_bytes(self, _v: &[u8]) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_some<T: ?Sized>(self, _value: &T) -> Result<Self::Ok, Self::Error>
    where
        T: Serialize,
    {
        todo!()
    }

    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
    ) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_newtype_struct<T: ?Sized>(
        self,
        _name: &'static str,
        _value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: Serialize,
    {
        todo!()
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
        todo!()
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        todo!()
    }

    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        todo!()
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        todo!()
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        todo!()
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        todo!()
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

    // ... Implement other required methods with dummy or appropriate logic ...

    // Implement other methods as needed, returning errors for unsupported types
}

// Implementing the SerializeStruct trait for MySerializer
impl SerializeStruct for MySerializer {
    type Ok = Values;
    type Error = de::value::Error;

    fn serialize_field<T: ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Self::Error>
    where
        T: Serialize,
    {
        let serialized_value = value.serialize(MySerializer::new())?;
        self.state.insert(key.to_string(), serialized_value); // Inserting into the state
        Ok(())
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        // Convert the accumulated state into a Values::StructValue
        Ok(Values::StructValue(self.state))
    }
}

// Finally, provide a function to initiate serialization of MyStruct1
fn serialize_my_struct<S>(my_struct: &MyStruct1, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    my_struct.serialize(serializer)
}

// Usage example
fn main() {
    let my_struct = MyStruct1 {
        a: MyStruct3 {
            x: "Hello".to_owned(),
            y: "World".to_owned(),
        },
        b: MyStruct2 { a: 1, b: 2 },
    };

    let serialized = serialize_my_struct(&my_struct, MySerializer::new()).unwrap();
    println!("{:?}", serialized);
}
