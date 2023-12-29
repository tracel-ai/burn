use serde::de::{self, DeserializeSeed, IntoDeserializer, SeqAccess, Visitor};
use serde::{forward_to_deserialize_any, Deserialize, Serialize};
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
    // c will be missing in the input and should be set to 0 (default value)
    c: i32,
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
    DefaultValue,
}

// A Deserializer that works specifically with Values.
struct ValuesDeserializer {
    value: Option<Values>,
}

impl ValuesDeserializer {
    fn new(value: Values) -> Self {
        ValuesDeserializer { value: Some(value) }
    }
}

// Implementing the Deserializer trait for ValuesDeserializer.
// This involves providing implementations for the various methods required to interpret data.
impl<'de> de::Deserializer<'de> for ValuesDeserializer {
    type Error = de::value::Error;

    // This is a simplified version of what methods you might need.
    // You'll need to implement the methods specific to the types you're working with.

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        match self.value {
            Some(Values::StructValue(map)) => visitor.visit_map(MyMapAccess::new(map)),
            Some(Values::StringValue(s)) => visitor.visit_string(s),
            Some(Values::IntValue(i)) => visitor.visit_i32(i),
            Some(_) => Err(de::Error::custom("Expected struct, string or int")),
            // Some(Values::DefaultValue) => visitor.visit_i32(Default::default()),
            None => Err(de::Error::custom("No value to deserialize")),
        }
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
            Some(Values::StructValue(map)) => {
                // Add missing fields into the map with default value
                let mut map = map;
                for field in fields.iter().map(|s| s.to_string()) {
                    map.entry(field).or_insert(Values::DefaultValue);
                }
                visitor.visit_map(MyMapAccess::new(map))
            }
            _ => Err(de::Error::custom("Expected struct")),
        }
    }

    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map enum identifier ignored_any
    }
}

use serde::de::MapAccess;

struct MyMapAccess {
    iter: std::collections::hash_map::IntoIter<String, Values>,
    next_value: Option<Values>,
}

impl MyMapAccess {
    fn new(map: HashMap<String, Values>) -> Self {
        MyMapAccess {
            iter: map.into_iter(),
            next_value: None,
        }
    }
}

impl<'de> MapAccess<'de> for MyMapAccess
where
    String: IntoDeserializer<'de>,
    Values: IntoDeserializer<'de> + Clone,
{
    type Error = serde::de::value::Error;

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
            Some(Values::DefaultValue) => seed.deserialize(DefaultDeserializer),
            Some(v) => seed.deserialize(v.into_deserializer()),
            None => Err(serde::de::Error::custom("value missing")),
            // None => seed.deserialize(DefaultDeserializer),
        }
    }
}

impl<'de> IntoDeserializer<'de, de::value::Error> for Values {
    type Deserializer = ValuesDeserializer;

    fn into_deserializer(self) -> Self::Deserializer {
        ValuesDeserializer::new(self)
    }
}

struct DefaultDeserializer;

impl<'de> de::Deserializer<'de> for DefaultDeserializer {
    type Error = de::value::Error;

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
    type Error = de::value::Error;

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

fn main() {
    // Constructing MyStruct3 representation.
    let mut struct3_map = HashMap::new();
    struct3_map.insert("x".to_string(), Values::StringValue("Hello".to_string()));
    struct3_map.insert("y".to_string(), Values::StringValue("World".to_string()));

    // Constructing MyStruct2 representation.
    let mut struct2_map = HashMap::new();
    struct2_map.insert("a".to_string(), Values::IntValue(1));
    struct2_map.insert("b".to_string(), Values::IntValue(2));

    // Constructing MyStruct1 representation.
    let mut struct1_map = HashMap::new();
    struct1_map.insert("a".to_string(), Values::StructValue(struct3_map));
    struct1_map.insert("b".to_string(), Values::StructValue(struct2_map));

    // The struct1_map is now a representation of MyStruct1 with the corresponding values.
    // Use this map to create a Values representing MyStruct1.
    let values = Values::StructValue(struct1_map);

    // Create the deserializer.
    let deserializer = ValuesDeserializer::new(values);

    // Deserialize into MyStruct1.
    // This requires that MyStruct1 implements the `Deserialize` trait, which might be derived.
    let result = MyStruct1::deserialize(deserializer);

    // Handle the result.
    match result {
        Ok(my_struct1) => println!("Deserialized MyStruct1: {:?}", my_struct1),
        Err(e) => println!("Failed to deserialize: {:?}", e),
    }
}
