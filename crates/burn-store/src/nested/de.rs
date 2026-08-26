use core::ptr;
use std::collections::HashMap;

use super::data::NestedValue;
use super::{adapter::BurnModuleAdapter, error::Error};

use serde::de::{EnumAccess, VariantAccess};
use serde::{
    de::{self, DeserializeSeed, IntoDeserializer, MapAccess, SeqAccess, Visitor},
    forward_to_deserialize_any,
};

const RECORD_ITEM_SUFFIX: &str = "RecordItem";

#[inline]
fn custom_err<T: core::fmt::Display>(msg: T) -> Error {
    <Error as de::Error>::custom(msg)
}

/// A deserializer for the nested value data structure.
pub struct Deserializer<A: BurnModuleAdapter> {
    // This string starts with the input data and characters are truncated off
    // the beginning as data is parsed.
    value: Option<NestedValue>,
    default_for_missing_fields: bool,
    phantom: std::marker::PhantomData<A>,
}

impl<A: BurnModuleAdapter> Deserializer<A> {
    /// Creates a new deserializer with the given nested value.
    ///
    /// # Arguments
    ///
    /// * `value` - A nested value.
    /// * `default_for_missing_fields` - A boolean indicating whether to add missing fields with default value.
    pub fn new(value: NestedValue, default_for_missing_fields: bool) -> Self {
        Self {
            value: Some(value),
            default_for_missing_fields,
            phantom: std::marker::PhantomData,
        }
    }

    fn extract_scalar<T>(
        self,
        expected: &'static str,
        extractor: impl FnOnce(NestedValue) -> Option<T>,
    ) -> Result<T, Error> {
        let value = self
            .value
            .ok_or_else(|| custom_err(format!("expected {expected}, found nothing")))?;

        let value_debug = format!("{value:?}");
        extractor(value)
            .ok_or_else(|| custom_err(format!("expected {expected} but got {value_debug}")))
    }
}

impl<'de, A: BurnModuleAdapter> serde::Deserializer<'de> for Deserializer<A> {
    type Error = Error;

    fn deserialize_any<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        unimplemented!("deserialize_any is not implemented")
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
        let value = match self.value {
            Some(value) => {
                // Adapt modules
                if let Some(name) = name.strip_suffix(RECORD_ITEM_SUFFIX) {
                    A::adapt(name, value)
                } else {
                    value
                }
            }
            None => {
                return Err(custom_err(format!(
                    "Expected some value but got {:?}",
                    self.value
                )));
            }
        };

        match value {
            NestedValue::Map(map) => {
                // Add missing fields into the map with default value if needed.
                let map = if self.default_for_missing_fields {
                    let mut map = map;
                    for field in fields.iter().map(|s| s.to_string()) {
                        map.entry(field.clone())
                            .or_insert(NestedValue::Default(Some(field)));
                    }
                    map
                } else {
                    map
                };

                visitor.visit_map(HashMapAccess::<A>::new(
                    map,
                    self.default_for_missing_fields,
                ))
            }

            _ => Err(custom_err(format!("Expected struct but got {value:?}"))),
        }
    }

    fn deserialize_string<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let val = self.extract_scalar("string", |v| v.as_string())?;
        visitor.visit_string(val)
    }

    fn deserialize_ignored_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_unit()
    }

    fn deserialize_map<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        match self.value {
            Some(NestedValue::Map(map)) => visitor.visit_map(HashMapAccess::<A>::new(
                map,
                self.default_for_missing_fields,
            )),

            _ => Err(custom_err(format!(
                "Expected map value but got {:?}",
                self.value
            ))),
        }
    }

    fn deserialize_bool<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let val = self.extract_scalar("bool", |v| v.as_bool())?;
        visitor.visit_bool(val)
    }

    fn deserialize_i8<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        unimplemented!("deserialize_i8 is not implemented")
    }

    fn deserialize_i16<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let val = self.extract_scalar("i16", |v| v.as_i16())?;
        visitor.visit_i16(val)
    }

    fn deserialize_i32<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let val = self.extract_scalar("i32", |v| v.as_i32())?;
        visitor.visit_i32(val)
    }

    fn deserialize_i64<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let val = self.extract_scalar("i64", |v| v.as_i64())?;
        visitor.visit_i64(val)
    }

    fn deserialize_u8<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let val = self.extract_scalar("u8", |v| v.as_u8())?;
        visitor.visit_u8(val)
    }

    fn deserialize_u16<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let val = self.extract_scalar("u16", |v| v.as_u16())?;
        visitor.visit_u16(val)
    }

    fn deserialize_u32<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        unimplemented!("deserialize_u32 is not implemented")
    }

    fn deserialize_u64<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let val = self.extract_scalar("u64", |v| v.as_u64())?;
        visitor.visit_u64(val)
    }

    fn deserialize_f32<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let val = self.extract_scalar("f32", |v| v.as_f32())?;
        visitor.visit_f32(val)
    }

    fn deserialize_f64<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let val = self.extract_scalar("f64", |v| v.as_f64())?;
        visitor.visit_f64(val)
    }

    fn deserialize_char<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        unimplemented!("deserialize_char is not implemented")
    }

    fn deserialize_str<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let val = self.extract_scalar("str", |v| v.as_string())?;
        visitor.visit_str(&val)
    }

    fn deserialize_bytes<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        unimplemented!("deserialize_bytes is not implemented")
    }

    fn deserialize_byte_buf<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let value = self
            .value
            .ok_or_else(|| custom_err("expected byte buffer, found nothing"))?;
        let value_debug = format!("{value:?}");
        let bytes = value
            .as_bytes()
            .ok_or_else(|| custom_err(format!("expected byte buffer but got {value_debug}")))?;
        match bytes.try_into_vec::<u8>() {
            Ok(bytes) => visitor.visit_byte_buf(bytes),
            Err(bytes) => visitor.visit_bytes(&bytes),
        }
    }

    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        if let Some(value) = self.value {
            visitor.visit_some(Deserializer::<A>::new(
                value,
                self.default_for_missing_fields,
            ))
        } else {
            visitor.visit_none()
        }
    }

    fn deserialize_unit<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        unimplemented!("deserialize_unit is not implemented")
    }

    fn deserialize_unit_struct<V>(
        self,
        _name: &'static str,
        _visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        unimplemented!("deserialize_unit_struct is not implemented")
    }

    fn deserialize_newtype_struct<V>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let value = self
            .value
            .ok_or_else(|| custom_err("expected value for newtype struct but got None"))?;
        visitor.visit_newtype_struct(Deserializer::<A>::new(
            value,
            self.default_for_missing_fields,
        ))
    }

    fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        if let Some(value) = self.value {
            match value {
                NestedValue::Vec(_) => visitor.visit_seq(VecSeqAccess::<A, NestedValue>::new(
                    value,
                    self.default_for_missing_fields,
                )?),
                NestedValue::U8s(_) => visitor.visit_seq(VecSeqAccess::<A, u8>::new(
                    value,
                    self.default_for_missing_fields,
                )?),
                NestedValue::U16s(_) => visitor.visit_seq(VecSeqAccess::<A, u16>::new(
                    value,
                    self.default_for_missing_fields,
                )?),
                NestedValue::F32s(_) => visitor.visit_seq(VecSeqAccess::<A, f32>::new(
                    value,
                    self.default_for_missing_fields,
                )?),
                _ => Err(custom_err(format!("Expected Vec but got {value:?}"))),
            }
        } else {
            Err(custom_err("Expected Vec but got None"))
        }
    }

    fn deserialize_tuple<V>(self, _len: usize, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        unimplemented!("deserialize_tuple is not implemented")
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
        unimplemented!("deserialize_tuple_struct is not implemented")
    }

    fn deserialize_enum<V>(
        self,
        _name: &'static str,
        variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        fn clone_unsafely<T>(thing: &T) -> T {
            unsafe {
                let mut clone = std::mem::MaybeUninit::<T>::uninit();
                let clone_ptr = clone.as_mut_ptr();
                ptr::copy_nonoverlapping(thing as *const T, clone_ptr, 1);
                clone.assume_init()
            }
        }

        let value = self
            .value
            .ok_or_else(|| custom_err("expected value for enum but got None"))?;

        // Try each variant in order
        for &variant in variants {
            let cloned_visitor = clone_unsafely(&visitor);
            let result = cloned_visitor.visit_enum(ProbeEnumAccess::<A>::new(
                value.clone(),
                variant.to_owned(),
                self.default_for_missing_fields,
            ));

            if result.is_ok() {
                return result;
            }
        }

        Err(custom_err("No variant match"))
    }

    fn deserialize_identifier<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        unimplemented!("deserialize_identifier is not implemented")
    }
}

/// A sequence access for a vector in the nested value data structure.
struct VecSeqAccess<A: BurnModuleAdapter, I> {
    iter: Box<dyn Iterator<Item = I>>,
    default_for_missing_fields: bool,
    phantom: std::marker::PhantomData<A>,
}

impl<A: BurnModuleAdapter> VecSeqAccess<A, NestedValue> {
    fn new(vec: NestedValue, default_for_missing_fields: bool) -> Result<Self, Error> {
        match vec {
            NestedValue::Vec(v) => Ok(VecSeqAccess {
                iter: Box::new(v.into_iter()),
                default_for_missing_fields,
                phantom: std::marker::PhantomData,
            }),
            other => Err(custom_err(format!("expected Vec, found {other:?}"))),
        }
    }
}

impl<A: BurnModuleAdapter> VecSeqAccess<A, u8> {
    fn new(vec: NestedValue, default_for_missing_fields: bool) -> Result<Self, Error> {
        match vec {
            NestedValue::U8s(v) => Ok(VecSeqAccess {
                iter: Box::new(v.into_iter()),
                default_for_missing_fields,
                phantom: std::marker::PhantomData,
            }),
            other => Err(custom_err(format!("expected U8s, found {other:?}"))),
        }
    }
}

impl<A: BurnModuleAdapter> VecSeqAccess<A, u16> {
    fn new(vec: NestedValue, default_for_missing_fields: bool) -> Result<Self, Error> {
        match vec {
            NestedValue::U16s(v) => Ok(VecSeqAccess {
                iter: Box::new(v.into_iter()),
                default_for_missing_fields,
                phantom: std::marker::PhantomData,
            }),
            other => Err(custom_err(format!("expected U16s, found {other:?}"))),
        }
    }
}

impl<A: BurnModuleAdapter> VecSeqAccess<A, f32> {
    fn new(vec: NestedValue, default_for_missing_fields: bool) -> Result<Self, Error> {
        match vec {
            NestedValue::F32s(v) => Ok(VecSeqAccess {
                iter: Box::new(v.into_iter()),
                default_for_missing_fields,
                phantom: std::marker::PhantomData,
            }),
            other => Err(custom_err(format!("expected F32s, found {other:?}"))),
        }
    }
}

impl<'de, A> SeqAccess<'de> for VecSeqAccess<A, NestedValue>
where
    NestedValueWrapper<A>: IntoDeserializer<'de, Error>,
    A: BurnModuleAdapter,
{
    type Error = Error;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        let item = match self.iter.next() {
            Some(v) => v,
            None => return Ok(None),
        };

        seed.deserialize(
            NestedValueWrapper::<A>::new(item, self.default_for_missing_fields).into_deserializer(),
        )
        .map(Some)
    }
}

impl<'de, A> SeqAccess<'de> for VecSeqAccess<A, u8>
where
    NestedValueWrapper<A>: IntoDeserializer<'de, Error>,
    A: BurnModuleAdapter,
{
    type Error = Error;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        let item = match self.iter.next() {
            Some(v) => v,
            None => return Ok(None),
        };

        seed.deserialize(
            NestedValueWrapper::<A>::new(NestedValue::U8(item), self.default_for_missing_fields)
                .into_deserializer(),
        )
        .map(Some)
    }
}

impl<'de, A> SeqAccess<'de> for VecSeqAccess<A, u16>
where
    NestedValueWrapper<A>: IntoDeserializer<'de, Error>,
    A: BurnModuleAdapter,
{
    type Error = Error;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        let item = match self.iter.next() {
            Some(v) => v,
            None => return Ok(None),
        };

        seed.deserialize(
            NestedValueWrapper::<A>::new(NestedValue::U16(item), self.default_for_missing_fields)
                .into_deserializer(),
        )
        .map(Some)
    }
}

impl<'de, A> SeqAccess<'de> for VecSeqAccess<A, f32>
where
    NestedValueWrapper<A>: IntoDeserializer<'de, Error>,
    A: BurnModuleAdapter,
{
    type Error = Error;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        let item = match self.iter.next() {
            Some(v) => v,
            None => return Ok(None),
        };

        seed.deserialize(
            NestedValueWrapper::<A>::new(NestedValue::F32(item), self.default_for_missing_fields)
                .into_deserializer(),
        )
        .map(Some)
    }
}

/// A map access for a map in the nested value data structure.
struct HashMapAccess<A: BurnModuleAdapter> {
    iter: std::collections::hash_map::IntoIter<String, NestedValue>,
    next_value: Option<NestedValue>,
    default_for_missing_fields: bool,
    phantom: std::marker::PhantomData<A>,
}

impl<A: BurnModuleAdapter> HashMapAccess<A> {
    fn new(map: HashMap<String, NestedValue>, default_for_missing_fields: bool) -> Self {
        HashMapAccess {
            iter: map.into_iter(),
            next_value: None,
            default_for_missing_fields,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<'de, A> MapAccess<'de> for HashMapAccess<A>
where
    String: IntoDeserializer<'de, Error>,
    NestedValueWrapper<A>: IntoDeserializer<'de, Error>,
    A: BurnModuleAdapter,
{
    type Error = Error;

    fn next_key_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        match self.iter.next() {
            Some((k, v)) => {
                self.next_value = Some(v);
                seed.deserialize(k.into_deserializer()).map(Some)
            }
            None => Ok(None),
        }
    }

    fn next_value_seed<T>(&mut self, seed: T) -> Result<T::Value, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        match self.next_value.take() {
            Some(NestedValue::Default(originator)) => {
                seed.deserialize(DefaultDeserializer::new(originator))
            }
            Some(v) => seed.deserialize(
                NestedValueWrapper::new(v, self.default_for_missing_fields).into_deserializer(),
            ),
            None => seed.deserialize(DefaultDeserializer::new(None)),
        }
    }
}

struct ProbeEnumAccess<A: BurnModuleAdapter> {
    value: NestedValue,
    current_variant: String,
    default_for_missing_fields: bool,
    phantom: std::marker::PhantomData<A>,
}

impl<A: BurnModuleAdapter> ProbeEnumAccess<A> {
    fn new(value: NestedValue, current_variant: String, default_for_missing_fields: bool) -> Self {
        ProbeEnumAccess {
            value,
            current_variant,
            default_for_missing_fields,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<'de, A> EnumAccess<'de> for ProbeEnumAccess<A>
where
    A: BurnModuleAdapter,
{
    type Error = Error;
    type Variant = Self;

    fn variant_seed<V>(self, seed: V) -> Result<(V::Value, Self::Variant), Self::Error>
    where
        V: DeserializeSeed<'de>,
    {
        seed.deserialize(self.current_variant.clone().into_deserializer())
            .map(|v| (v, self))
    }
}

impl<'de, A> VariantAccess<'de> for ProbeEnumAccess<A>
where
    A: BurnModuleAdapter,
{
    type Error = Error;

    fn newtype_variant_seed<T>(self, seed: T) -> Result<T::Value, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        let value = seed.deserialize(
            NestedValueWrapper::<A>::new(self.value, self.default_for_missing_fields)
                .into_deserializer(),
        )?;
        Ok(value)
    }

    fn unit_variant(self) -> Result<(), Self::Error> {
        match self.value {
            NestedValue::Map(value) if value.contains_key("DType") => match value.get("DType") {
                Some(NestedValue::String(variant)) => {
                    if *variant == self.current_variant {
                        Ok(())
                    } else {
                        Err(Error::Other("Wrong variant".to_string()))
                    }
                }
                _ => Err(custom_err("expected DType variant as string")),
            },
            _ => unimplemented!(
                "unit variant is not implemented because it is not used in the burn module"
            ),
        }
    }

    fn tuple_variant<V>(self, _len: usize, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        unimplemented!("tuple variant is not implemented because it is not used in the burn module")
    }

    fn struct_variant<V>(
        self,
        _fields: &'static [&'static str],
        _visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        unimplemented!(
            "struct variant is not implemented because it is not used in the burn module"
        )
    }
}

/// A wrapper for the nested value data structure with a burn module adapter.
struct NestedValueWrapper<A: BurnModuleAdapter> {
    value: NestedValue,
    default_for_missing_fields: bool,
    phantom: std::marker::PhantomData<A>,
}

impl<A: BurnModuleAdapter> NestedValueWrapper<A> {
    fn new(value: NestedValue, default_for_missing_fields: bool) -> Self {
        Self {
            value,
            default_for_missing_fields,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<A: BurnModuleAdapter> IntoDeserializer<'_, Error> for NestedValueWrapper<A> {
    type Deserializer = Deserializer<A>;

    fn into_deserializer(self) -> Self::Deserializer {
        Deserializer::<A>::new(self.value, self.default_for_missing_fields)
    }
}

/// A default deserializer that always returns the default value.
struct DefaultDeserializer {
    originator_field_name: Option<String>,
}

impl DefaultDeserializer {
    fn new(originator_field_name: Option<String>) -> Self {
        Self {
            originator_field_name,
        }
    }
}

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

    fn deserialize_i16<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i16(Default::default())
    }

    fn deserialize_i64<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i64(Default::default())
    }

    fn deserialize_u16<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u16(Default::default())
    }

    fn deserialize_u64<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u64(Default::default())
    }

    fn deserialize_f64<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_f64(Default::default())
    }

    fn deserialize_bool<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_bool(Default::default())
    }

    fn deserialize_char<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_char(Default::default())
    }

    fn deserialize_str<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_str(Default::default())
    }

    fn deserialize_i8<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i8(Default::default())
    }

    fn deserialize_u8<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u8(Default::default())
    }

    fn deserialize_u32<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u32(Default::default())
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
        visitor.visit_seq(DefaultSeqAccess::new(None))
    }

    fn deserialize_string<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_string(Default::default())
    }

    fn deserialize_struct<V>(
        self,
        name: &'static str,
        _fields: &'static [&'static str],
        _visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        Err(Error::Other(format!(
            "Missing source values for the '{}' field of type '{}'. Please verify the source data and ensure the field name is correct",
            self.originator_field_name.unwrap_or("UNKNOWN".to_string()),
            name,
        )))
    }

    fn deserialize_tuple_struct<V>(
        self,
        _name: &'static str,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_seq(DefaultSeqAccess::new(Some(len)))
    }

    fn deserialize_tuple<V>(self, len: usize, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_seq(DefaultSeqAccess::new(Some(len)))
    }

    fn deserialize_map<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_map(DefaultMapAccess::new())
    }

    forward_to_deserialize_any! {
        u128 bytes byte_buf unit unit_struct newtype_struct
        enum identifier ignored_any
    }
}

/// A default sequence access that always returns None (empty sequence).
pub struct DefaultSeqAccess {
    size: Option<usize>,
}

impl Default for DefaultSeqAccess {
    fn default() -> Self {
        Self::new(None)
    }
}

impl DefaultSeqAccess {
    /// Creates a new default sequence access with the given size hint.
    pub fn new(size: Option<usize>) -> Self {
        DefaultSeqAccess { size }
    }
}

impl<'de> SeqAccess<'de> for DefaultSeqAccess {
    type Error = Error;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        match self.size {
            Some(0) => Ok(None),
            Some(ref mut size) => {
                *size -= 1;
                seed.deserialize(DefaultDeserializer::new(None)).map(Some)
            }
            None => Ok(None),
        }
    }

    fn size_hint(&self) -> Option<usize> {
        self.size
    }
}

/// A default map access that always returns None (empty map).
pub struct DefaultMapAccess;

impl Default for DefaultMapAccess {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultMapAccess {
    /// Creates a new default map access.
    pub fn new() -> Self {
        DefaultMapAccess
    }
}

impl<'de> MapAccess<'de> for DefaultMapAccess {
    type Error = Error;

    fn next_key_seed<T>(&mut self, _seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        Ok(None)
    }

    fn next_value_seed<T>(&mut self, _seed: T) -> Result<T::Value, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        unimplemented!("This should never be called since next_key_seed always returns None")
    }

    fn size_hint(&self) -> Option<usize> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nested::adapter::DefaultAdapter;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct AllScalars {
        b: bool,
        i16_val: i16,
        i32_val: i32,
        i64_val: i64,
        u8_val: u8,
        u16_val: u16,
        u64_val: u64,
        f32_val: f32,
        f64_val: f64,
        s: String,
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct Config {
        hidden_size: i32,
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct Newtype(i32);

    #[test]
    fn should_deserialize_all_scalars_correctly() {
        let mut map = HashMap::new();
        map.insert("b".to_string(), NestedValue::Bool(true));
        map.insert("i16_val".to_string(), NestedValue::I16(16));
        map.insert("i32_val".to_string(), NestedValue::I32(32));
        map.insert("i64_val".to_string(), NestedValue::I64(64));
        map.insert("u8_val".to_string(), NestedValue::U8(8));
        map.insert("u16_val".to_string(), NestedValue::U16(16));
        map.insert("u64_val".to_string(), NestedValue::U64(64));
        map.insert("f32_val".to_string(), NestedValue::F32(1.25));
        map.insert("f64_val".to_string(), NestedValue::F64(2.5));
        map.insert("s".to_string(), NestedValue::String("burn".to_string()));

        let de = Deserializer::<DefaultAdapter>::new(NestedValue::Map(map), false);
        let config = AllScalars::deserialize(de).unwrap();

        assert_eq!(
            config,
            AllScalars {
                b: true,
                i16_val: 16,
                i32_val: 32,
                i64_val: 64,
                u8_val: 8,
                u16_val: 16,
                u64_val: 64,
                f32_val: 1.25,
                f64_val: 2.5,
                s: "burn".to_string(),
            }
        );
    }

    #[test]
    fn should_return_err_on_type_mismatch() {
        let mut map = HashMap::new();
        map.insert(
            "hidden_size".to_string(),
            NestedValue::String("768".to_string()),
        );

        let de = Deserializer::<DefaultAdapter>::new(NestedValue::Map(map), false);
        let result = Config::deserialize(de);

        assert!(result.is_err());
    }

    #[test]
    fn should_return_err_on_missing_field() {
        let map = HashMap::new();
        let de = Deserializer::<DefaultAdapter>::new(NestedValue::Map(map), false);
        let result = Config::deserialize(de);

        assert!(result.is_err());
    }

    #[test]
    fn should_deserialize_newtype_struct() {
        let de = Deserializer::<DefaultAdapter>::new(NestedValue::I32(42), false);
        let result = Newtype::deserialize(de).unwrap();
        assert_eq!(result, Newtype(42));
    }

    #[test]
    fn should_return_err_on_invalid_seq() {
        #[derive(Debug, Deserialize)]
        #[allow(dead_code)]
        struct SeqHolder(Vec<i32>);

        let de = Deserializer::<DefaultAdapter>::new(NestedValue::I32(42), false);
        let result = SeqHolder::deserialize(de);
        assert!(result.is_err());
    }
}
