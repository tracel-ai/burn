//! Deserialization of a database row into a user type.
//!
//! Turso has no serde integration, so this reimplements the mapping `serde_rusqlite` used to
//! provide: a row is handed to serde as a map from column name to column value. That is what lets
//! a target struct name a subset of the table's columns, in any order.

use serde::de::value::{MapDeserializer, SeqDeserializer};
use serde::de::{DeserializeOwned, Deserializer, IntoDeserializer, Visitor};
use serde::forward_to_deserialize_any;
use turso::{Row, Value};

/// Error raised while deserializing a row into the target type.
#[derive(thiserror::Error, Debug)]
pub enum RowError {
    /// A column could not be read out of the row.
    #[error("column `{column}`: {source}")]
    Column {
        /// Name of the offending column.
        column: String,
        /// The underlying database error.
        source: turso::Error,
    },

    /// Serde could not build the target type from the row.
    #[error("{0}")]
    Message(String),
}

impl serde::de::Error for RowError {
    fn custom<T: core::fmt::Display>(msg: T) -> Self {
        RowError::Message(msg.to_string())
    }
}

/// Deserializes a row into `I`, matching the fields of `I` against `columns` by name.
///
/// `columns` must line up positionally with the row's columns, which is exactly what
/// [`turso::Statement::column_names`] reports for the statement that produced the row.
pub fn from_row_with_columns<I: DeserializeOwned>(
    row: &Row,
    columns: &[String],
) -> Result<I, RowError> {
    let mut entries = Vec::with_capacity(columns.len());
    for (index, column) in columns.iter().enumerate() {
        let value = row.get_value(index).map_err(|source| RowError::Column {
            column: column.clone(),
            source,
        })?;
        entries.push((column.as_str(), ValueDeserializer(value)));
    }

    I::deserialize(RowDeserializer { entries })
}

/// Deserializes a complete row, either by column name or position depending on the target type.
struct RowDeserializer<'a> {
    entries: Vec<(&'a str, ValueDeserializer)>,
}

impl RowDeserializer<'_> {
    fn into_first_value(self) -> Result<ValueDeserializer, RowError> {
        self.entries
            .into_iter()
            .next()
            .map(|(_, value)| value)
            .ok_or_else(|| RowError::Message("cannot deserialize an empty row".to_string()))
    }
}

impl<'de> Deserializer<'de> for RowDeserializer<'_> {
    type Error = RowError;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RowError> {
        self.into_first_value()?.deserialize_any(visitor)
    }

    fn deserialize_bool<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RowError> {
        self.into_first_value()?.deserialize_bool(visitor)
    }

    fn deserialize_f32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RowError> {
        self.into_first_value()?.deserialize_f32(visitor)
    }

    fn deserialize_f64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RowError> {
        self.into_first_value()?.deserialize_f64(visitor)
    }

    fn deserialize_byte_buf<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RowError> {
        self.into_first_value()?.deserialize_byte_buf(visitor)
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RowError> {
        self.into_first_value()?.deserialize_option(visitor)
    }

    fn deserialize_unit<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RowError> {
        self.into_first_value()?.deserialize_unit(visitor)
    }

    fn deserialize_unit_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        visitor: V,
    ) -> Result<V::Value, RowError> {
        self.into_first_value()?
            .deserialize_unit_struct(name, visitor)
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, RowError> {
        visitor.visit_newtype_struct(self.into_first_value()?)
    }

    fn deserialize_tuple<V: Visitor<'de>>(
        self,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value, RowError> {
        let values = self.entries.into_iter().map(|(_, value)| value);
        visitor.visit_seq(SeqDeserializer::new(values))
    }

    fn deserialize_map<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RowError> {
        visitor.visit_map(MapDeserializer::new(self.entries.into_iter()))
    }

    fn deserialize_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, RowError> {
        self.deserialize_map(visitor)
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        name: &'static str,
        variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, RowError> {
        self.into_first_value()?
            .deserialize_enum(name, variants, visitor)
    }

    forward_to_deserialize_any! {
        i8 i16 i32 i64 u8 u16 u32 u64 char str string bytes
        seq tuple_struct identifier ignored_any
    }
}

/// Deserializes a single column value.
struct ValueDeserializer(Value);

impl<'de> IntoDeserializer<'de, RowError> for ValueDeserializer {
    type Deserializer = Self;

    fn into_deserializer(self) -> Self {
        self
    }
}

impl<'de> Deserializer<'de> for ValueDeserializer {
    type Error = RowError;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RowError> {
        match self.0 {
            Value::Null => visitor.visit_none(),
            Value::Integer(value) => visitor.visit_i64(value),
            Value::Real(value) => visitor.visit_f64(value),
            Value::Text(value) => visitor.visit_string(value),
            // A `Vec<u8>` field derives as a sequence rather than as bytes, so a blob has to be
            // offered element by element for the common `column_bytes: Vec<u8>` case to work.
            // Fields tagged `#[serde(with = "serde_bytes")]` take the cheaper path below instead.
            Value::Blob(value) => visitor.visit_seq(value.into_deserializer()),
        }
    }

    /// SQLite has no boolean type; booleans arrive as integers.
    fn deserialize_bool<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RowError> {
        match self.0 {
            Value::Integer(value) => visitor.visit_bool(value != 0),
            Value::Real(value) => visitor.visit_bool(value != 0.0),
            _ => self.deserialize_any(visitor),
        }
    }

    /// A NULL read into a bare float becomes NaN, as it did under `serde_rusqlite`. Deserializing
    /// it into an `Option<f32>` instead yields `None`.
    fn deserialize_f32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RowError> {
        match self.0 {
            Value::Null => visitor.visit_f32(f32::NAN),
            _ => self.deserialize_any(visitor),
        }
    }

    /// See [`Self::deserialize_f32`].
    fn deserialize_f64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RowError> {
        match self.0 {
            Value::Null => visitor.visit_f64(f64::NAN),
            _ => self.deserialize_any(visitor),
        }
    }

    /// Hands a blob over as bytes, moving it instead of walking it element by element.
    fn deserialize_byte_buf<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RowError> {
        match self.0 {
            Value::Blob(value) => visitor.visit_byte_buf(value),
            _ => self.deserialize_any(visitor),
        }
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RowError> {
        match self.0 {
            Value::Null => visitor.visit_none(),
            _ => visitor.visit_some(self),
        }
    }

    fn deserialize_unit<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RowError> {
        match self.0 {
            Value::Null => visitor.visit_unit(),
            _ => self.deserialize_any(visitor),
        }
    }

    fn deserialize_unit_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        visitor: V,
    ) -> Result<V::Value, RowError> {
        match self.0 {
            Value::Text(value) if value == name => visitor.visit_unit(),
            _ => self.deserialize_any(visitor),
        }
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, RowError> {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, RowError> {
        match self.0 {
            Value::Text(value) => visitor.visit_enum(value.into_deserializer()),
            _ => self.deserialize_any(visitor),
        }
    }

    forward_to_deserialize_any! {
        i8 i16 i32 i64 u8 u16 u32 u64 char str string bytes
        seq tuple tuple_struct map struct identifier ignored_any
    }
}
