use std::marker::PhantomData;

use polars::datatypes::AnyValue;
use polars::frame::row::Row;
use polars::frame::DataFrame;
use serde::{
    de::{self, DeserializeOwned, Deserializer, SeqAccess, Visitor},
    forward_to_deserialize_any, Deserialize,
};

use crate::Dataset;

type Result<T> = core::result::Result<T, DataframeDatasetError>;

/// Error type for DataframeDataset
#[derive(thiserror::Error, Debug)]
pub enum DataframeDatasetError {
    /// Error occurred during deserialization
    #[error("{0}")]
    Other(String),
}

impl de::Error for DataframeDatasetError {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        DataframeDatasetError::Other(msg.to_string())
    }
}

/// Dataset implementation for Polars DataFrame
#[derive(Debug)]
pub struct DataframeDataset<I> {
    dataframe: DataFrame,
    len: usize,
    column_name_mapping: Vec<usize>,
    phantom: PhantomData<I>,
}

impl<I> DataframeDataset<I>
where
    I: Clone + Send + Sync + DeserializeOwned,
{
    // TODO from Lazy frame

    /// Create a new DataframeDataset from a Polars DataFrame
    pub fn from_dataframe(df: DataFrame) -> Result<Self> {
        let len = df.height();

        let field_names = extract_field_names::<I>();

        let column_name_mapping = field_names
            .iter()
            .map(|name| {
                df.schema()
                    .try_get_full(name)
                    .expect("Corresponding column should exist in the DataFrame")
                    .0
            })
            .collect::<Vec<_>>();

        Ok(DataframeDataset {
            dataframe: df,
            len,
            column_name_mapping,
            phantom: PhantomData,
        })
    }
}

impl<I> Dataset<I> for DataframeDataset<I>
where
    I: Clone + Send + Sync + DeserializeOwned,
{
    fn get(&self, index: usize) -> Option<I> {
        let row = self.dataframe.get_row(index).ok()?;

        let mut deserializer = RowDeserializer::new(&row, &self.column_name_mapping);
        I::deserialize(&mut deserializer).ok()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.dataframe.is_empty()
    }
}

struct RowDeserializer<'a> {
    row: &'a Row<'a>,
    column_name_mapping: &'a Vec<usize>,
    index: usize,
}

impl<'a> RowDeserializer<'a> {
    fn new(
        row: &'a Row, //schema: &'a Schema,
        column_name_mapping: &'a Vec<usize>,
    ) -> RowDeserializer<'a> {
        RowDeserializer {
            row,
            column_name_mapping,
            index: 0,
        }
    }
}

impl<'de, 'a> Deserializer<'de> for &'a mut RowDeserializer<'a> {
    type Error = DataframeDatasetError;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        let i = self.column_name_mapping[self.index];

        let value = &self.row.0[i];

        match value {
            AnyValue::Null => visitor.visit_none(),
            AnyValue::Boolean(b) => visitor.visit_bool(*b),
            AnyValue::Int32(i) => visitor.visit_i32(*i),
            AnyValue::Int64(i) => visitor.visit_i64(*i),
            AnyValue::UInt32(i) => visitor.visit_u32(*i),
            AnyValue::UInt64(i) => visitor.visit_u64(*i),
            AnyValue::Float32(f) => visitor.visit_f32(*f),
            AnyValue::Float64(f) => visitor.visit_f64(*f),
            AnyValue::String(s) => visitor.visit_string(s.to_string()),
            // TODO: implement other types
            _ => Err(DataframeDatasetError::Other("Unsupported type".to_string())),
        }
    }

    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_seq(self)
    }

    // Forward other methods to deserialize_any
    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string
        bytes byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map enum identifier ignored_any
    }
}

impl<'de, 'a> SeqAccess<'de> for RowDeserializer<'a> {
    type Error = DataframeDatasetError;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>>
    where
        T: de::DeserializeSeed<'de>,
    {
        if self.index >= self.row.0.len() {
            return Ok(None);
        }
        let mut deserializer = RowDeserializer {
            row: self.row,
            // schema: self.schema,
            column_name_mapping: self.column_name_mapping,
            index: self.index,
        };
        self.index += 1;
        seed.deserialize(&mut deserializer).map(Some)
    }
}

fn extract_field_names<'de, T>() -> Vec<&'static str>
where
    T: Deserialize<'de>,
{
    struct FieldExtractor {
        fields: Vec<&'static str>,
    }

    impl<'de> Deserializer<'de> for &mut FieldExtractor {
        type Error = de::value::Error;

        fn deserialize_any<V>(self, _visitor: V) -> core::result::Result<V::Value, Self::Error>
        where
            V: Visitor<'de>,
        {
            Err(de::Error::custom("Field extractor"))
        }

        fn deserialize_struct<V>(
            self,
            _name: &'static str,
            fields: &'static [&'static str],
            _visitor: V,
        ) -> core::result::Result<V::Value, Self::Error>
        where
            V: Visitor<'de>,
        {
            self.fields.extend_from_slice(fields);
            Err(de::Error::custom("Field extractor"))
        }

        serde::forward_to_deserialize_any! {
            bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes
            byte_buf option unit unit_struct newtype_struct seq tuple
            tuple_struct map enum identifier ignored_any
        }
    }

    let mut extractor = FieldExtractor { fields: Vec::new() };
    let _ = T::deserialize(&mut extractor);
    extractor.fields
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;
    use serde::Deserialize;

    #[derive(Clone, Debug, Deserialize, PartialEq)]
    struct TestData {
        a: i32,
        b: bool,
        c: f64,
    }

    fn create_test_dataframe() -> DataFrame {
        let s0 = Series::new("a", &[1, 2, 3]);
        let s1 = Series::new("b", &[true, false, true]);
        let s2 = Series::new("c", &[1.1, 2.2, 3.3]);
        let s3 = Series::new("d", &["Boo", "Boo2", "Boo3"]);
        DataFrame::new(vec![s1, s2, s0, s3]).unwrap()
    }

    #[test]
    fn test_dataframe_dataset() {
        let df = create_test_dataframe();
        let dataset: DataframeDataset<TestData> = DataframeDataset::from_dataframe(df).unwrap();

        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.is_empty(), false);

        let item = dataset.get(1).unwrap();
        assert_eq!(
            item,
            TestData {
                a: 2,
                b: false,
                c: 2.2
            }
        );

        let item = dataset.get(2).unwrap();
        assert_eq!(
            item,
            TestData {
                a: 3,
                b: true,
                c: 3.3
            }
        );
    }

    // TODO test non existing struct fields
}
