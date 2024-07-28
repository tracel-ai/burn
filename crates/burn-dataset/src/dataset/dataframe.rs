use std::marker::PhantomData;

use polars::frame::row::Row;
use polars::prelude::*;
use serde::{
    de::{self, DeserializeOwned, Deserializer, SeqAccess, Visitor},
    Deserialize, forward_to_deserialize_any,
};
use serde::de::DeserializeSeed;

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

/// Enum to hold either a DataFrame or a LazyFrame
pub enum Frame {
    /// Represents a Polars Dataframe
    Data(DataFrame),
    /// Represents a Polars LazyFrame
    Lazy(LazyFrame),
}

/// Dataset implementation for Polars Frame (DataFrame or LazyFrame)
pub struct DataframeDataset<I> {
    frame: Frame,
    len: usize,
    column_name_mapping: Vec<usize>,
    phantom: PhantomData<I>,
}

impl<I> DataframeDataset<I>
where
    I: Clone + Send + Sync + DeserializeOwned,
{
    /// Create a new DataframeDataset from a Polars DataFrame or LazyFrame
    pub fn new(df: Frame) -> Result<Self> {
        match df {
            Frame::Data(data) => {
                let len = data.height();
                let field_names = extract_field_names::<I>();

                let column_name_mapping = field_names
                    .iter()
                    .map(|name| {
                        data.schema()
                            .try_get_full(name)
                            .expect("Corresponding column should exist in the DataFrame")
                            .0
                    })
                    .collect::<Vec<_>>();

                Ok(DataframeDataset {
                    frame: Frame::Data(data),
                    len,
                    column_name_mapping,
                    phantom: PhantomData,
                })
            },
            Frame::Lazy(lazy) => {
                let schema = lazy
                    .clone()
                    .schema()
                    .map_err(
                        |e| DataframeDatasetError::Other(e.to_string()))?;
                // LazyFrame doesn't know the length upfront so set to zero initially 
                let len = 0;
                let field_names = extract_field_names::<I>();

                let column_name_mapping = field_names
                    .iter()
                    .filter_map(|name| {
                        schema
                            .iter()
                            .position(|(col_name, _)| col_name == name)
                    })
                    .collect::<Vec<_>>();
                Ok(DataframeDataset {
                    frame: Frame::Lazy(lazy),
                    len,
                    column_name_mapping,
                    phantom: PhantomData,
                })
            },
        }
    }

    /// Create a new DataframeDataset from a Polars LazyFrame
    pub fn from_lazyframe(lf: LazyFrame) -> Result<Self> {
        Self::new(Frame::Lazy(lf))
    }
    /// Create a new DataframeDataset from a Polars Dataframe
    pub fn from_dataframe(df: DataFrame) -> Result<Self> {
        Self::new(Frame::Data(df))
    }

    /// Collect the LazyFrame into a DataFrame and get the length
    pub fn collect_and_get_len(&mut self) -> Result<usize> {
        if let Frame::Lazy(lazy) = &self.frame {
            let df = lazy.clone().collect().map_err(
                |e|
                DataframeDatasetError::Other(e.to_string())
            )?;
            self.len = df.height();
            self.frame = Frame::Data(df);
        }
        Ok(self.len)
    }
}

impl<I> Dataset<I> for DataframeDataset<I>
where
    I: Clone + Send + Sync + DeserializeOwned,
{
    fn get(&self, index: usize) -> Option<I> {
        if let Frame::Data(ref df) = self.frame {
            let row = df.get_row(index).ok()?;

            let mut deserializer = RowDeserializer::new(&row, &self.column_name_mapping);
            I::deserialize(&mut deserializer).ok()
        } else {
            None // Or handle LazyFrame differently if needed
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool { self.len == 0 }
}

struct RowDeserializer<'a> {
    row: &'a Row<'a>,
    column_name_mapping: &'a Vec<usize>,
    index: usize,
}

impl<'a> RowDeserializer<'a> {
    fn new(row: &'a Row, column_name_mapping: &'a Vec<usize>) -> RowDeserializer<'a> {
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
            AnyValue::Int8(i) => visitor.visit_i8(*i),
            AnyValue::Int16(i) => visitor.visit_i16(*i),
            AnyValue::Int32(i) => visitor.visit_i32(*i),
            AnyValue::Int64(i) => visitor.visit_i64(*i),
            AnyValue::UInt8(i) => visitor.visit_u8(*i),
            AnyValue::UInt16(i) => visitor.visit_u16(*i),
            AnyValue::UInt32(i) => visitor.visit_u32(*i),
            AnyValue::UInt64(i) => visitor.visit_u64(*i),
            AnyValue::Float32(f) => visitor.visit_f32(*f),
            AnyValue::Float64(f) => visitor.visit_f64(*f),
            AnyValue::Date(i) => visitor.visit_i32(*i),
            AnyValue::String(s) => visitor.visit_string(s.to_string()),
            AnyValue::Binary(b) => {
                let v: Vec<usize> = b.iter().map(|&x| x as usize).collect();
                visitor.visit_seq(de::value::SeqDeserializer::new(v.into_iter()))
            },
            // TODO: add support for complex types
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
        T: DeserializeSeed<'de>,
    {
        if self.index >= self.row.0.len() {
            return Ok(None);
        }
        let mut deserializer = RowDeserializer {
            row: self.row,
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

        forward_to_deserialize_any! {
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
    use polars::prelude::*;
    use serde::Deserialize;

    use super::*;

    #[derive(Clone, Debug, Deserialize, PartialEq)]
    struct TestData {
        a: i32,
        b: bool,
        c: f64,
        d: String,
        e: Vec<i32>
    }

    fn create_test_dataframe() -> DataFrame {
        let s0 = Series::new("a", &[1, 2, 3, ]);
        let s1 = Series::new("b", &[true, false, true]);
        let s2 = Series::new("c", &[1.1, 2.2, 3.3]);
        let s3 = Series::new("d", &["Boo", "Boo2", "Boo3"]);
        let s4 = Series::new("e", &[
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9]
        ]);
        DataFrame::new(vec![s1, s2, s0, s3, s4]).unwrap()
    }

    fn create_test_lazyframe() -> LazyFrame {
        let s0 = Series::new("a", &[1, 2, 3, ]);
        let s1 = Series::new("b", &[true, false, true]);
        let s2 = Series::new("c", &[1.1, 2.2, 3.3]);
        let s3 = Series::new("d", &["Boo", "Boo2", "Boo3"]);
        let s4 = Series::new("e", &[
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9]
        ]);
        DataFrame::new(vec![s1, s2, s0, s3, s4]).unwrap().lazy()
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
                c: 2.2,
                d: "Boo2".to_string(),
                e: vec![4, 5, 6]
            }
        );

        let item = dataset.get(2).unwrap();

        assert_eq!(
            item,
            TestData {
                a: 3,
                b: true,
                c: 3.3,
                d: "Boo3".to_string(),
                e: vec![7, 8, 9]
            }
        );
    }

    #[test]
    fn test_lazyframe_dataset() {
        let df = create_test_lazyframe();
        let mut dataset: DataframeDataset<TestData> = DataframeDataset::from_lazyframe(df).unwrap();
        dataset.collect_and_get_len().unwrap();

        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.is_empty(), false);

        let item = dataset.get(1).unwrap();
        assert_eq!(
            item,
            TestData {
                a: 2,
                b: false,
                c: 2.2,
                d: "Boo2".to_string(),
                e: vec![4, 5, 6],
            }
        );

        let item = dataset.get(2).unwrap();

        assert_eq!(
            item,
            TestData {
                a: 3,
                b: true,
                c: 3.3,
                d: "Boo3".to_string(),
                e: vec![7, 8, 9],
            }
        );
    }

    // TODO test non existing struct fields
}
