use std::marker::PhantomData;

use crate::Dataset;

use polars::frame::row::Row;
use polars::prelude::*;
use serde::de::DeserializeSeed;
use serde::{
    Deserialize,
    de::{self, DeserializeOwned, Deserializer, SeqAccess, Visitor},
    forward_to_deserialize_any,
};

/// Error type for DataframeDataset
#[derive(thiserror::Error, Debug)]
pub enum DataframeDatasetError {
    /// Error occurred during deserialization or other operations
    #[error("{0}")]
    Other(String),
}

impl de::Error for DataframeDatasetError {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        DataframeDatasetError::Other(msg.to_string())
    }
}

/// Dataset implementation for Polars DataFrame
///
/// This struct provides a way to access data from a Polars DataFrame
/// as if it were a Dataset of type I.
pub struct DataframeDataset<I> {
    df: DataFrame,
    len: usize,
    column_name_mapping: Vec<usize>,
    phantom: PhantomData<I>,
}

impl<I> DataframeDataset<I>
where
    I: Clone + Send + Sync + DeserializeOwned,
{
    /// Create a new DataframeDataset from a Polars DataFrame
    ///
    /// # Arguments
    ///
    /// * `df` - A Polars DataFrame
    ///
    /// # Returns
    ///
    /// A Result containing the new DataframeDataset or a DataframeDatasetError
    pub fn new(df: DataFrame) -> Result<Self, DataframeDatasetError> {
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
            df,
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
    /// Get an item from the dataset at the specified index
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the item to retrieve
    ///
    /// # Returns
    ///
    /// An Option containing the item if it exists, or None if it doesn't
    fn get(&self, index: usize) -> Option<I> {
        let row = self.df.get_row(index).ok()?;

        let mut deserializer = RowDeserializer::new(&row, &self.column_name_mapping);
        I::deserialize(&mut deserializer).ok()
    }

    /// Get the length of the dataset
    fn len(&self) -> usize {
        self.len
    }

    /// Check if the dataset is empty
    fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A deserializer for Polars DataFrame rows
struct RowDeserializer<'a> {
    row: &'a Row<'a>,
    column_name_mapping: &'a Vec<usize>,
    index: usize,
}

impl<'a> RowDeserializer<'a> {
    /// Create a new RowDeserializer
    ///
    /// # Arguments
    ///
    /// * `row` - A reference to a Polars DataFrame row
    /// * `column_name_mapping` - A reference to a vector mapping field names to column indices
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

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, DataframeDatasetError>
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
                visitor.visit_seq(de::value::SeqDeserializer::new(b.iter().copied()))
            }
            AnyValue::Time(t) => visitor.visit_i64(*t),
            ty => Err(DataframeDatasetError::Other(
                format!("Unsupported type: {ty:?}").to_string(),
            )),
        }
    }

    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, DataframeDatasetError>
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

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, DataframeDatasetError>
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

/// Extract field names from a type T that implements Deserialize
///
/// # Returns
///
/// A vector of field names as static string slices
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
        int32: i32,
        bool: bool,
        float64: f64,
        string: String,
        int16: i16,
        uint32: u32,
        uint64: u64,
        float32: f32,
        int64: i64,
        int8: i8,
        binary: Vec<u8>,
    }

    fn create_test_dataframe() -> DataFrame {
        let s0 = Column::new("int32".into(), &[1i32, 2i32, 3i32]);
        let s1 = Column::new("bool".into(), &[true, false, true]);
        let s2 = Column::new("float64".into(), &[1.1f64, 2.2f64, 3.3f64]);
        let s3 = Column::new("string".into(), &["Boo", "Boo2", "Boo3"]);
        let s6 = Column::new("int16".into(), &[1i16, 2i16, 3i16]);
        let s8 = Column::new("uint32".into(), &[1u32, 2u32, 3u32]);
        let s9 = Column::new("uint64".into(), &[1u64, 2u64, 3u64]);
        let s10 = Column::new("float32".into(), &[1.1f32, 2.2f32, 3.3f32]);
        let s11 = Column::new("int64".into(), &[1i64, 2i64, 3i64]);
        let s12 = Column::new("int8".into(), &[1i8, 2i8, 3i8]);

        let binary_data: Vec<&[u8]> = vec![&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]];

        let s13 = Column::new("binary".into(), binary_data);
        DataFrame::new(vec![s0, s1, s2, s3, s6, s8, s9, s10, s11, s12, s13]).unwrap()
    }

    #[test]
    fn test_dataframe_dataset_creation() {
        let df = create_test_dataframe();
        let dataset = DataframeDataset::<TestData>::new(df);
        assert!(dataset.is_ok());
    }

    #[test]
    fn test_dataframe_dataset_length() {
        let df = create_test_dataframe();
        let dataset = DataframeDataset::<TestData>::new(df).unwrap();
        assert_eq!(dataset.len(), 3);
        assert!(!dataset.is_empty());
    }

    #[test]
    fn test_dataframe_dataset_get() {
        let df = create_test_dataframe();
        let dataset = DataframeDataset::<TestData>::new(df).unwrap();

        let expected_items = vec![
            TestData {
                int32: 1,
                bool: true,
                float64: 1.1,
                string: "Boo".to_string(),
                int16: 1,
                uint32: 1,
                uint64: 1,
                float32: 1.1,
                int64: 1,
                int8: 1,
                binary: vec![1, 2, 3],
            },
            TestData {
                int32: 2,
                bool: false,
                float64: 2.2,
                string: "Boo2".to_string(),
                int16: 2,
                uint32: 2,
                uint64: 2,
                float32: 2.2,
                int64: 2,
                int8: 2,
                binary: vec![4, 5, 6],
            },
            TestData {
                int32: 3,
                bool: true,
                float64: 3.3,
                string: "Boo3".to_string(),
                int16: 3,
                uint32: 3,
                uint64: 3,
                float32: 3.3,
                int64: 3,
                int8: 3,
                binary: vec![7, 8, 9],
            },
        ];

        for (index, expected_item) in expected_items.iter().enumerate() {
            let item = dataset.get(index).unwrap();
            assert_eq!(&item, expected_item);
        }
    }

    #[test]
    fn test_dataframe_dataset_out_of_bounds() {
        let df = create_test_dataframe();
        let dataset = DataframeDataset::<TestData>::new(df).unwrap();
        assert!(dataset.get(3).is_none());
    }

    #[test]
    fn test_dataframe_dataset() {
        let df = create_test_dataframe();
        let dataset: DataframeDataset<TestData> = DataframeDataset::new(df).unwrap();

        assert_eq!(dataset.len(), 3);
        assert!(!dataset.is_empty());

        let item = dataset.get(1).unwrap();
        assert_eq!(
            item,
            TestData {
                int32: 2,
                bool: false,
                float64: 2.2,
                string: "Boo2".to_string(),
                int16: 2,
                uint32: 2,
                uint64: 2,
                float32: 2.2,
                int64: 2,
                int8: 2,
                binary: vec![4, 5, 6],
            }
        );

        let item = dataset.get(2).unwrap();

        assert_eq!(
            item,
            TestData {
                int32: 3,
                bool: true,
                float64: 3.3,
                string: "Boo3".to_string(),
                int16: 3,
                uint32: 3,
                uint64: 3,
                float32: 3.3,
                int64: 3,
                int8: 3,
                binary: vec![7, 8, 9],
            }
        );
    }

    #[test]
    #[should_panic = "Corresponding column should exist in the DataFrame: SchemaFieldNotFound(ErrString(\"non_existent\"))"]
    fn test_non_existing_struct_fields() {
        #[derive(Clone, Debug, Deserialize, PartialEq)]
        struct PartialTestData {
            int32: i32,
            bool: bool,
            non_existent: String,
        }

        let df = create_test_dataframe();
        let dataset = DataframeDataset::<PartialTestData>::new(df);

        assert!(dataset.is_err());
        if let Err(e) = dataset {
            assert!(matches!(e, DataframeDatasetError::Other(_)));
        }
    }

    #[test]
    fn test_partial_table() {
        #[derive(Clone, Debug, Deserialize, PartialEq)]
        struct PartialTestData {
            int32: i32,
            bool: bool,
            string: String,
        }

        let df = create_test_dataframe();
        let dataset = DataframeDataset::<PartialTestData>::new(df).unwrap();

        assert_eq!(dataset.len(), 3);
        assert!(!dataset.is_empty());

        let item = dataset.get(1).unwrap();
        assert_eq!(
            item,
            PartialTestData {
                int32: 2,
                bool: false,
                string: "Boo2".to_string(),
            }
        );

        let item = dataset.get(2).unwrap();
        assert_eq!(
            item,
            PartialTestData {
                int32: 3,
                bool: true,
                string: "Boo3".to_string(),
            }
        );
    }
}
