use std::marker::PhantomData;

use polars::datatypes::AnyValue;
use polars::frame::DataFrame;
use polars::frame::row::Row;
use polars::prelude::Schema;
use polars::series::Series;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{to_string, Value};
use serde_json::value::Serializer;

use crate::Dataset;

pub type Result<T> = core::result::Result<T, DataframeDatasetError>;

/// Custom error type for DataframeDataset type
#[derive(thiserror::Error, Debug)]
pub enum DataframeDatasetError {
    /// Any other error.
    #[error("{0}")]
    Other(&'static str),
}

impl From<&'static str> for DataframeDatasetError {
    fn from(s: &'static str) -> Self {
        DataframeDatasetError::Other(s)
    }
}
/// A struct that is used to hold a polars dataframe.
/// each instance of this struct will represent a single polars dataframe.
/// this struct provides basic Dataset functionality to a polars dataframe, so that they are able
/// to natively be used for training/validation.
/// Currently there is no native way to deserialize a polars row, so a trial and error approach
/// has been taken. This could very well change in the future.
#[derive(Debug)]
pub struct DataframeDataset<I> {
    dataframe: DataFrame,
    len: usize,
    phantom: PhantomData<I>,
}

impl<I> DataframeDataset<I> {
    /// Initialise a DataframeDataset from a polars dataframe
    pub fn from_dataframe(df: DataFrame) -> Result<Self> {
        let len = df.height();
        Ok(DataframeDataset {
            dataframe: df,
            len,
            phantom: PhantomData,
        })
    }
    fn anyvalue_list_to_json(series: &Series) -> Value {
        let json_array: Vec<Value> = series
            .iter()
            .map(|av| match av {
                AnyValue::Null => Value::Null,
                AnyValue::Boolean(b) => b.serialize(Serializer).unwrap(),
                AnyValue::String(s) => s.serialize(Serializer).unwrap(),
                AnyValue::Int32(i) => i.serialize(Serializer).unwrap(),
                AnyValue::Int64(i) => i.serialize(Serializer).unwrap(),
                AnyValue::UInt32(i) => i.serialize(Serializer).unwrap(),
                AnyValue::UInt64(i) => i.serialize(Serializer).unwrap(),
                AnyValue::Float32(f) => f.serialize(Serializer).unwrap(),
                AnyValue::Float64(f) => f.serialize(Serializer).unwrap(),
                AnyValue::List(inner_series) => Self::anyvalue_list_to_json(&inner_series), // Recursive call for nested lists
                _ => panic!("Unsupported AnyValue type"),
            })
            .collect();
        Value::Array(json_array)
    }

    fn row_to_serde_value(row: &Row, schema: &Schema) -> Result<Value> {
        let mut obj = serde_json::Map::new();
        for (field, any_value) in schema.iter_fields().zip(row.0.iter()) {
            let value = match any_value {
                AnyValue::Null => Value::Null,
                AnyValue::Boolean(v) => v.serialize(Serializer).unwrap(),
                AnyValue::String(v) => v.serialize(Serializer).unwrap(),
                AnyValue::Int32(v) => v.serialize(Serializer).unwrap(),
                AnyValue::Int64(v) => v.serialize(Serializer).unwrap(),
                AnyValue::UInt32(v) => v.serialize(Serializer).unwrap(),
                AnyValue::UInt64(v) => v.serialize(Serializer).unwrap(),
                AnyValue::Float32(v) => v.serialize(Serializer).unwrap(),
                AnyValue::Float64(v) => v.serialize(Serializer).unwrap(),
                AnyValue::Date(v) => v.serialize(Serializer).unwrap(),
                AnyValue::Binary(v) => v.serialize(Serializer).unwrap(),
                AnyValue::List(v) => Self::anyvalue_list_to_json(v),
                _ => Value::Null,
            };
            obj.insert(field.name().to_string(), value);
        }
        Ok(Value::Object(obj))
    }
}

impl<I> Dataset<I> for DataframeDataset<I>
where
    I: Clone + Send + Sync + DeserializeOwned,
{
    /// This method will return the row by its index in the format of the struct I
    fn get(&self, index: usize) -> Option<I> {
        let row = self.dataframe.get_row(index).unwrap();
        let schema = self.dataframe.schema();

        let serialized_row = Self::row_to_serde_value(&row, &schema).unwrap();
        let serde_map = serialized_row.as_object().unwrap();
        let json_str = to_string(serde_map).unwrap();
        let deserialized_row = serde_json::from_str::<I>(&json_str).unwrap();
        Some(deserialized_row)
    }
    fn len(&self) -> usize {self.len}
    fn is_empty(&self) -> bool {self.dataframe.is_empty()}
}

#[cfg(test)]
mod tests {
    use polars::datatypes::DataType;
    use polars::prelude::NamedFrom;
    use rstest::{fixture, rstest};
    use serde::{Deserialize, Serialize};

    use crate::Dataset;

    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct SampleDf {
        bool_column: bool,
        int32_column: i32,
        int64_column: i64,
        uint32_column: u32,
        uint64_column: u64,
        float32_column: f32,
        float64_column: f64,
        utf8_column: String,
    }

    #[fixture]
    fn ds() -> DataframeDataset<SampleDf> {
        let s_bool = Series::new("bool_column", &[true, false, true]);
        let s_int32 = Series::new("int32_column", &[1, 2, 3]);
        let s_int64 = Series::new("int64_column", &[10i64, 20, 30]);
        let s_uint32 = Series::new("uint32_column", &[100u32, 200, 300]);
        let s_uint64 = Series::new("uint64_column", &[1000u64, 2000, 3000]);
        let s_float32 = Series::new("float32_column", &[1.1f32, 2.2, 3.3]);
        let s_float64 = Series::new("float64_column", &[10.1f64, 20.2, 30.3]);
        let s_utf8 = Series::new("utf8_column", &["a", "b", "c"]);
        let s_date = Series::new("date_column", &[1, 2, 3]).cast(&DataType::Date).unwrap();

        let df = DataFrame::new(vec![
            s_bool, s_int32, s_int64, s_uint32, s_uint64,
            s_float32, s_float64, s_utf8, s_date,
        ]).unwrap();
        DataframeDataset::<SampleDf>::from_dataframe(df).unwrap()
    }

    #[rstest]
    fn len(ds: DataframeDataset<SampleDf>) {
        assert_eq!(ds.len(), 3);
    }

    #[rstest]
    fn get(ds: DataframeDataset<SampleDf>) {
        let item = ds.get(0).unwrap();
        assert_eq!(item.bool_column, true);
        assert_eq!(item.int32_column, 1);
        assert_eq!(item.int64_column, 10i64);
        assert_eq!(item.utf8_column, "a".to_string());
        assert_eq!(item.float32_column, 1.1f32);
    }

    #[rstest]
    fn get_none(ds: DataframeDataset<SampleDf>) {
        assert_eq!(ds.get(10), None);
    }
}
