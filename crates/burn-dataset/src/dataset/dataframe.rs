use std::io;
use std::marker::PhantomData;

use polars::frame::DataFrame;
use polars::frame::row::Row;
use polars::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, to_string, Value};
use serde_json::value::Serializer;

use crate::Dataset;

pub type Result<T> = core::result::Result<T, DataframeDatasetError>;

/// Custom error type for DataframeDataset type
#[derive(thiserror::Error, Debug)]
pub enum DataframeDatasetError {
    /// IO related error.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    /// Serde related error.
    #[error("Serde error: {0}")]
    Serde(#[from] rmp_serde::encode::Error),
    /// Polars related error
    #[error("Polars error: {0}")]
    PolarsError(#[from] PolarsError),
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
    /// Utility method to check if a row can indeed be serialised
    fn check_if_row_is_serializable(row: &Row, schema: &Schema) -> Result<bool> {
        match Self::row_to_serde_value(row, schema) {
            Ok(s) => {
                if !s.is_null() { Ok(true) } else { Ok(false) }
            },
            Err(e) => Err(e)
        }
    }

    fn anyvalue_list_to_json(series: &Series) -> Value {
        let json_array: Vec<Value> = series
            .iter()
            .map(|av| match av {
                AnyValue::Null => Value::Null,
                AnyValue::Boolean(b) => json!(b),
                AnyValue::String(s) => json!(s),
                AnyValue::Int32(i) => json!(i),
                AnyValue::Int64(i) => json!(i),
                AnyValue::UInt32(i) => json!(i),
                AnyValue::UInt64(i) => json!(i),
                AnyValue::Float32(f) => json!(f),
                AnyValue::Float64(f) => json!(f),
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
                AnyValue::Null => json!(null),
                AnyValue::Boolean(v) => json!(v),
                AnyValue::String(v) => json!(v),
                AnyValue::Int32(v) => json!(v),
                AnyValue::Int64(v) => json!(v),
                AnyValue::UInt32(v) => json!(v),
                AnyValue::UInt64(v) => json!(v),
                AnyValue::Float32(v) => json!(v),
                AnyValue::Float64(v) => json!(v),
                AnyValue::Date(v) => json!(v),
                AnyValue::Binary(v) => json!(v),
                AnyValue::List(v) => Self::anyvalue_list_to_json(v),
                _ => json!(null),
            };
            obj.insert(field.name().to_string(), value);
        }
        Ok(Value::Object(obj))
    }

    fn row_to_serde_value_native(row: &Row, schema: &Schema) -> Result<Value> {
        let mut obj = serde_json::Map::new();
        for (field, any_value) in schema.iter_fields().zip(row.0.iter()) {
            if let Ok(value) = any_value.serialize(Serializer) {
                obj.insert(field.name().to_string(), value);
            }
        }
        Ok(Value::Object(obj))
    }
}

impl<I> Dataset<I> for DataframeDataset<I>
where
    I: Clone + Send + Sync + DeserializeOwned,
{
    /// This method will return the row by its index
    fn get(&self, index: usize) -> Option<I> {
        match self.dataframe.get_row(index) { 
            Ok(row) => {
                let schema = self.dataframe.schema();
                if let Ok(serialized_row) = Self::row_to_serde_value_native(&row, &schema) {
                    println!("Serialized Row: {}", serialized_row);
                    if let Some(serde_map) = serialized_row.as_object() {
                        println!("Serde Map: {:?}", serde_map);
                        if let Ok(json_str) = to_string(serde_map) {
                            println!("JSON String: {:?}", json_str);
                            return match serde_json::from_str::<I>(json_str.as_str()) {
                                Ok(row) => Some(row),
                                Err(e) => None
                            }
                        } else { None }
                    } else { None }
                } else { None }
                // match Self::check_if_row_is_serializable(&row, &schema) {
                //     Ok(check) => {
                //         match check {
                //             true => {
                //                 match Self::row_to_serde_value(&row, &schema) {
                //                     Ok(serialized_row) => {
                //                         println!("Serialized Row: {}", serialized_row);
                //                         if let Some(serde_map) = serialized_row.as_object() {
                //                             println!("Serde Map: {:?}", serde_map);
                //                             if let Ok(json_str) = to_string(serde_map) {
                //                                 println!("JSON String: {:?}", json_str);
                //                                 return match serde_json::from_str::<I>(&json_str) {
                //                                     Ok(deserialized_row) => Some(deserialized_row),
                //                                     Err(e) => {
                //                                         println!("An error occurred while \
                //                                         deserializing string into struct: {}", e);
                //                                         None
                //                                     }
                //                                 }
                //                             } else { None }
                //                         } else { None }
                //                     },
                //                     Err(e) => {
                //                         println!("An error occurred while serializing Polars dataframe row. Error: {:?}", e);
                //                         None
                //                     }
                //                 }
                //             },
                //             false => { None }
                //         }
                //     },
                //     Err(e) => {
                //         println!("An error occurred while checking if a Polars row is serializable: Error: {:?}", e);
                //         None
                //     }
                // }
            },
            Err(e) =>{
                println!("An error occurred while getting row from polars dataframe. Error: {}", e);
                None
            }
        }
    }
    fn len(&self) -> usize {self.len}
    fn is_empty(&self) -> bool {self.dataframe.is_empty()}
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};
    use serde::{Deserialize, Serialize};

    use crate::Dataset;

    use super::*;

    type DataframeSample = DataframeDataset<SampleDf>;

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
    fn train_dataframe() -> DataframeSample {
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
    fn len(train_dataframe: DataframeSample) {
        assert_eq!(train_dataframe.len(), 3);
    }

    #[rstest]
    fn get(train_dataframe: DataframeSample) {
        let item = train_dataframe.get(0).unwrap();
        assert_eq!(item.bool_column, true);
        assert_eq!(item.int32_column, 1);
        assert_eq!(item.int64_column, 10i64);
        assert_eq!(item.utf8_column, "a".to_string());
        assert_eq!(item.float32_column, 1.1f32);
    }

    #[rstest]
    fn get_none(train_dataframe: DataframeSample) {
        assert_eq!(train_dataframe.get(10), None);
    }
}
