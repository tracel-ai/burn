use std::io;
use std::marker::PhantomData;
use polars::frame::DataFrame;
use polars::frame::row::{Row};
use polars::prelude::*;
use serde::{de::DeserializeOwned};
use serde_json::{json, Value};
use crate::Dataset;
pub type Result<T> = core::result::Result<T, DataframeDatasetError>;


/// Custom error type for `DataframeDataset` type
#[derive(thiserror::Error, Debug)]
pub enum DataframeDatasetError{
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

/// A struct that is used to hold a polars dataframe.
/// each instance of this struct will represent a single polars dataframe.
/// this struct provides basic Dataset functionality to a polars dataframe, so that they are able
/// to natively be used for training/validation.
/// Currently there is no native way to deserialize a polars row, so a trial and error approach 
/// has been taken. This could very well change in the future.
#[derive(Debug)]
pub struct DataframeDataset<I>{
    dataframe: DataFrame,
    len: usize,
    phantom: PhantomData<I>
}

impl<I> DataframeDataset<I>{
    /// Initialise a `DataframeDataset` from a polars dataframe
    pub fn from_dataframe(df: DataFrame) -> Result<Self>{
        let len = df.height();
        Ok(DataframeDataset{
            dataframe: df,
            len,
            phantom: PhantomData
        })


    }
    /// Utility method to check if a row can indeed be serialised
    fn check_if_row_is_serializable(row: &Row, schema: &Schema) -> Result<bool>{
        match Self::row_to_serde_value(row, schema){
            Ok(s) => {
                if !s.is_null(){Ok(true) } else { Ok(false) }
            },
            Err(e) => Err(e)
        }
    }
    /// Utility method to serialize rows from dataframe
    fn row_to_serde_value(row: &Row, schema: &Schema) -> Result<Value>{
        let mut obj = serde_json::Map::new();
        for (field, any_value) in schema.iter_fields().zip(row.0.iter()) {
            let value = match any_value {
                AnyValue::Null => json!(null),
                AnyValue::Boolean(v) => json!(v),
                AnyValue::String(v) => json!(v),
                AnyValue::Int8(v) => json!(v),
                AnyValue::Int16(v) => json!(v),
                AnyValue::Int32(v) => json!(v),
                AnyValue::Int64(v) => json!(v),
                AnyValue::UInt8(v) => json!(v),
                AnyValue::UInt16(v) => json!(v),
                AnyValue::UInt32(v) => json!(v),
                AnyValue::UInt64(v) => json!(v),
                AnyValue::Float32(v) => json!(v),
                AnyValue::Float64(v) => json!(v),
                AnyValue::Date(v) => json!(v),
                AnyValue::Binary(v) => json!(v),
                // Handle other types as needed
                _ => json!(null),
            };
            obj.insert(field.name().to_string(), value);
        }
        Ok(Value::Object(obj))
}
}

impl <I> Dataset<I> for DataframeDataset<I> 
where 
    I: Clone + Send + Sync + DeserializeOwned, 
{
    /// This method will return the row by index 
    fn get(&self, index: usize) -> Option<I> {
        let row = self.dataframe.get_row(index).unwrap();
        let schema = self.dataframe.schema();
        match Self::check_if_row_is_serializable(&row, &schema){
            Ok(check) => {
                match check {
                    true => {
                        match Self::row_to_serde_value(&row, &schema){
                            Ok(serialised_row) => {
                                let deserialised_row: String = serde_json::from_value(serialised_row).unwrap();
                                let slice = deserialised_row.as_bytes();
                                Some(rmp_serde::from_slice(slice).unwrap())
                            },
                            Err(e) => {
                                println!("An error occurred while serializing Polars dataframe row\
                                . Error: {:?}", e);
                                None
                            }
                        }
                    },
                    false => { None }
                }
            },
            Err(e) => {
                println!("An error occurred while check if a Polars row is serializable: Error: \
                {:?}", e);
                None
            }
        }
    }
    /// This method will return the number of rows in a dataframe
    fn len(&self) -> usize {self.len}
    /// This method will return a bool if the dataframe is empty or now
    fn is_empty(&self) -> bool {self.dataframe.is_empty()}
}

#[cfg(test)]
mod tests{
    use polars::prelude::*;
    use rstest::fixture;
    use crate::DataframeDataset;
    
    type DataframeSample = DataframeDataset<SampleDf>;
    pub struct SampleDf {
        bool_column: bool,
        int32_column: i32,
        int64_column: i64 ,
        uint32_column: u32,
        uint64_column: u64,
        float32_column: f32,
        float64_column: f64,
        utf8_column: Vec<u8>,
    }
    
    #[fixture]
    fn train_dataframe() -> DataframeSample{
        let s_bool = Series::new("bool_column", &[true, false, true]);
        let s_int32 = Series::new("int32_column", &[1, 2, 3]);
        let s_int64 = Series::new("int64_column", &[10i64, 20, 30]);
        let s_uint32 = Series::new("uint32_column", &[100u32, 200, 300]);
        let s_uint64 = Series::new("uint64_column", &[1000u64, 2000, 3000]);
        let s_float32 = Series::new("float32_column", &[1.1f32, 2.2, 3.3]);
        let s_float64 = Series::new("float64_column", &[10.1f64, 20.2, 30.3]);
        let s_utf8 = Series::new("utf8_column", &["a", "b", "c"]);
        let s_date = Series::new("date_column", &[1, 2, 3]).cast(&DataType::Date).unwrap();
    
        // Create DataFrame
        let df = DataFrame::new(vec![
            s_bool, s_int32, s_int64, s_uint32, s_uint64,
            s_float32, s_float64, s_utf8, s_date,
        ]).unwrap();
        DataframeDataset::<SampleDf>::from_dataframe(df).unwrap()
    }
}
