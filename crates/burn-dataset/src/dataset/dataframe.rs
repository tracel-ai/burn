use std::io;
use std::marker::PhantomData;
use polars::frame::DataFrame;
use polars::frame::row::{Row};
use polars::prelude::*;
use serde::{de::DeserializeOwned};
use serde_json::{json, Value};
use crate::{Dataset};
pub type Result<T> = core::result::Result<T, DataframeDatasetError>;
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
    PolarsError(#[from] polars::error::PolarsError),
    /// Any other error.
    #[error("{0}")]
    Other(&'static str),
}

pub struct DataframeDataset<I>{
    dataframe: DataFrame,
    len: usize,
    phantom: PhantomData<I>
}

impl<I> DataframeDataset<I>{
    pub fn from_dataframe(df: DataFrame) -> Result<Self>{
        let len = df.height();
        Ok(DataframeDataset{
            dataframe: df,
            len,
            phantom: PhantomData
        })
        
        
    }
    
    fn check_if_row_is_serializable(&self, row: &Row, schema: &Schema) -> Result<bool>{
        match Self::row_to_serde_value(row, schema){
            Ok(s) => {
                if !s.is_null(){Ok(true) } else { Ok(false) }
            },
            Err(e) => Err(e)
        }
    }
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

impl <I> Dataset<I> for DataframeDataset<I> where I: Clone + Send + Sync + DeserializeOwned {
    fn get(&self, index: usize) -> Option<I> {
        let row = self.dataframe.get_row(index).unwrap();
        let schema = self.dataframe.schema();
        match self.check_if_row_is_serializable(&row, &schema){
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
                                . Error: {}", e);
                                None
                            }
                        }
                    },
                    false => { None }
                }
            },
            Err(e) => {
                println!("An error occurred while check if a Polars row is serializable: Error: \
                {}", e);
                None
            }
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.dataframe.is_empty()
    }
}