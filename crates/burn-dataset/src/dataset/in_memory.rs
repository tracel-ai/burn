use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use serde::de::DeserializeOwned;

use crate::Dataset;

#[cfg(feature = "arrow")]
use ar_row::arrow::array::RecordBatch;
#[cfg(feature = "arrow")]
use ar_row::deserialize::ArRowDeserialize;

/// Dataset where all items are stored in ram.
pub struct InMemDataset<I> {
    items: Vec<I>,
}

impl<I> InMemDataset<I> {
    /// Creates a new in memory dataset from the given items.
    pub fn new(items: Vec<I>) -> Self {
        InMemDataset { items }
    }
}

impl<I> Dataset<I> for InMemDataset<I>
where
    I: Clone + Send + Sync,
{
    fn get(&self, index: usize) -> Option<I> {
        self.items.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.items.len()
    }
}

impl<I> InMemDataset<I>
where
    I: Clone + DeserializeOwned,
{
    /// Create from a dataset. All items are loaded in memory.
    pub fn from_dataset(dataset: &impl Dataset<I>) -> Self {
        let items: Vec<I> = dataset.iter().collect();
        Self::new(items)
    }

    /// Create from a json rows file (one json per line).
    ///
    /// [Supported field types](https://docs.rs/serde_json/latest/serde_json/value/enum.Value.html)
    pub fn from_json_rows<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut items = Vec::new();

        for line in reader.lines() {
            let item = serde_json::from_str(line.unwrap().as_str()).unwrap();
            items.push(item);
        }

        let dataset = Self::new(items);

        Ok(dataset)
    }

    /// Create from a csv file.
    ///
    /// The provided `csv::ReaderBuilder` can be configured to fit your csv format.
    ///
    /// The supported field types are: String, integer, float, and bool.
    ///
    /// See:
    /// - [Reading with Serde](https://docs.rs/csv/latest/csv/tutorial/index.html#reading-with-serde)
    /// - [Delimiters, quotes and variable length records](https://docs.rs/csv/latest/csv/tutorial/index.html#delimiters-quotes-and-variable-length-records)
    pub fn from_csv<P: AsRef<Path>>(
        path: P,
        builder: &csv::ReaderBuilder,
    ) -> Result<Self, std::io::Error> {
        let mut rdr = builder.from_path(path)?;

        let mut items = Vec::new();

        for result in rdr.deserialize() {
            let item: I = result?;
            items.push(item);
        }

        let dataset = Self::new(items);

        Ok(dataset)
    }
}

// Support for arrow record batches
#[cfg(feature = "arrow")]
impl<I> InMemDataset<I>
where
    I: ArRowDeserialize + Clone,
{
    /// Create from Arrow Record Batches
    /// See:
    /// - [Deserialize structs in a row-oriented way from Apache Arrow](https://docs.rs/ar_row_derive/latest/ar_row_derive/index.html#examples)
    pub fn from_arrow_batches(record_batches: Vec<RecordBatch>) -> Result<Self, std::io::Error> {
        let items: Vec<I> = record_batches
            .into_iter()
            .flat_map(|batch| -> Vec<I> { I::from_record_batch(batch).unwrap() })
            .collect();

        let dataset = Self::new(items);

        Ok(dataset)
    }
}
#[cfg(test)]
mod tests {

    use super::*;
    use crate::{SqliteDataset, test_data};

    use rstest::{fixture, rstest};
    use serde::{Deserialize, Serialize};

    // To test Arrow Record Batch Reads
    #[cfg(feature = "arrow")]
    use datafusion::prelude::*;
    #[cfg(feature = "arrow")]
    use tokio::runtime::Runtime;

    const DB_FILE: &str = "tests/data/sqlite-dataset.db";
    const JSON_FILE: &str = "tests/data/dataset.json";
    const CSV_FILE: &str = "tests/data/dataset.csv";
    const CSV_FMT_FILE: &str = "tests/data/dataset-fmt.csv";

    type SqlDs = SqliteDataset<Sample>;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct Sample {
        column_str: String,
        column_bytes: Vec<u8>,
        column_int: i64,
        column_bool: bool,
        column_float: f64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct SampleCsv {
        column_str: String,
        column_int: i64,
        column_bool: bool,
        column_float: f64,
    }

    #[cfg(feature = "arrow")]
    #[derive(Debug, Default, Clone, Serialize, ar_row_derive::ArRowDeserialize, PartialEq)]
    pub struct SampleCsvArrow {
        column_str: String,
        column_int: i64,
        column_bool: bool,
        column_float: f64,
    }

    #[cfg(feature = "arrow")]
    #[derive(Debug, Default, Clone, Serialize, ar_row_derive::ArRowDeserialize, PartialEq)]
    pub struct SampleCsvArrowReorder {
        column_bool_r: bool,
        column_float_r: f64,
        column_str_r: String,
        column_int_r: i64,
    }

    #[fixture]
    fn train_dataset() -> SqlDs {
        SqliteDataset::from_db_file(DB_FILE, "train").unwrap()
    }

    #[rstest]
    pub fn from_dataset(train_dataset: SqlDs) {
        let dataset = InMemDataset::from_dataset(&train_dataset);

        let non_existing_record_index: usize = 10;
        let record_index: usize = 0;

        assert_eq!(train_dataset.get(non_existing_record_index), None);
        assert_eq!(dataset.get(record_index).unwrap().column_str, "HI1");
    }

    #[test]
    pub fn from_json_rows() {
        let dataset = InMemDataset::<Sample>::from_json_rows(JSON_FILE).unwrap();

        let non_existing_record_index: usize = 10;
        let record_index: usize = 1;

        assert_eq!(dataset.get(non_existing_record_index), None);
        assert_eq!(dataset.get(record_index).unwrap().column_str, "HI2");
        assert!(!dataset.get(record_index).unwrap().column_bool);
    }

    #[test]
    pub fn from_csv_rows() {
        let rdr = csv::ReaderBuilder::new();
        let dataset = InMemDataset::<SampleCsv>::from_csv(CSV_FILE, &rdr).unwrap();

        let non_existing_record_index: usize = 10;
        let record_index: usize = 1;

        assert_eq!(dataset.get(non_existing_record_index), None);
        assert_eq!(dataset.get(record_index).unwrap().column_str, "HI2");
        assert_eq!(dataset.get(record_index).unwrap().column_int, 1);
        assert!(!dataset.get(record_index).unwrap().column_bool);
        assert_eq!(dataset.get(record_index).unwrap().column_float, 1.0);
    }

    #[test]
    #[cfg(feature = "arrow")]
    pub fn from_csv_arrow() {
        let rt = Runtime::new().unwrap();
        let record_batches: Vec<RecordBatch> = rt.block_on(async {
            let ctx = SessionContext::new();

            let df = ctx.read_csv(CSV_FILE, CsvReadOptions::new()).await.unwrap();
            df.collect().await.unwrap()
        });
        let dataset = InMemDataset::<SampleCsvArrow>::from_arrow_batches(record_batches).unwrap();

        let non_existing_record_index: usize = 10;
        let record_index: usize = 1;

        assert_eq!(dataset.get(non_existing_record_index), None);
        assert_eq!(dataset.get(record_index).unwrap().column_str, "HI2");
        assert_eq!(dataset.get(record_index).unwrap().column_int, 1);
        assert!(!dataset.get(record_index).unwrap().column_bool);
        assert_eq!(dataset.get(record_index).unwrap().column_float, 1.0);
    }

    #[test]
    #[cfg(feature = "arrow")]
    pub fn from_csv_arrow_renamed() {
        let rt = Runtime::new().unwrap();
        let record_batches: Vec<RecordBatch> = rt.block_on(async {
            let ctx = SessionContext::new();

            ctx.register_csv("test_csv", CSV_FILE, CsvReadOptions::new())
                .await
                .unwrap();

            let df = ctx
                .sql(
                    r#"
                SELECT
                    column_bool AS column_bool_r,
                    column_float AS column_float_r,
                    column_str AS column_str_r,
                    column_int AS column_int_r,
                FROM test_csv
                "#,
                )
                .await
                .unwrap();

            df.collect().await.unwrap()
        });
        let dataset =
            InMemDataset::<SampleCsvArrowReorder>::from_arrow_batches(record_batches).unwrap();

        let non_existing_record_index: usize = 10;
        let record_index: usize = 1;

        assert_eq!(dataset.get(non_existing_record_index), None);
        assert_eq!(dataset.get(record_index).unwrap().column_str_r, "HI2");
        assert_eq!(dataset.get(record_index).unwrap().column_int_r, 1);
        assert!(!dataset.get(record_index).unwrap().column_bool_r);
        assert_eq!(dataset.get(record_index).unwrap().column_float_r, 1.0);
    }

    #[test]
    pub fn from_csv_rows_fmt() {
        let mut rdr = csv::ReaderBuilder::new();
        let rdr = rdr.delimiter(b' ').has_headers(false);
        let dataset = InMemDataset::<SampleCsv>::from_csv(CSV_FMT_FILE, rdr).unwrap();

        let non_existing_record_index: usize = 10;
        let record_index: usize = 1;

        assert_eq!(dataset.get(non_existing_record_index), None);
        assert_eq!(dataset.get(record_index).unwrap().column_str, "HI2");
        assert_eq!(dataset.get(record_index).unwrap().column_int, 1);
        assert!(!dataset.get(record_index).unwrap().column_bool);
        assert_eq!(dataset.get(record_index).unwrap().column_float, 1.0);
    }

    #[test]
    pub fn given_in_memory_dataset_when_iterate_should_iterate_though_all_items() {
        let items_original = test_data::string_items();
        let dataset = InMemDataset::new(items_original.clone());

        let items: Vec<String> = dataset.iter().collect();

        assert_eq!(items_original, items);
    }
}
