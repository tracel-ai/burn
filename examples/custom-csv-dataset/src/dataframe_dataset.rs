use burn_dataset::{DataframeDataset, Dataset};
use polars::prelude::*;
use serde::Deserialize;
use std::{fs::File, io::copy, path::Path, sync::Arc};

/// Diabetes patient record for DataframeDataset.
/// Note: Field types must match Polars column types exactly.
/// Polars reads integers as i64 and floats as f64 by default.
#[derive(Deserialize, Debug, Clone)]
pub struct DiabetesPatient {
    /// Age in years
    #[serde(rename = "AGE")]
    pub age: i64,

    /// Sex categorical label
    #[serde(rename = "SEX")]
    pub sex: i64,

    /// Body mass index
    #[serde(rename = "BMI")]
    pub bmi: f64,

    /// Average blood pressure
    #[serde(rename = "BP")]
    pub bp: f64,

    /// S1: total serum cholesterol
    #[serde(rename = "S1")]
    pub tc: i64,

    /// S2: low-density lipoproteins
    #[serde(rename = "S2")]
    pub ldl: f64,

    /// S3: high-density lipoproteins
    #[serde(rename = "S3")]
    pub hdl: f64,

    /// S4: total cholesterol
    #[serde(rename = "S4")]
    pub tch: f64,

    /// S5: possibly log of serum triglycerides level
    #[serde(rename = "S5")]
    pub ltg: f64,

    /// S6: blood sugar level
    #[serde(rename = "S6")]
    pub glu: i64,

    /// Y: quantitative measure of disease progression one year after baseline
    #[serde(rename = "Y")]
    pub response: i64,
}

/// Diabetes dataset using Polars DataframeDataset as the backend.
pub struct DiabetesDataframeDataset {
    dataset: DataframeDataset<DiabetesPatient>,
}

impl DiabetesDataframeDataset {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Download dataset csv file
        let path = DiabetesDataframeDataset::download();

        // Define schema explicitly to ensure correct types
        // Some columns contain float values that might be misdetected as integers
        let schema = Schema::from_iter([
            Field::new("AGE".into(), DataType::Int64),
            Field::new("SEX".into(), DataType::Int64),
            Field::new("BMI".into(), DataType::Float64),
            Field::new("BP".into(), DataType::Float64),
            Field::new("S1".into(), DataType::Int64),
            Field::new("S2".into(), DataType::Float64),
            Field::new("S3".into(), DataType::Float64),
            Field::new("S4".into(), DataType::Float64),
            Field::new("S5".into(), DataType::Float64),
            Field::new("S6".into(), DataType::Int64),
            Field::new("Y".into(), DataType::Int64),
        ]);

        let df = LazyCsvReader::new(PlPath::new(&path))
            .with_has_header(true)
            .with_separator(b'\t')
            .with_schema(Some(Arc::new(schema)))
            .finish()?
            .collect()?;

        let dataset = DataframeDataset::new(df)?;

        Ok(Self { dataset })
    }
    /// Download the CSV file from its original source on the web.
    /// Panics if the download cannot be completed or the content of the file cannot be written to disk.
    fn download() -> String {
        // Point file to current example directory
        let example_dir = Path::new(file!()).parent().unwrap().parent().unwrap();
        let file_name = example_dir.join("diabetes.csv");

        if file_name.exists() {
            println!("File already downloaded at {file_name:?}");
        } else {
            // Get file from web
            println!("Downloading file to {file_name:?}");
            let url = "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt";
            let mut response = reqwest::blocking::get(url).unwrap();

            // Create file to write the downloaded content to
            let mut file = File::create(&file_name).unwrap();

            // Copy the downloaded contents
            copy(&mut response, &mut file).unwrap();
        };

        file_name.to_str().unwrap().to_string()
    }
}

impl Default for DiabetesDataframeDataset {
    fn default() -> Self {
        Self::new().expect("Could not load diabetes dataset")
    }
}

// Implement the `Dataset` trait which requires `get` and `len`
impl Dataset<DiabetesPatient> for DiabetesDataframeDataset {
    fn get(&self, index: usize) -> Option<DiabetesPatient> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}
