use burn::data::dataset::{Dataset, InMemDataset};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::copy,
    path::{Path, PathBuf},
};

/// Diabetes patient record.
/// For each field, we manually specify the expected header name for serde as all names
/// are capitalized and some field names are not very informative.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct DiabetesPatient {
    /// Age in years
    #[serde(rename = "AGE")]
    pub age: u8,

    /// Sex categorical label
    #[serde(rename = "SEX")]
    pub sex: u8,

    /// Body mass index
    #[serde(rename = "BMI")]
    pub bmi: f32,

    /// Average blood pressure
    #[serde(rename = "BP")]
    pub bp: f32,

    /// S1: total serum cholesterol
    #[serde(rename = "S1")]
    pub tc: u16,

    /// S2: low-density lipoproteins
    #[serde(rename = "S2")]
    pub ldl: f32,

    /// S3: high-density lipoproteins
    #[serde(rename = "S3")]
    pub hdl: f32,

    /// S4: total cholesterol
    #[serde(rename = "S4")]
    pub tch: f32,

    /// S5: possibly log of serum triglycerides level
    #[serde(rename = "S5")]
    pub ltg: f32,

    /// S6: blood sugar level
    #[serde(rename = "S6")]
    pub glu: u8,

    /// Y: quantitative measure of disease progression one year after baseline
    #[serde(rename = "Y")]
    pub response: u16,
}

/// Diabetes patients dataset, also used in [scikit-learn](https://scikit-learn.org/stable/).
/// See [Diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset).
///
/// The data is parsed from a single csv file (tab as the delimiter).
/// The dataset contains 10 baseline variables (age, sex, body mass index, average blood pressure and
/// 6 blood serum measurements for a total of 442 diabetes patients.
/// For each patient, the response of interest, a quantitative measure of disease progression one year
/// after baseline, was collected. This represents the target variable.
pub struct DiabetesDataset {
    dataset: InMemDataset<DiabetesPatient>,
}

impl DiabetesDataset {
    pub fn new() -> Result<Self, std::io::Error> {
        // Download dataset csv file
        let path = DiabetesDataset::download();

        // Build dataset from csv with tab ('\t') delimiter
        let mut rdr = csv::ReaderBuilder::new();
        let rdr = rdr.delimiter(b'\t');

        let dataset = InMemDataset::from_csv(path, rdr).unwrap();

        let dataset = Self { dataset };

        Ok(dataset)
    }
    /// Download the CSV file from its original source on the web.
    /// Panics if the download cannot be completed or the content of the file cannot be written to disk.
    fn download() -> PathBuf {
        // Point file to current example directory
        let example_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
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

        file_name
    }
}

// Implement the `Dataset` trait which requires `get` and `len`
impl Dataset<DiabetesPatient> for DiabetesDataset {
    fn get(&self, index: usize) -> Option<DiabetesPatient> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}
