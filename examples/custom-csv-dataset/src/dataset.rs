use burn::data::dataset::{Dataset, InMemDataset};
use crate::{
    utils::download_csv_if_missing,
    diabetes_patient::DiabetesPatient,
};

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
        let path = download_csv_if_missing();

        // Build dataset from csv with tab ('\t') delimiter
        let mut rdr = csv::ReaderBuilder::new();
        let rdr = rdr.delimiter(b'\t');

        let dataset = InMemDataset::from_csv(path, rdr).unwrap();

        let dataset = Self { dataset };

        Ok(dataset)
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
