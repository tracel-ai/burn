use crate::{diabetes_patient::DiabetesPatient, utils::download_csv_if_missing};
use burn_dataset::{DataframeDataset, Dataset};
use polars::prelude::*;
/// Diabetes dataset using Polars DataframeDataset as the backend.
pub struct DiabetesDataframeDataset {
    dataset: DataframeDataset<DiabetesPatient>,
}

impl DiabetesDataframeDataset {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Download dataset csv file
        let path = download_csv_if_missing();

        // Column definitions: (name, schema_type for parsing, cast_type for final output)
        const COLS: &[(&str, DataType, DataType)] = &[
            ("AGE", DataType::Int64, DataType::Int8),
            ("SEX", DataType::Int64, DataType::Int8),
            ("BMI", DataType::Float64, DataType::Float32),
            ("BP", DataType::Float64, DataType::Float32),
            ("S1", DataType::Int64, DataType::Int16),
            ("S2", DataType::Float64, DataType::Float32),
            ("S3", DataType::Float64, DataType::Float32),
            ("S4", DataType::Float64, DataType::Float32),
            ("S5", DataType::Float64, DataType::Float32),
            ("S6", DataType::Int64, DataType::Int8),
            ("Y", DataType::Int64, DataType::Int16),
        ];

        // Build Schema
        let schema = Schema::from_iter(
            COLS.iter()
                .map(|(name, schema_type, _)| Field::new((*name).into(), schema_type.clone())),
        );

        let mut df = LazyCsvReader::new(PlPath::new(path.to_str().unwrap()))
            .with_has_header(true)
            .with_separator(b'\t')
            .with_schema(Some(Arc::new(schema)))
            .finish()?
            .collect()?;

        // cast columns
        for (col, _, cast_type) in COLS {
            df.with_column(df.column(col)?.cast(cast_type)?.clone())?;
        }

        let dataset = DataframeDataset::new(df)?;

        Ok(Self { dataset })
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
