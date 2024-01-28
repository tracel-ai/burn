use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::transform::{PartialDataset, ShuffledDataset};
use burn::data::dataset::{Dataset, HuggingfaceDatasetLoader, SqliteDataset};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DiabetesItem {
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

type ShuffledData = ShuffledDataset<SqliteDataset<DiabetesItem>, DiabetesItem>;
type PartialData = PartialDataset<ShuffledData, DiabetesItem>;

pub struct DiabetesDataset {
    dataset: PartialData,
}

impl Dataset<DiabetesItem> for DiabetesDataset {
    fn get(&self, index: usize) -> Option<DiabetesItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl DiabetesDataset {
    pub fn train() -> Self {
        Self::new("train")
    }

    pub fn test() -> Self {
        Self::new("test")
    }

    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<DiabetesItem> =
            HuggingfaceDatasetLoader::new("Jayabalambika/toy-diabetes")
                .dataset("train")
                .unwrap();

        let len = dataset.len();

        // Shuffle the dataset with a defined seed such that train and test sets have no overlap
        // when splitting by indexes
        let dataset = ShuffledDataset::with_seed(dataset, 42);

        // The dataset from HuggingFace has only train split, so we manually split the train dataset into train
        // and test in a 80-20 ratio

        let filtered_dataset = match split {
            "train" => PartialData::new(dataset, 0, len * 8 / 10), // Get first 80% dataset
            "test" => PartialData::new(dataset, len * 8 / 10, len), // Take remaining 20%
            _ => panic!("Invalid split type"),                     // Handle unexpected split types
        };

        Self {
            dataset: filtered_dataset,
        }
    }
}

pub struct DiabetesBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct DiabetesBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1>,
}

impl<B: Backend> DiabetesBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    pub fn min_max_norm<const D: usize>(&self, inp: Tensor<B, D>) -> Tensor<B, D> {
        let min = inp.clone().min_dim(0);
        let max = inp.clone().max_dim(0);
        (inp.clone() - min.clone()).div(max - min)
    }
}

impl<B: Backend> Batcher<DiabetesItem, DiabetesBatch<B>> for DiabetesBatcher<B> {
    fn batch(&self, items: Vec<DiabetesItem>) -> DiabetesBatch<B> {
        let mut inputs: Vec<Tensor<B, 2>> = Vec::new();

        for item in items.iter() {
            let input_tensor = Tensor::<B, 1>::from_floats(
                [
                    item.age as f32,
                    item.sex as f32,
                    item.bmi,
                    item.bp,
                    item.tc as f32,
                    item.ldl,
                    item.hdl,
                    item.tch,
                    item.ltg,
                    item.glu as f32,
                ],
                &self.device,
            );

            inputs.push(input_tensor.unsqueeze());
        }

        let inputs = Tensor::cat(inputs, 0);
        let inputs = self.min_max_norm(inputs);

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_floats([item.response as f32], &self.device))
            .collect();

        let targets = Tensor::cat(targets, 0);
        let targets = self.min_max_norm(targets);

        DiabetesBatch { inputs, targets }
    }
}
