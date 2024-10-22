use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{
            transform::{PartialDataset, ShuffledDataset},
            Dataset, HuggingfaceDatasetLoader, SqliteDataset,
        },
    },
    prelude::*,
};

pub const NUM_FEATURES: usize = 8;
const FEATURES_MIN: [f32; NUM_FEATURES] = [0.4999, 1., 0.8461, 0.3333, 3., 0.6923, 32.54, -124.35];
const FEATURES_MAX: [f32; NUM_FEATURES] = [
    15., 52., 141.9091, 34.0667, 35682., 1243.3333, 41.95, -114.31,
];

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HousingDistrictItem {
    /// Median income
    #[serde(rename = "MedInc")]
    pub median_income: f32,

    /// Median house age
    #[serde(rename = "HouseAge")]
    pub house_age: f32,

    /// Average number of rooms per household
    #[serde(rename = "AveRooms")]
    pub avg_rooms: f32,

    /// Average number of bedrooms per household
    #[serde(rename = "AveBedrms")]
    pub avg_bedrooms: f32,

    /// Block group population
    #[serde(rename = "Population")]
    pub population: f32,

    /// Average number of household members
    #[serde(rename = "AveOccup")]
    pub avg_occupancy: f32,

    /// Block group latitude
    #[serde(rename = "Latitude")]
    pub latitude: f32,

    /// Block group longitude
    #[serde(rename = "Longitude")]
    pub longitude: f32,

    /// Median house value (in 100 000$)
    #[serde(rename = "MedHouseVal")]
    pub median_house_value: f32,
}

type ShuffledData = ShuffledDataset<SqliteDataset<HousingDistrictItem>, HousingDistrictItem>;
type PartialData = PartialDataset<ShuffledData, HousingDistrictItem>;

pub struct HousingDataset {
    dataset: PartialData,
}

impl Dataset<HousingDistrictItem> for HousingDataset {
    fn get(&self, index: usize) -> Option<HousingDistrictItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl HousingDataset {
    pub fn train() -> Self {
        Self::new("train")
    }

    pub fn test() -> Self {
        Self::new("test")
    }

    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<HousingDistrictItem> =
            HuggingfaceDatasetLoader::new("gvlassis/california_housing")
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

/// Normalizer for the housing dataset.
#[derive(Clone, Debug)]
pub struct Normalizer<B: Backend> {
    pub min: Tensor<B, 2>,
    pub max: Tensor<B, 2>,
}

impl<B: Backend> Normalizer<B> {
    /// Creates a new normalizer.
    pub fn new(device: &B::Device, min: &[f32], max: &[f32]) -> Self {
        let min = Tensor::<B, 1>::from_floats(min, device).unsqueeze();
        let max = Tensor::<B, 1>::from_floats(max, device).unsqueeze();
        Self { min, max }
    }

    /// Normalizes the input image according to the housing dataset min/max.
    pub fn normalize(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        (input - self.min.clone()) / (self.max.clone() - self.min.clone())
    }
}

#[derive(Clone, Debug)]
pub struct HousingBatcher<B: Backend> {
    device: B::Device,
    normalizer: Normalizer<B>,
}

#[derive(Clone, Debug)]
pub struct HousingBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1>,
}

impl<B: Backend> HousingBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            device: device.clone(),
            normalizer: Normalizer::new(&device, &FEATURES_MIN, &FEATURES_MAX),
        }
    }

    pub fn min_max_norm<const D: usize>(&self, inp: Tensor<B, D>) -> Tensor<B, D> {
        let min = inp.clone().min_dim(0);
        let max = inp.clone().max_dim(0);
        (inp.clone() - min.clone()).div(max - min)
    }
}

impl<B: Backend> Batcher<HousingDistrictItem, HousingBatch<B>> for HousingBatcher<B> {
    fn batch(&self, items: Vec<HousingDistrictItem>) -> HousingBatch<B> {
        let mut inputs: Vec<Tensor<B, 2>> = Vec::new();

        for item in items.iter() {
            let input_tensor = Tensor::<B, 1>::from_floats(
                [
                    item.median_income,
                    item.house_age,
                    item.avg_rooms,
                    item.avg_bedrooms,
                    item.population,
                    item.avg_occupancy,
                    item.latitude,
                    item.longitude,
                ],
                &self.device,
            );

            inputs.push(input_tensor.unsqueeze());
        }

        let inputs = Tensor::cat(inputs, 0);
        // let inputs = self.min_max_norm(inputs);
        let inputs = self.normalizer.normalize(inputs);

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_floats([item.median_house_value], &self.device))
            .collect();

        let targets = Tensor::cat(targets, 0);

        HousingBatch { inputs, targets }
    }
}
