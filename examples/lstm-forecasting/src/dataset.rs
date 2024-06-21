use std::usize;

use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{transform::PartialDataset, Dataset, HuggingfaceDatasetLoader, SqliteDataset},
    },
    prelude::*,
};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StockItem {
    #[serde()]
    pub adj_close: f64,
}

type Data = PartialDataset<SqliteDataset<StockItem>, StockItem>;
type PartialData = PartialDataset<WindowedDataset, Vec<StockItem>>;

pub struct StockDataset {
    dataset: PartialData,
}

pub struct WindowedDataset {
    dataset: Data,
    window_size: usize,
}

impl Dataset<Vec<StockItem>> for WindowedDataset {
    fn get(&self, index: usize) -> Option<Vec<StockItem>> {
        (index..index + self.window_size)
            .map(|x| self.dataset.get(x))
            .collect()
    }

    fn len(&self) -> usize {
        self.dataset.len() - self.window_size + 1
    }
}

impl Dataset<Vec<StockItem>> for StockDataset {
    fn get(&self, index: usize) -> Option<Vec<StockItem>> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl StockDataset {
    pub fn train(window_size: usize, dataset_size: usize) -> Self {
        Self::new("train", window_size, dataset_size)
    }

    pub fn test(window_size: usize, dataset_size: usize) -> Self {
        Self::new("test", window_size, dataset_size)
    }

    pub fn new(split: &str, window_size: usize, dataset_size: usize) -> Self {
        let hugging_face_dataset =
            HuggingfaceDatasetLoader::new("edarchimbaud/timeseries-1d-stocks")
                .dataset("train")
                .unwrap();

        let len = hugging_face_dataset.len();

        let limit = match dataset_size {
            0 => 0,
            _ => len - dataset_size - window_size,
        };

        let dataset: WindowedDataset = WindowedDataset {
            dataset: PartialDataset::new(hugging_face_dataset, limit, len),
            window_size,
        };

        let len = dataset.len();

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

#[derive(Clone, Debug)]
pub struct StockBatch<B: Backend> {
    pub inputs: Tensor<B, 3>,
    pub targets: Tensor<B, 1>,
}

#[derive(Clone, Debug)]
pub struct StockBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> StockBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    pub fn min_max_norm<const D: usize>(&self, inp: Tensor<B, D>) -> Tensor<B, D> {
        let min = inp.clone().min_dim(0);
        let max = inp.clone().max_dim(0);
        (inp.clone() - min.clone()).div(max - min)
    }
}

impl<B: Backend> Batcher<Vec<StockItem>, StockBatch<B>> for StockBatcher<B> {
    fn batch(&self, items: Vec<Vec<StockItem>>) -> StockBatch<B> {
        // Create input from time windows, shape of a window is ([window_size -1, 1])
        let inputs: Vec<Tensor<B, 2>> = items
            .iter()
            .map(|sequence| {
                Tensor::stack::<2>(
                    sequence
                        .split_at(sequence.len() - 1)
                        .0
                        .iter()
                        .map(|x| Tensor::<B, 1>::from_floats([x.adj_close as f32], &self.device))
                        .collect(),
                    0,
                )
            })
            .collect();

        let inputs = self.min_max_norm(Tensor::stack(inputs, 0));

        // Create targets from last time steps
        let targets = items
            .iter()
            .map(|sequence| {
                Tensor::<B, 1>::from_floats(
                    [sequence[sequence.len() - 1].adj_close as f32],
                    &self.device,
                )
            })
            .collect();

        let targets = Tensor::cat(targets, 0);
        let targets = self.min_max_norm(targets);

        StockBatch { inputs, targets }
    }
}
