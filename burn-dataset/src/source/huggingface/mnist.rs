use crate::source::huggingface::downloader::HuggingfaceDatasetLoader;
use crate::{Dataset, InMemDataset};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MNISTItem {
    pub image: [[f32; 28]; 28],
    pub label: usize,
}

pub struct MNISTDataset {
    dataset: InMemDataset<MNISTItem>,
}

impl Dataset<MNISTItem> for MNISTDataset {
    fn get(&self, index: usize) -> Option<MNISTItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl MNISTDataset {
    pub fn train() -> Self {
        Self::new("train")
    }

    pub fn test() -> Self {
        Self::new("test")
    }

    fn new(split: &str) -> Self {
        let dataset = HuggingfaceDatasetLoader::new("mnist", split)
            .extract_image("image")
            .extract_number("label")
            .deps(&["pillow", "numpy"])
            .load_in_memory()
            .unwrap();

        Self { dataset }
    }
}
