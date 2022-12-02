use burn::data::dataset::{
    source::huggingface::downloader::HuggingfaceDatasetLoader, Dataset, InMemDataset,
};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TextClassificationItem {
    pub text: String,
    pub label: usize,
}

impl Dataset<TextClassificationItem> for AgNewsDataset {
    fn get(&self, index: usize) -> Option<TextClassificationItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

pub trait TextClassificationDataset: Dataset<TextClassificationItem> {
    fn num_classes() -> usize;
    fn class_name(label: usize) -> String;
}

pub struct AgNewsDataset {
    dataset: InMemDataset<TextClassificationItem>,
}

impl TextClassificationDataset for AgNewsDataset {
    fn num_classes() -> usize {
        4
    }

    fn class_name(label: usize) -> String {
        match label {
            0 => "World",
            1 => "Sports",
            2 => "Business",
            3 => "Technology",
            _ => panic!("invalid class"),
        }
        .to_string()
    }
}

impl AgNewsDataset {
    pub fn train() -> Self {
        let dataset: InMemDataset<TextClassificationItem> =
            HuggingfaceDatasetLoader::new("ag_news", "train")
                .extract_string("text")
                .extract_number("label")
                .load_in_memory()
                .unwrap();
        Self { dataset }
    }

    pub fn test() -> Self {
        let dataset: InMemDataset<TextClassificationItem> =
            HuggingfaceDatasetLoader::new("ag_news", "test")
                .extract_string("text")
                .extract_number("label")
                .load_in_memory()
                .unwrap();
        Self { dataset }
    }
}
