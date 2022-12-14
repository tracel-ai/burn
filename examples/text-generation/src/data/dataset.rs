use burn::data::dataset::{
    source::huggingface::downloader::HuggingfaceDatasetLoader, Dataset, InMemDataset,
};

#[derive(new, Clone, Debug)]
pub struct TextGenerationItem {
    pub text: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AgNewsItem {
    pub text: String,
}

pub struct AgNewsDataset {
    dataset: InMemDataset<AgNewsItem>,
}

impl Dataset<TextGenerationItem> for AgNewsDataset {
    fn get(&self, index: usize) -> Option<TextGenerationItem> {
        self.dataset
            .get(index)
            .map(|item| TextGenerationItem::new(item.text))
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl AgNewsDataset {
    pub fn train() -> Self {
        let dataset: InMemDataset<AgNewsItem> = HuggingfaceDatasetLoader::new("ag_news", "train")
            .extract_string("text")
            .extract_number("label")
            .load_in_memory()
            .unwrap();
        Self { dataset }
    }

    pub fn test() -> Self {
        let dataset: InMemDataset<AgNewsItem> = HuggingfaceDatasetLoader::new("ag_news", "test")
            .extract_string("text")
            .extract_number("label")
            .load_in_memory()
            .unwrap();
        Self { dataset }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DbPediaItem {
    pub content: String,
}

pub struct DbPediaDataset {
    dataset: InMemDataset<DbPediaItem>,
}

impl Dataset<TextGenerationItem> for DbPediaDataset {
    fn get(&self, index: usize) -> Option<TextGenerationItem> {
        self.dataset
            .get(index)
            .map(|item| TextGenerationItem::new(item.content))
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl DbPediaDataset {
    pub fn train() -> Self {
        let dataset: InMemDataset<DbPediaItem> =
            HuggingfaceDatasetLoader::new("dbpedia_14", "train")
                .extract_string("content")
                .load_in_memory()
                .unwrap();
        Self { dataset }
    }

    pub fn test() -> Self {
        let dataset: InMemDataset<DbPediaItem> =
            HuggingfaceDatasetLoader::new("dbpedia_14", "test")
                .extract_string("content")
                .load_in_memory()
                .unwrap();
        Self { dataset }
    }
}
