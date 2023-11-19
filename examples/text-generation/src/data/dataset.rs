use burn::data::dataset::{source::huggingface::HuggingfaceDatasetLoader, Dataset, SqliteDataset};

#[derive(new, Clone, Debug)]
pub struct TextGenerationItem {
    pub text: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DbPediaItem {
    pub content: String,
}

pub struct DbPediaDataset {
    dataset: SqliteDataset<DbPediaItem>,
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
        Self::new("train")
    }

    pub fn test() -> Self {
        Self::new("test")
    }
    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<DbPediaItem> = HuggingfaceDatasetLoader::new("dbpedia_14")
            .dataset(split)
            .unwrap();
        Self { dataset }
    }
}
