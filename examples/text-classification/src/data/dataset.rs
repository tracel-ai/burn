use burn::data::dataset::{
    source::huggingface::downloader::HuggingfaceDatasetLoader, Dataset, SqliteDataset,
};

#[derive(new, Clone, Debug)]
pub struct TextClassificationItem {
    pub text: String,
    pub label: usize,
}

pub trait TextClassificationDataset: Dataset<TextClassificationItem> {
    fn num_classes() -> usize;
    fn class_name(label: usize) -> String;
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AgNewsItem {
    pub text: String,
    pub label: usize,
}

pub struct AgNewsDataset {
    dataset: SqliteDataset<AgNewsItem>,
}

impl Dataset<TextClassificationItem> for AgNewsDataset {
    fn get(&self, index: usize) -> Option<TextClassificationItem> {
        self.dataset
            .get(index)
            .map(|item| TextClassificationItem::new(item.text, item.label))
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl AgNewsDataset {
    pub fn train() -> Self {
        Self::new("train")
    }

    pub fn test() -> Self {
        Self::new("test")
    }

    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<AgNewsItem> = HuggingfaceDatasetLoader::new("ag_news")
            .dataset(split)
            .unwrap();
        Self { dataset }
    }
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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DbPediaItem {
    pub title: String,
    pub content: String,
    pub label: usize,
}

pub struct DbPediaDataset {
    dataset: SqliteDataset<DbPediaItem>,
}

impl Dataset<TextClassificationItem> for DbPediaDataset {
    fn get(&self, index: usize) -> Option<TextClassificationItem> {
        self.dataset.get(index).map(|item| {
            TextClassificationItem::new(
                format!("Title: {} - Content: {}", item.title, item.content),
                item.label,
            )
        })
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

impl TextClassificationDataset for DbPediaDataset {
    fn num_classes() -> usize {
        14
    }

    fn class_name(label: usize) -> String {
        match label {
            0 => "Company",
            1 => "EducationalInstitution",
            2 => "Artist",
            3 => "Athlete",
            4 => "OfficeHolder",
            5 => "MeanOfTransportation",
            6 => "Building",
            7 => "NaturalPlace",
            8 => "Village",
            9 => "Animal",
            10 => "Plant",
            11 => "Album",
            12 => "Film",
            13 => "WrittenWork",
            _ => panic!("invalid class"),
        }
        .to_string()
    }
}
