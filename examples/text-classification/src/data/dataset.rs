use burn::data::dataset::{
    source::huggingface::downloader::HuggingfaceDatasetLoader, Dataset, InMemDataset,
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
    dataset: InMemDataset<AgNewsItem>,
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
    dataset: InMemDataset<DbPediaItem>,
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
        let dataset: InMemDataset<DbPediaItem> =
            HuggingfaceDatasetLoader::new("dbpedia_14", "train")
                .extract_string("title")
                .extract_string("content")
                .extract_number("label")
                .load_in_memory()
                .unwrap();
        Self { dataset }
    }

    pub fn test() -> Self {
        let dataset: InMemDataset<DbPediaItem> =
            HuggingfaceDatasetLoader::new("dbpedia_14", "test")
                .extract_string("title")
                .extract_string("content")
                .extract_number("label")
                .load_in_memory()
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
