use crate::source::huggingface::downloader::HuggingfaceDatasetLoader;
use crate::transform::{Mapper, MapperDataset};
use crate::{Dataset, SqliteDataset};

use image;
use serde::{Deserialize, Serialize};

const WIDTH: usize = 28;
const HEIGHT: usize = 28;

/// MNIST item.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MNISTItem {
    /// Image as a 2D array of floats.
    pub image: [[f32; WIDTH]; HEIGHT],

    /// Label of the image.
    pub label: usize,
}

#[derive(Deserialize, Debug, Clone)]
struct MNISTItemRaw {
    pub image_bytes: Vec<u8>,
    pub label: usize,
}

struct BytesToImage;

impl Mapper<MNISTItemRaw, MNISTItem> for BytesToImage {
    /// Convert a raw MNIST item (image bytes) to a MNIST item (2D array image).
    fn map(&self, item: &MNISTItemRaw) -> MNISTItem {
        let image = image::load_from_memory(&item.image_bytes).unwrap();
        let image = image.as_luma8().unwrap();

        // Ensure the image dimensions are correct.
        debug_assert_eq!(image.dimensions(), (WIDTH as u32, HEIGHT as u32));

        // Convert the image to a 2D array of floats.
        let mut image_array = [[0f32; WIDTH]; HEIGHT];
        for (i, pixel) in image.as_raw().iter().enumerate() {
            let x = i % WIDTH;
            let y = i / HEIGHT;
            image_array[y][x] = *pixel as f32;
        }

        MNISTItem {
            image: image_array,
            label: item.label,
        }
    }
}

type MappedDataset = MapperDataset<SqliteDataset<MNISTItemRaw>, BytesToImage, MNISTItemRaw>;

/// MNIST dataset from Huggingface.
///
/// The data is downloaded from Huggingface and stored in a SQLite database.
pub struct MNISTDataset {
    dataset: MappedDataset,
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
    /// Creates a new train dataset.
    pub fn train() -> Self {
        Self::new("train")
    }

    /// Creates a new test dataset.
    pub fn test() -> Self {
        Self::new("test")
    }

    fn new(split: &str) -> Self {
        let dataset = HuggingfaceDatasetLoader::new("mnist")
            .dataset(split)
            .unwrap();

        let dataset = MapperDataset::new(dataset, BytesToImage);

        Self { dataset }
    }
}
