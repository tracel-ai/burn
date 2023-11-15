use burn::data::dataset::transform::{Mapper, MapperDataset};
use burn::data::dataset::{Dataset, InMemDataset};

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
    pub label: u8,
}

#[derive(Deserialize, Debug, Clone)]
struct MNISTItemRaw {
    pub image_bytes: Vec<u8>,
    pub label: u8,
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

type MappedDataset = MapperDataset<InMemDataset<MNISTItemRaw>, BytesToImage, MNISTItemRaw>;

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
    /// Creates a new dataset.
    pub fn new(labels: &[u8], images: &[u8], lengths: &[u16]) -> Self {
        // Decoding is here.
        // Encoding is done at `examples/train-web/web/src/train.ts`.
        debug_assert!(labels.len() == lengths.len());
        let mut start = 0 as usize;
        let raws = labels
            .iter()
            .zip(lengths.iter())
            .map(|(label, length_ptr)| {
                let length = *length_ptr as usize;
                let end = start + length;
                let raw = MNISTItemRaw {
                    label: *label,
                    image_bytes: images[start..end].to_vec(),
                };
                start = end;
                raw
            })
            .collect();
        let dataset = MapperDataset::new(InMemDataset::new(raws), BytesToImage);
        Self { dataset }
    }
}
