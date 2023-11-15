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
        assert!(
            labels.len() == lengths.len(),
            "`labels` has {} elements and `lengths` has {} elements, but they should be equal in size.",
            labels.len(),
            lengths.len(),
        );
        let mut start = 0 as usize;
        let raws = labels
            .iter()
            .zip(lengths.iter())
            .map(|(label, length_ptr)| {
                let length = *length_ptr as usize;
                let end = start + length;
                let image_bytes = images[start..end].to_vec();
                // Assert that the incoming data is a valid PNG.
                // Really just here to ensure the decoding process isn't off by one.
                debug_assert_eq!(
                    image_bytes[0..8],
                    vec![137, 80, 78, 71, 13, 10, 26, 10] // Starts with bytes from the spec http://www.libpng.org/pub/png/spec/1.2/PNG-Structure.html
                );
                debug_assert_eq!(
                    image_bytes[image_bytes.len() - 8..],
                    vec![73, 69, 78, 68, 174, 66, 96, 130] // Ends with the IEND chunk. I don't think the spec specifies the last 4 bytes, but `mnist.db`'s images all end with it.
                );
                let raw = MNISTItemRaw {
                    label: *label,
                    image_bytes,
                };
                start = end;
                raw
            })
            .collect();
        let dataset = MapperDataset::new(InMemDataset::new(raws), BytesToImage);
        Self { dataset }
    }
}
