use std::fs::{File, create_dir_all};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};

use crate::{
    Dataset, DatasetError, InMemDataset,
    transform::{Mapper, MapperDataset},
};

use crate::network::downloader::download_file_as_bytes;

// CVDF mirror of http://yann.lecun.com/exdb/mnist/
const URL: &str = "https://storage.googleapis.com/cvdf-datasets/mnist/";
const TRAIN_IMAGES: &str = "train-images-idx3-ubyte";
const TRAIN_LABELS: &str = "train-labels-idx1-ubyte";
const TEST_IMAGES: &str = "t10k-images-idx3-ubyte";
const TEST_LABELS: &str = "t10k-labels-idx1-ubyte";

const WIDTH: usize = 28;
const HEIGHT: usize = 28;

/// MNIST item.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MnistItem {
    /// Image as a 2D array of floats.
    pub image: [[f32; WIDTH]; HEIGHT],

    /// Label of the image.
    pub label: u8,
}

#[derive(Deserialize, Debug, Clone)]
struct MnistItemRaw {
    pub image_bytes: Vec<u8>,
    pub label: u8,
}

struct BytesToImage;

impl Mapper<MnistItemRaw, MnistItem> for BytesToImage {
    /// Convert a raw MNIST item (image bytes) to a MNIST item (2D array image).
    fn map(&self, item: &MnistItemRaw) -> MnistItem {
        // Ensure the image dimensions are correct.
        debug_assert_eq!(item.image_bytes.len(), WIDTH * HEIGHT);

        // Convert the image to a 2D array of floats.
        let mut image_array = [[0f32; WIDTH]; HEIGHT];
        for (i, pixel) in item.image_bytes.iter().enumerate() {
            let x = i % WIDTH;
            let y = i / HEIGHT;
            image_array[y][x] = *pixel as f32;
        }

        MnistItem {
            image: image_array,
            label: item.label,
        }
    }
}

type MappedDataset = MapperDataset<InMemDataset<MnistItemRaw>, BytesToImage, MnistItemRaw>;

/// The MNIST dataset consists of 70,000 28x28 black-and-white images in 10 classes (one for each digits), with 7,000
/// images per class. There are 60,000 training images and 10,000 test images.
///
/// The data is downloaded from the web from the [CVDF mirror](https://github.com/cvdfoundation/mnist).
pub struct MnistDataset {
    dataset: MappedDataset,
}

impl Dataset<MnistItem> for MnistDataset {
    fn get(&self, index: usize) -> Result<MnistItem, DatasetError> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl MnistDataset {
    /// Creates a new train dataset.
    pub fn train() -> Self {
        Self::new("train")
    }

    /// Creates a new test dataset.
    pub fn test() -> Self {
        Self::new("test")
    }

    fn new(split: &str) -> Self {
        // Download dataset
        let root = MnistDataset::download(split);

        // MNIST is tiny so we can load it in-memory
        // Train images (u8): 28 * 28 * 60000 = 47.04Mb
        // Test images (u8): 28 * 28 * 10000 = 7.84Mb
        let images = MnistDataset::read_images(&root, split);
        let labels = MnistDataset::read_labels(&root, split);

        // Collect as vector of MnistItemRaw
        let items: Vec<_> = images
            .into_iter()
            .zip(labels)
            .map(|(image_bytes, label)| MnistItemRaw { image_bytes, label })
            .collect();

        let dataset = InMemDataset::new(items);
        let dataset = MapperDataset::new(dataset, BytesToImage);

        Self { dataset }
    }

    /// Download the MNIST dataset files from the web.
    /// Panics if the download cannot be completed or the content of the file cannot be written to disk.
    fn download(split: &str) -> PathBuf {
        // Dataset files are stored in the burn-dataset cache directory
        let cache_dir = dirs::cache_dir()
            .expect("Could not get cache directory")
            .join("burn-dataset");
        let split_dir = cache_dir.join("mnist").join(split);

        if !split_dir.exists() {
            create_dir_all(&split_dir).expect("Failed to create base directory");
        }

        // Download split files
        match split {
            "train" => {
                MnistDataset::download_file(TRAIN_IMAGES, &split_dir);
                MnistDataset::download_file(TRAIN_LABELS, &split_dir);
            }
            "test" => {
                MnistDataset::download_file(TEST_IMAGES, &split_dir);
                MnistDataset::download_file(TEST_LABELS, &split_dir);
            }
            _ => panic!("Invalid split specified {split}"),
        };

        split_dir
    }

    /// Download a file from the MNIST dataset URL to the destination directory.
    /// File download progress is reported with the help of a [progress bar](indicatif).
    fn download_file<P: AsRef<Path>>(name: &str, dest_dir: &P) -> PathBuf {
        // Output file name
        let file_name = dest_dir.as_ref().join(name);

        if !file_name.exists() {
            // Download gzip file
            let bytes = download_file_as_bytes(&format!("{URL}{name}.gz"), name);

            // Create file to write the downloaded content to
            let mut output_file = File::create(&file_name).unwrap();

            // Decode gzip file content and write to disk
            let mut gz_buffer = GzDecoder::new(&bytes[..]);
            std::io::copy(&mut gz_buffer, &mut output_file).unwrap();
        }

        file_name
    }

    /// Read images at the provided path for the specified split.
    /// Each image is a vector of bytes.
    fn read_images<P: AsRef<Path>>(root: &P, split: &str) -> Vec<Vec<u8>> {
        let file_name = if split == "train" {
            TRAIN_IMAGES
        } else {
            TEST_IMAGES
        };
        let file_name = root.as_ref().join(file_name);

        // Read number of images from 16-byte header metadata
        let mut f = File::open(file_name).unwrap();
        let mut buf = [0u8; 4];
        let _ = f.seek(SeekFrom::Start(4)).unwrap();
        f.read_exact(&mut buf)
            .expect("Should be able to read image file header");
        let size = u32::from_be_bytes(buf);

        // The item count comes from the file, so it cannot be trusted to size an allocation:
        // a corrupt or hostile header would otherwise reserve an arbitrary amount of memory
        // before a single payload byte is read. Bound it by the bytes the file can supply.
        let expected_len = (WIDTH * HEIGHT) as u64 * size as u64;
        let available = f.metadata().unwrap().len().saturating_sub(16);
        assert!(
            expected_len <= available,
            "image file declares {size} items ({expected_len} bytes), but only {available} bytes \
             follow the header"
        );

        let mut buf_images: Vec<u8> = vec![0u8; expected_len as usize];
        let _ = f.seek(SeekFrom::Start(16)).unwrap();
        f.read_exact(&mut buf_images)
            .expect("Should be able to read image file header");

        buf_images
            .chunks(WIDTH * HEIGHT)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    /// Read labels at the provided path for the specified split.
    fn read_labels<P: AsRef<Path>>(root: &P, split: &str) -> Vec<u8> {
        let file_name = if split == "train" {
            TRAIN_LABELS
        } else {
            TEST_LABELS
        };
        let file_name = root.as_ref().join(file_name);

        // Read number of labels from 8-byte header metadata
        let mut f = File::open(file_name).unwrap();
        let mut buf = [0u8; 4];
        let _ = f.seek(SeekFrom::Start(4)).unwrap();
        f.read_exact(&mut buf)
            .expect("Should be able to read label file header");
        let size = u32::from_be_bytes(buf);

        // Same reasoning as `read_images`: the count is file-supplied, so bound the buffer by
        // what the file actually holds after its 8-byte header.
        let available = f.metadata().unwrap().len().saturating_sub(8);
        assert!(
            size as u64 <= available,
            "label file declares {size} labels, but only {available} bytes follow the header"
        );

        let mut buf_labels: Vec<u8> = vec![0u8; size as usize];
        let _ = f.seek(SeekFrom::Start(8)).unwrap();
        f.read_exact(&mut buf_labels)
            .expect("Should be able to read labels from file");

        buf_labels
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write an IDX image file whose header declares `declared_count` items while only
    /// `payload_bytes` bytes follow the 16-byte header.
    fn write_train_images(dir: &Path, declared_count: u32, payload_bytes: usize) {
        let mut f = File::create(dir.join(TRAIN_IMAGES)).unwrap();
        f.write_all(&[0, 0, 0x08, 0x03]).unwrap(); // unsigned byte, 3 dimensions
        f.write_all(&declared_count.to_be_bytes()).unwrap();
        f.write_all(&(WIDTH as u32).to_be_bytes()).unwrap();
        f.write_all(&(HEIGHT as u32).to_be_bytes()).unwrap();
        f.write_all(&vec![0u8; payload_bytes]).unwrap();
    }

    /// The item count comes out of the file, so a header claiming more items than the file
    /// could possibly hold has to be rejected before anything is allocated from it.
    ///
    /// Before the bound existed, the `u32::MAX` case below asked for ~3.3 TiB; the resulting
    /// allocation failure aborts the process, which no assertion can observe.
    #[test]
    #[should_panic(expected = "bytes follow the header")]
    fn read_images_rejects_a_count_the_file_cannot_hold() {
        let dir = tempfile::tempdir().unwrap();
        write_train_images(dir.path(), u32::MAX, 4);

        let _ = MnistDataset::read_images(&dir.path(), "train");
    }

    /// Same shape for labels: the count is file-supplied and the header is only 8 bytes.
    #[test]
    #[should_panic(expected = "bytes follow the header")]
    fn read_labels_rejects_a_count_the_file_cannot_hold() {
        let dir = tempfile::tempdir().unwrap();
        let mut f = File::create(dir.path().join(TRAIN_LABELS)).unwrap();
        f.write_all(&[0, 0, 0x08, 0x01]).unwrap(); // unsigned byte, 1 dimension
        f.write_all(&u32::MAX.to_be_bytes()).unwrap();
        f.write_all(&[0u8; 4]).unwrap();
        drop(f);

        let _ = MnistDataset::read_labels(&dir.path(), "train");
    }

    /// The bound must not reject a well-formed file.
    #[test]
    fn read_images_accepts_a_well_formed_file() {
        let dir = tempfile::tempdir().unwrap();
        write_train_images(dir.path(), 2, WIDTH * HEIGHT * 2);

        let images = MnistDataset::read_images(&dir.path(), "train");
        assert_eq!(images.len(), 2);
        assert_eq!(images[0].len(), WIDTH * HEIGHT);
    }
}
