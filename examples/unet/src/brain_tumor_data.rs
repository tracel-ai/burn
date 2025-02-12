use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::data::dataset::InMemDataset;
use burn::prelude::Backend;
use burn::tensor::Float;
use burn::tensor::Int;
use burn::tensor::Tensor;
use burn::tensor::TensorData;
use image::error::ParameterError;
use image::error::ParameterErrorKind;
use image::DynamicImage;
use image::ImageBuffer;
use image::ImageError;
use image::ImageFormat;
use image::Rgb;
use image::RgbImage;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::path::Path;
use std::path::PathBuf;

// height and width of the images used in training
pub const WIDTH: usize = 256;
pub const HEIGHT: usize = 256;
const TRAINING_DATA_DIRECTORY_STR: &str = "data1";

// fn load_image_paths(path: &Path, output_vec: &mut Vec<Box<Path>>) -> Result<(), std::io::Error> {
//     let supported_extensions = ["png", "jpg", "jpeg", "bmp", "tiff"];

//     for entry in std::fs::read_dir(path)? {
//         if let Ok(entry) = entry {
//             let path = entry.path();
//             if path.is_file()
//                 && path
//                     .extension()
//                     .and_then(|ext| ext.to_str())
//                     .map_or(false, |ext_str| {
//                         supported_extensions
//                             .iter()
//                             .any(|&supported_ext| supported_ext.eq_ignore_ascii_case(ext_str))
//                     })
//             {
//                 output_vec.push(path.into_boxed_path());
//             }
//         }
//     }

//     output_vec.sort();
//     Ok(())
// }

fn load_image_paths(path: &Path, output_vec: &mut Vec<Box<Path>>) -> Result<(), std::io::Error> {
    let supported_extensions = ["png", "jpg", "jpeg", "bmp", "tiff"];

    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .and_then(|ext| ext.to_str())
                .filter(|ext_str| {
                    supported_extensions
                        .iter()
                        .any(|&supported_ext| supported_ext.eq_ignore_ascii_case(ext_str))
                })
                .is_some()
        {
            output_vec.push(path.into_boxed_path());
        }
    }

    output_vec.sort();
    Ok(())
}

// Define an enum for supported channel configurations
#[derive(Debug, Clone, Copy)]
enum ChannelConfig {
    SingleToRGB, // 1 channel input -> 3 channel output
    RGBToSingle, // 3 channel input -> 1 channel output
}

impl ChannelConfig {
    fn from_input_channels(channels: usize) -> Result<Self, ImageError> {
        match channels {
            1 => Ok(ChannelConfig::SingleToRGB),
            3 => Ok(ChannelConfig::RGBToSingle),
            _ => Err(ImageError::Parameter(ParameterError::from_kind(
                ParameterErrorKind::Generic("Unsupported number of channels".to_string()),
            ))),
        }
    }

    fn output_repeat(&self) -> usize {
        match self {
            ChannelConfig::SingleToRGB => 3,
            ChannelConfig::RGBToSingle => 1,
        }
    }
}

pub fn save_image<B: Backend, Q: AsRef<Path>>(
    image_tensor: Tensor<B, 4, Float>,
    path: Q,
    format: ImageFormat,
) -> Result<(), ImageError> {
    let width = image_tensor.dims()[3] as u32;
    let height = image_tensor.dims()[2] as u32;

    // Determine channel configuration based on input tensor
    let channel_config = ChannelConfig::from_input_channels(image_tensor.dims()[1])?;

    // Assume batch_size is 1, only saves a single image
    // ---Scale the input tensor from f32 (0.0-1.0) to (0.0-255.0) for casting to u8 in vec.
    let image_tensor_0: Tensor<B, 3, Float> = image_tensor
        .slice([Some((0, 1)), None, None, None])
        .squeeze(0)
        .clamp(0.0, 255.0)
        //.mul_scalar(255.0) // class 0 -> 0.0, class 1.0 -> 255.0
        .round();

    let image: Vec<u8> = image_tensor_0.into_data().iter::<u8>().collect::<Vec<u8>>();
    // Apply channel transformation based on configuration
    let image: Vec<u8> = image
        .into_iter()
        .flat_map(|n| std::iter::repeat(n).take(channel_config.output_repeat()))
        .collect();

    let image_buf: ImageBuffer<Rgb<u8>, Vec<u8>> = RgbImage::from_vec(width, height, image)
        .ok_or_else(|| {
            ImageError::Parameter(ParameterError::from_kind(ParameterErrorKind::Generic(
                "Failed to create image buffer".to_string(),
            )))
        })?;

    if image_buf.is_empty() {
        return Err(ImageError::Parameter(ParameterError::from_kind(
            ParameterErrorKind::Generic("Image buffer is empty".to_string()),
        )));
    }

    image_buf.save_with_format(path, format)
}

// Define a custom error type for better error handling
#[derive(Debug)]
pub enum DatasetError {
    IoError(std::io::Error),
    InvalidDirectory(String),
    ImageError(image::ImageError),
    Other(String),
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetError::IoError(e) => write!(f, "IO error: {}", e),
            DatasetError::InvalidDirectory(msg) => write!(f, "Invalid directory: {}", msg),
            DatasetError::ImageError(e) => write!(f, "ImageError: {}", e),
            DatasetError::Other(msg) => write!(f, "Unknown error: {}", msg),
        }
    }
}

impl From<std::io::Error> for DatasetError {
    fn from(error: std::io::Error) -> Self {
        DatasetError::IoError(error)
    }
}

impl From<image::ImageError> for DatasetError {
    fn from(err: image::ImageError) -> Self {
        DatasetError::ImageError(err)
    }
}

#[derive(Debug, Clone)]
pub struct BrainTumorItem {
    pub source_image_vec: Vec<f32>,
    pub target_mask_vec: Vec<u8>,
}

pub fn source_dynamic_image_to_vector(img: &DynamicImage) -> Vec<f32> {
    // Ensure the image dimensions match WIDTH and HEIGHT
    assert_eq!(
        img.width(),
        WIDTH as u32,
        "Image width does not match WIDTH constant"
    );
    assert_eq!(
        img.height(),
        HEIGHT as u32,
        "Image height does not match HEIGHT constant"
    );

    // Convert to RGB if not already
    let rgb_img = match img {
        DynamicImage::ImageRgb8(rgb_img) => rgb_img.clone(),
        _ => img.to_rgb8(),
    };

    // Create array to store result
    let mut result: Vec<f32> = Vec::<f32>::with_capacity(HEIGHT * WIDTH * 3);

    // Iterate over pixels and fill the vector
    for x in 0..WIDTH {
        for y in 0..HEIGHT {
            let pixel = rgb_img.get_pixel(x as u32, y as u32);
            result.push(pixel[0] as f32 / 255.0);
            result.push(pixel[1] as f32 / 255.0);
            result.push(pixel[2] as f32 / 255.0);
        }
    }
    result
}

pub fn target_dynamic_image_to_vector(img: &DynamicImage) -> Vec<u8> {
    // Ensure the image dimensions match WIDTH and HEIGHT
    assert_eq!(
        img.width(),
        WIDTH as u32,
        "Image width does not match WIDTH constant"
    );
    assert_eq!(
        img.height(),
        HEIGHT as u32,
        "Image height does not match HEIGHT constant"
    );

    // Convert to RGB if not already
    let rgb_img = match img {
        DynamicImage::ImageRgb8(rgb_img) => rgb_img.clone(),
        _ => img.to_rgb8(),
    };

    // Create array to store result
    let mut result: Vec<u8> = Vec::<u8>::with_capacity(HEIGHT * WIDTH * 3);

    // Iterate over pixels and fill the vector
    // This particular dataset has only two classes:
    // black (0,0,0) -> class 0
    // white (255,255,255) -> class 1
    for x in 0..WIDTH {
        for y in 0..HEIGHT {
            let pixel = rgb_img.get_pixel(x as u32, y as u32);
            result.push(pixel[0] / 255_u8);
        }
    }
    result
}

pub struct BrainTumorDataset {
    dataset: InMemDataset<BrainTumorItem>,
}

impl Dataset<BrainTumorItem> for BrainTumorDataset {
    fn get(&self, index: usize) -> Option<BrainTumorItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl BrainTumorDataset {
    /// Creates a new training dataset.
    pub fn train() -> Result<Self, DatasetError> {
        let (images_data_dir, masks_data_dir) = Self::data_dirs()?;
        Self::new(images_data_dir.as_ref(), masks_data_dir.as_ref(), "train")
    }

    /// Creates a new test dataset.
    pub fn test() -> Result<Self, DatasetError> {
        let (images_data_dir, masks_data_dir) = Self::data_dirs()?;
        Self::new(images_data_dir.as_ref(), masks_data_dir.as_ref(), "test")
    }

    /// Creates a new validation dataset.
    pub fn valid() -> Result<Self, DatasetError> {
        let (images_data_dir, masks_data_dir) = Self::data_dirs()?;
        Self::new(images_data_dir.as_ref(), masks_data_dir.as_ref(), "valid")
    }

    /// Helper function to create the data directories.
    fn data_dirs() -> Result<(PathBuf, PathBuf), DatasetError> {
        let images_data_dir: PathBuf = Path::new(TRAINING_DATA_DIRECTORY_STR).join("images");
        let masks_data_dir: PathBuf = Path::new(TRAINING_DATA_DIRECTORY_STR).join("masks");
        Ok((images_data_dir, masks_data_dir))
    }

    fn new(source_dir: &Path, target_dir: &Path, split: &str) -> Result<Self, DatasetError> {
        // Verify source directory
        if !source_dir.exists() {
            return Err(DatasetError::InvalidDirectory(format!(
                "Source directory {:?} does not exist.",
                source_dir
            )));
        }
        if !source_dir.is_dir() {
            return Err(DatasetError::InvalidDirectory(format!(
                "Source path {:?} is not a directory.",
                source_dir
            )));
        }

        // Verify target directory
        if !target_dir.exists() {
            return Err(DatasetError::InvalidDirectory(format!(
                "Target directory {:?} does not exist.",
                target_dir
            )));
        }
        if !target_dir.is_dir() {
            return Err(DatasetError::InvalidDirectory(format!(
                "Target path {:?} is not a directory.",
                target_dir
            )));
        }

        // Initialize the vector to hold the image paths
        let mut source_paths: Vec<Box<Path>> = Vec::new();
        // Call the function with the directory path and the mutable vector
        load_image_paths(source_dir, &mut source_paths)?;

        // Initialize the vector to hold the image paths
        let mut target_paths: Vec<Box<Path>> = Vec::new();
        // Call the function with the directory path and the mutable vector
        load_image_paths(target_dir, &mut target_paths)?;

        // Optional: Verify that the number of source and target images match
        if source_paths.len() != target_paths.len() {
            return Err(DatasetError::InvalidDirectory(
                "The number of source images does not match the number of target images."
                    .to_string(),
            ));
        }

        // Create indices and shuffle them
        let mut indices: Vec<usize> = (0..source_paths.len()).collect();
        let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility
        indices.shuffle(&mut rng);

        // Calculate split sizes
        let total_size = indices.len();
        let train_size = (total_size as f32 * 0.8) as usize;
        let val_size = (total_size as f32 * 0.1) as usize;

        // Get indices for the requested split
        let selected_indices: Vec<usize> = match split.to_lowercase().as_str() {
            "train" => indices[..train_size].to_vec(),
            "valid" => indices[train_size..train_size + val_size].to_vec(),
            "test" => indices[train_size + val_size..].to_vec(),
            _ => {
                return Err(DatasetError::InvalidDirectory(
                    "Invalid split specified. Use 'train', 'valid', or 'test'.".to_string(),
                ))
            }
        };

        let filtered_source_paths: Vec<&Path> = selected_indices
            .iter()
            .map(|&i| &*source_paths[i]) // Dereference the Box<Path> to get a &Path
            .collect();

        let filtered_target_paths: Vec<&Path> = selected_indices
            .iter()
            .map(|&i| &*target_paths[i]) // Dereference the Box<Path> to get a &Path
            .collect();

        // Create iterator over pairs of source and target paths
        let items: Vec<BrainTumorItem> = filtered_source_paths
            .into_iter()
            .zip(filtered_target_paths)
            .map(|(source_path, target_path)| {
                let source_image: DynamicImage = image::open(source_path)?;
                let target_mask: DynamicImage = image::open(target_path)?;
                Ok(BrainTumorItem {
                    source_image_vec: source_dynamic_image_to_vector(&source_image),
                    target_mask_vec: target_dynamic_image_to_vector(&target_mask),
                })
            })
            .collect::<Result<Vec<_>, DatasetError>>()?;

        let dataset = InMemDataset::new(items);

        Ok(Self { dataset })
    }
}

/// batcher struct
#[derive(Clone)]
pub struct BrainTumorBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> BrainTumorBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

/// Represents a batch of image pairs
#[derive(Debug, Clone)]
pub struct BrainTumorBatch<B: Backend> {
    pub source_tensor: Tensor<B, 4, Float>,
    pub target_tensor: Tensor<B, 4, Int>,
}

impl<B: Backend> Batcher<BrainTumorItem, BrainTumorBatch<B>> for BrainTumorBatcher<B> {
    fn batch(&self, items: Vec<BrainTumorItem>) -> BrainTumorBatch<B> {
        let mut sources: Vec<Tensor<B, 3, Float>> = Vec::with_capacity(items.len());
        let mut targets: Vec<Tensor<B, 3, Int>> = Vec::with_capacity(items.len());
        for item in items {
            let a: Box<[f32]> = item.source_image_vec.into_boxed_slice();
            let d = TensorData::from(&*a).convert::<B::FloatElem>();
            let u: Tensor<B, 1, Float> = Tensor::from_data(d, &self.device);
            let u: Tensor<B, 3, Float> = u.reshape([WIDTH, HEIGHT, 3]).swap_dims(0, 2);
            sources.push(u);

            let a: Box<[u8]> = item.target_mask_vec.into_boxed_slice();
            let d = TensorData::from(&*a).convert::<B::IntElem>();
            let u: Tensor<B, 1, Int> = Tensor::from_data(d, &self.device);
            let u: Tensor<B, 3, Int> = u.reshape([WIDTH, HEIGHT, 1]).swap_dims(0, 2);
            targets.push(u);
        }

        let source_tensor: Tensor<B, 4, Float> = Tensor::stack(sources, 0).to_device(&self.device);
        let target_tensor: Tensor<B, 4, Int> = Tensor::stack(targets, 0).to_device(&self.device);

        BrainTumorBatch {
            source_tensor,
            target_tensor,
        }
    }
}
