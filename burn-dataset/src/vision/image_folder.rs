use crate::transform::{Mapper, MapperDataset};
use crate::{Dataset, InMemDataset};

use globwalk::{self, DirEntry};
use image::{self, ColorType};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use thiserror::Error;

const SUPPORTED_FILES: [&str; 4] = ["bmp", "jpg", "jpeg", "png"];

/// Image data type.
#[derive(Debug, Clone, PartialEq)]
pub enum PixelDepth {
    /// 8-bit unsigned.
    U8(u8),
    /// 16-bit unsigned.
    U16(u16),
    /// 32-bit floating point.
    F32(f32),
}

impl TryFrom<PixelDepth> for u8 {
    type Error = &'static str;

    fn try_from(value: PixelDepth) -> Result<Self, Self::Error> {
        if let PixelDepth::U8(v) = value {
            Ok(v)
        } else {
            Err("Value is not u8")
        }
    }
}

impl TryFrom<PixelDepth> for u16 {
    type Error = &'static str;

    fn try_from(value: PixelDepth) -> Result<Self, Self::Error> {
        if let PixelDepth::U16(v) = value {
            Ok(v)
        } else {
            Err("Value is not u16")
        }
    }
}

impl TryFrom<PixelDepth> for f32 {
    type Error = &'static str;

    fn try_from(value: PixelDepth) -> Result<Self, Self::Error> {
        if let PixelDepth::F32(v) = value {
            Ok(v)
        } else {
            Err("Value is not f32")
        }
    }
}

/// Image target for different tasks.
#[derive(Debug, Clone, PartialEq)]
pub enum Annotation {
    /// Image-level label.
    Label(usize),
    /// Object bounding boxes.
    BoundingBoxes(Vec<BoundingBox>),
    /// Segmentation mask.
    SegmentationMask(SegmentationMask),
}

/// Segmentation mask annotation.
/// For semantic segmentation, a mask has a single channel (C = 1).
/// For instance segmentation, there may be multiple masks per image (C >= 1).
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentationMask {
    /// Segmentation mask.
    pub mask: Vec<usize>,
}

/// Object detection bounding box annotation.
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
pub struct BoundingBox {
    /// Coordinates.
    pub coords: [f32; 4],

    /// Box class label.
    pub label: usize,
}

/// Image dataset item.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageDatasetItem {
    /// Image as a vector with a valid image type.
    pub image: Vec<PixelDepth>,

    /// Annotation for the image.
    pub annotation: Annotation,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct ImageDatasetItemRaw {
    /// Image path.
    pub image_path: PathBuf,

    /// Image annotation.
    /// The annotation string can be a category name or path to annotation file.
    pub annotation: String,
}

struct PathToImageDatasetItem {
    classes: HashMap<String, usize>,
}

/// Parse the image annotation to the corresponding type.
fn parse_image_annotation(annotation: &str, classes: &HashMap<String, usize>) -> Annotation {
    // TODO: add support for other annotations
    // - [ ] Object bounding boxes
    // - [ ] Segmentation mask
    // For now, only image classification labels are supported.

    // Map class string to label id
    Annotation::Label(*classes.get(annotation).unwrap())
}

impl Mapper<ImageDatasetItemRaw, ImageDatasetItem> for PathToImageDatasetItem {
    /// Convert a raw image dataset item (path-like) to a 3D image array with a target label.
    fn map(&self, item: &ImageDatasetItemRaw) -> ImageDatasetItem {
        let annotation = parse_image_annotation(&item.annotation, &self.classes);

        // Load image from disk
        let image = image::open(&item.image_path).unwrap();

        // Image as Vec<PixelDepth>
        let img_vec = match image.color() {
            ColorType::L8 => image
                .into_luma8()
                .iter()
                .map(|&x| PixelDepth::U8(x))
                .collect(),
            ColorType::La8 => image
                .into_luma_alpha8()
                .iter()
                .map(|&x| PixelDepth::U8(x))
                .collect(),
            ColorType::L16 => image
                .into_luma16()
                .iter()
                .map(|&x| PixelDepth::U16(x))
                .collect(),
            ColorType::La16 => image
                .into_luma_alpha16()
                .iter()
                .map(|&x| PixelDepth::U16(x))
                .collect(),
            ColorType::Rgb8 => image
                .into_rgb8()
                .iter()
                .map(|&x| PixelDepth::U8(x))
                .collect(),
            ColorType::Rgba8 => image
                .into_rgba8()
                .iter()
                .map(|&x| PixelDepth::U8(x))
                .collect(),
            ColorType::Rgb16 => image
                .into_rgb16()
                .iter()
                .map(|&x| PixelDepth::U16(x))
                .collect(),
            ColorType::Rgba16 => image
                .into_rgba16()
                .iter()
                .map(|&x| PixelDepth::U16(x))
                .collect(),
            ColorType::Rgb32F => image
                .into_rgb32f()
                .iter()
                .map(|&x| PixelDepth::F32(x))
                .collect(),
            ColorType::Rgba32F => image
                .into_rgba32f()
                .iter()
                .map(|&x| PixelDepth::F32(x))
                .collect(),
            _ => panic!("Unrecognized image color type"),
        };

        ImageDatasetItem {
            image: img_vec,
            annotation,
        }
    }
}

/// Error type for [ImageFolderDataset](ImageFolderDataset).
#[derive(Error, Debug)]
pub enum ImageLoaderError {
    /// Unknown error.
    #[error("unknown: `{0}`")]
    Unknown(String),

    /// I/O operation error.
    #[error("I/O error: `{0}`")]
    IOError(String),

    /// Invalid file error.
    #[error("Invalid file extension: `{0}`")]
    InvalidFileExtensionError(String),
}

type ImageDatasetMapper =
    MapperDataset<InMemDataset<ImageDatasetItemRaw>, PathToImageDatasetItem, ImageDatasetItemRaw>;

/// A generic dataset to load classification images from disk.
pub struct ImageFolderDataset {
    dataset: ImageDatasetMapper,
}

impl Dataset<ImageDatasetItem> for ImageFolderDataset {
    fn get(&self, index: usize) -> Option<ImageDatasetItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl ImageFolderDataset {
    /// Create an image classification dataset from the root folder.
    ///
    /// # Arguments
    ///
    /// * `root` - Dataset root folder.
    ///
    /// # Returns
    /// A new dataset instance.
    pub fn new_classification<P: AsRef<Path>>(root: P) -> Result<Self, ImageLoaderError> {
        // New dataset containing any of the supported file types
        ImageFolderDataset::new_classification_with(root, &SUPPORTED_FILES)
    }

    /// Create an image classification dataset from the root folder.
    /// The included images are filtered based on the provided extensions.
    ///
    /// # Arguments
    ///
    /// * `root` - Dataset root folder.
    /// * `extensions` - List of allowed extensions.
    ///
    /// # Returns
    /// A new dataset instance.
    pub fn new_classification_with<P, S>(
        root: P,
        extensions: &[S],
    ) -> Result<Self, ImageLoaderError>
    where
        P: AsRef<Path>,
        S: AsRef<str>,
    {
        /// Check if extension is supported.
        fn check_extension<S: AsRef<str>>(extension: &S) -> Result<String, ImageLoaderError> {
            let extension = extension.as_ref();
            if !SUPPORTED_FILES.contains(&extension) {
                Err(ImageLoaderError::InvalidFileExtensionError(
                    extension.to_string(),
                ))
            } else {
                Ok(extension.to_string())
            }
        }

        // Glob all images with extensions
        let walker = globwalk::GlobWalkerBuilder::from_patterns(
            root.as_ref(),
            &[format!(
                "*.{{{}}}", // "*.{ext1,ext2,ext3}
                extensions
                    .iter()
                    .map(check_extension)
                    .collect::<Result<Vec<_>, _>>()?
                    .join(",")
            )],
        )
        .follow_links(true)
        .sort_by(|p1: &DirEntry, p2: &DirEntry| p1.path().cmp(p2.path())) // order by path
        .build()
        .map_err(|err| ImageLoaderError::Unknown(format!("{err:?}")))?
        .filter_map(Result::ok);

        // Get all dataset items
        let mut items = Vec::new();
        let mut classes = HashSet::new();
        for img in walker {
            let image_path = img.path();

            // Label name is represented by the parent folder name
            let label = image_path
                .parent()
                .ok_or_else(|| {
                    ImageLoaderError::IOError("Could not resolve image parent folder".to_string())
                })?
                .file_name()
                .ok_or_else(|| {
                    ImageLoaderError::IOError(
                        "Could not resolve image parent folder name".to_string(),
                    )
                })?
                .to_string_lossy()
                .into_owned();

            classes.insert(label.clone());

            items.push(ImageDatasetItemRaw {
                image_path: image_path.to_path_buf(),
                annotation: label,
            })
        }

        let dataset = InMemDataset::new(items);

        // Class names to index map
        let mut classes = classes.into_iter().collect::<Vec<_>>();
        classes.sort();
        let classes_map: HashMap<_, _> = classes
            .into_iter()
            .enumerate()
            .map(|(idx, cls)| (cls, idx))
            .collect();

        let mapper = PathToImageDatasetItem {
            classes: classes_map,
        };
        let dataset = MapperDataset::new(dataset, mapper);

        Ok(Self { dataset })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const DATASET_ROOT: &str = "tests/data/image_folder";

    #[test]
    pub fn image_folder_dataset() {
        let dataset = ImageFolderDataset::new_classification(DATASET_ROOT).unwrap();

        // Dataset has 3 elements
        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.get(3), None);

        // Dataset elements should be: orange (0), red (1), red (1)
        assert_eq!(dataset.get(0).unwrap().annotation, Annotation::Label(0));
        assert_eq!(dataset.get(1).unwrap().annotation, Annotation::Label(1));
        assert_eq!(dataset.get(2).unwrap().annotation, Annotation::Label(1));
    }

    #[test]
    pub fn image_folder_dataset_filtered() {
        let dataset = ImageFolderDataset::new_classification_with(DATASET_ROOT, &["jpg"]).unwrap();

        // Filtered dataset has 2 elements
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.get(2), None);

        // Dataset elements should be: orange (0), red (1)
        assert_eq!(dataset.get(0).unwrap().annotation, Annotation::Label(0));
        assert_eq!(dataset.get(1).unwrap().annotation, Annotation::Label(1));
    }

    #[test]
    #[should_panic]
    pub fn image_folder_dataset_invalid_extension() {
        // Some invalid file extension
        let _ = ImageFolderDataset::new_classification_with(DATASET_ROOT, &["ico"]).unwrap();
    }

    #[test]
    pub fn pixel_depth_try_into_u8() {
        let val = u8::MAX;
        let pix: u8 = PixelDepth::U8(val).try_into().unwrap();
        assert_eq!(pix, val);
    }

    #[test]
    #[should_panic]
    pub fn pixel_depth_try_into_u8_invalid() {
        let _: u8 = PixelDepth::U16(u8::MAX as u16 + 1).try_into().unwrap();
    }

    #[test]
    pub fn pixel_depth_try_into_u16() {
        let val = u16::MAX;
        let pix: u16 = PixelDepth::U16(val).try_into().unwrap();
        assert_eq!(pix, val);
    }

    #[test]
    #[should_panic]
    pub fn pixel_depth_try_into_u16_invalid() {
        let _: u16 = PixelDepth::F32(u16::MAX as f32).try_into().unwrap();
    }

    #[test]
    pub fn pixel_depth_try_into_f32() {
        let val = f32::MAX;
        let pix: f32 = PixelDepth::F32(val).try_into().unwrap();
        assert_eq!(pix, val);
    }

    #[test]
    #[should_panic]
    pub fn pixel_depth_try_into_f32_invalid() {
        let _: f32 = PixelDepth::U16(u16::MAX).try_into().unwrap();
    }
}
