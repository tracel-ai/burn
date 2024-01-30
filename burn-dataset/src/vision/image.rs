use crate::transform::{Mapper, MapperDataset};
use crate::{Dataset, InMemDataset};

use globwalk::{self, DirEntry};
use image::{self, ColorType};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

const SUPPORTED_FILES: [&str; 4] = ["bmp", "jpg", "jpeg", "png"];

/// Image data type.
#[derive(Debug, Clone)]
pub enum DataType {
    /// 8-bit unsigned.
    U8(u8),
    /// 16-bit unsigned.
    U16(u16),
    /// 32-bit floating point.
    F32(f32),
}

impl TryFrom<DataType> for u8 {
    type Error = &'static str;

    fn try_from(value: DataType) -> Result<Self, Self::Error> {
        if let DataType::U8(v) = value {
            Ok(v)
        } else {
            Err("Value is not u8")
        }
    }
}

impl TryFrom<DataType> for u16 {
    type Error = &'static str;

    fn try_from(value: DataType) -> Result<Self, Self::Error> {
        if let DataType::U16(v) = value {
            Ok(v)
        } else {
            Err("Value is not u16")
        }
    }
}

impl TryFrom<DataType> for f32 {
    type Error = &'static str;

    fn try_from(value: DataType) -> Result<Self, Self::Error> {
        if let DataType::F32(v) = value {
            Ok(v)
        } else {
            Err("Value is not f32")
        }
    }
}

/// Image target for different tasks.
#[derive(Debug, Clone)]
pub enum ImageTarget {
    /// Image-level label.
    Label(usize),
    /// Object bounding box.
    BoundingBox(BoundingBox),
    /// Segmentation mask.
    SegmentationMask(SegmentationMask),
}

/// Segmentation mask target.
/// For semantic segmentation, a mask has a single channel (C = 1).
/// For instance segmentation, there may be multiple masks per image (C >= 1).
#[derive(Debug, Clone)]
pub struct SegmentationMask {
    /// Segmentation mask.
    pub mask: Vec<usize>,
}

/// Object detection bounding box target.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct BoundingBox {
    /// Coordinates.
    pub coords: [f32; 4],

    /// Box class label.
    pub label: usize,
}

/// Image dataset item.
#[derive(Debug, Clone)]
pub struct ImageDatasetItem {
    /// Image as a vector with a valid image type.
    pub image: Vec<DataType>,

    /// Target for the image.
    pub target: ImageTarget,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct ImageDatasetItemRaw {
    /// Image path.
    pub image_path: PathBuf,

    /// Image target.
    /// The target string can be a category name or path to annotation file.
    pub target: String,
}

struct PathToImageClassificationItem {
    classes: HashMap<String, usize>,
}

impl Mapper<ImageDatasetItemRaw, ImageDatasetItem> for PathToImageClassificationItem {
    /// Convert a raw image dataset item (path-like) to a 3D image array with a target label.
    fn map(&self, item: &ImageDatasetItemRaw) -> ImageDatasetItem {
        // Map class string to label id
        let label = self.classes.get(&item.target).unwrap();

        // Load image from disk
        let image = image::open(&item.image_path).unwrap();

        // Image as Vec<DataType>
        let img_vec = match image.color() {
            ColorType::L8 => image
                .into_luma8()
                .iter()
                .map(|&x| DataType::U8(x))
                .collect(),
            ColorType::La8 => image
                .into_luma_alpha8()
                .iter()
                .map(|&x| DataType::U8(x))
                .collect(),
            ColorType::L16 => image
                .into_luma16()
                .iter()
                .map(|&x| DataType::U16(x))
                .collect(),
            ColorType::La16 => image
                .into_luma_alpha16()
                .iter()
                .map(|&x| DataType::U16(x))
                .collect(),
            ColorType::Rgb8 => image.into_rgb8().iter().map(|&x| DataType::U8(x)).collect(),
            ColorType::Rgba8 => image
                .into_rgba8()
                .iter()
                .map(|&x| DataType::U8(x))
                .collect(),
            ColorType::Rgb16 => image
                .into_rgb16()
                .iter()
                .map(|&x| DataType::U16(x))
                .collect(),
            ColorType::Rgba16 => image
                .into_rgba16()
                .iter()
                .map(|&x| DataType::U16(x))
                .collect(),
            ColorType::Rgb32F => image
                .into_rgb32f()
                .iter()
                .map(|&x| DataType::F32(x))
                .collect(),
            ColorType::Rgba32F => image
                .into_rgba32f()
                .iter()
                .map(|&x| DataType::F32(x))
                .collect(),
            _ => panic!("Unrecognized image color type"),
        };

        ImageDatasetItem {
            image: img_vec,
            target: ImageTarget::Label(*label),
        }
    }
}

type ClassificationDatasetMapper = MapperDataset<
    InMemDataset<ImageDatasetItemRaw>,
    PathToImageClassificationItem,
    ImageDatasetItemRaw,
>;

/// A generic dataset to load classification images from disk.
pub struct ImageFolderDataset {
    dataset: ClassificationDatasetMapper,
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
    pub fn new<P: AsRef<Path>>(root: P) -> Self {
        // New dataset containing any of the supported file types
        ImageFolderDataset::new_with(root, &SUPPORTED_FILES)
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
    pub fn new_with<P, S>(root: P, extensions: &[S]) -> Self
    where
        P: AsRef<Path>,
        S: AsRef<str>,
    {
        fn check_extension<S: AsRef<str>>(extension: &S) -> String {
            let extension = extension.as_ref();
            assert!(SUPPORTED_FILES.contains(&extension));

            extension.to_string()
        }
        // Glob all images with extensions
        let walker = globwalk::GlobWalkerBuilder::from_patterns(
            root.as_ref(),
            &[format!(
                "*.{{{}}}", // "*.{ext1,ext2,ext3}
                extensions
                    .iter()
                    .map(check_extension)
                    .collect::<Vec<_>>()
                    .join(",")
            )],
        )
        .follow_links(true)
        .sort_by(|p1: &DirEntry, p2: &DirEntry| p1.path().cmp(p2.path())) // order by path
        .build()
        .unwrap()
        .into_iter()
        .filter_map(Result::ok);

        // Get all dataset items
        let mut items = Vec::new();
        let mut classes = HashSet::new();
        for img in walker {
            let image_path = img.path();

            // Target name is represented by the parent folder name
            let target = image_path
                .parent()
                .unwrap()
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned();

            classes.insert(target.clone());

            items.push(ImageDatasetItemRaw {
                image_path: image_path.to_path_buf(),
                target,
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

        let mapper = PathToImageClassificationItem {
            classes: classes_map,
        };
        let dataset = MapperDataset::new(dataset, mapper);

        Self { dataset }
    }
}
