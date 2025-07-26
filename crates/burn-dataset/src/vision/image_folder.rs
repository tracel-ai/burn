use crate::transform::{Mapper, MapperDataset};
use crate::{Dataset, InMemDataset};

use globwalk::{self, DirEntry};
use image::{self, ColorType};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

const SUPPORTED_FILES: [&str; 4] = ["bmp", "jpg", "jpeg", "png"];
const BBOX_MIN_NUM_VALUES: usize = 4;

/// Image data type.
#[derive(Debug, Copy, Clone, PartialEq)]
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

/// Annotation type for different tasks.
#[derive(Debug, Clone, PartialEq)]
pub enum Annotation {
    /// Image-level label.
    Label(usize),
    /// Multiple image-level labels.
    MultiLabel(Vec<usize>),
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
    /// Coordinates in [x_min, y_min, width, height] format.
    pub coords: [f32; 4],

    /// Box class label.
    pub label: usize,
}

/// Image dataset item.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageDatasetItem {
    /// Image as a vector with a valid image type.
    pub image: Vec<PixelDepth>,

    /// Original source image width.
    pub image_width: usize,

    /// Original source image height.
    pub image_height: usize,

    /// Annotation for the image.
    pub annotation: Annotation,

    /// Original image source.
    pub image_path: String,
}

/// Raw annotation types.
#[derive(Deserialize, Serialize, Debug, Clone)]
enum AnnotationRaw {
    Label(String),
    MultiLabel(Vec<String>),
    BoundingBoxes(Vec<BoundingBox>),
    SegmentationMask(PathBuf),
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct ImageDatasetItemRaw {
    /// Image path.
    image_path: PathBuf,

    /// Image annotation.
    annotation: AnnotationRaw,
}

impl ImageDatasetItemRaw {
    fn new<P: AsRef<Path>>(image_path: P, annotation: AnnotationRaw) -> ImageDatasetItemRaw {
        ImageDatasetItemRaw {
            image_path: image_path.as_ref().to_path_buf(),
            annotation,
        }
    }
}

struct PathToImageDatasetItem {
    classes: HashMap<String, usize>,
}

fn segmentation_mask_to_vec_usize(mask_path: &PathBuf) -> Vec<usize> {
    // Load image from disk
    let image = image::open(mask_path).unwrap();

    // Image as Vec<PixelDepth>
    // if rgb8 or rgb16, keep only the first channel assuming all channels are the same

    match image.color() {
        ColorType::L8 => image.into_luma8().iter().map(|&x| x as usize).collect(),
        ColorType::L16 => image.into_luma16().iter().map(|&x| x as usize).collect(),
        ColorType::Rgb8 => image
            .into_rgb8()
            .iter()
            .step_by(3)
            .map(|&x| x as usize)
            .collect(),
        ColorType::Rgb16 => image
            .into_rgb16()
            .iter()
            .step_by(3)
            .map(|&x| x as usize)
            .collect(),
        _ => panic!("Unrecognized image color type"),
    }
}

/// Parse the image annotation to the corresponding type.
fn parse_image_annotation(
    annotation: &AnnotationRaw,
    classes: &HashMap<String, usize>,
) -> Annotation {
    // TODO: add support for other annotations
    // - [ ] Object bounding boxes
    // - [x] Segmentation mask
    // For now, only image classification labels and segmentation are supported.

    // Map class string to label id
    match annotation {
        AnnotationRaw::Label(name) => Annotation::Label(*classes.get(name).unwrap()),
        AnnotationRaw::MultiLabel(names) => Annotation::MultiLabel(
            names
                .iter()
                .map(|name| *classes.get(name).unwrap())
                .collect(),
        ),
        AnnotationRaw::SegmentationMask(mask_path) => {
            Annotation::SegmentationMask(SegmentationMask {
                mask: segmentation_mask_to_vec_usize(mask_path),
            })
        }
        AnnotationRaw::BoundingBoxes(v) => Annotation::BoundingBoxes(v.clone()),
    }
}

/// Retrieve all available classes from the COCO JSON
fn parse_coco_classes(
    json: &serde_json::Value,
) -> Result<HashMap<String, usize>, ImageLoaderError> {
    let mut classes = HashMap::new();

    if let Some(json_classes) = json["categories"].as_array() {
        for class in json_classes {
            let id = class["id"]
                .as_u64()
                .ok_or_else(|| ImageLoaderError::ParsingError("Invalid class ID".to_string()))
                .and_then(|v| {
                    usize::try_from(v).map_err(|_| {
                        ImageLoaderError::ParsingError("Class ID out of usize range".to_string())
                    })
                })?;

            let name = class["name"]
                .as_str()
                .filter(|&s| !s.is_empty())
                .ok_or_else(|| ImageLoaderError::ParsingError("Invalid class name".to_string()))?
                .to_string();

            classes.insert(name, id);
        }
    }

    if classes.is_empty() {
        return Err(ImageLoaderError::ParsingError(
            "No classes found in annotations".to_string(),
        ));
    }

    Ok(classes)
}

/// Retrieve annotations from COCO JSON
fn parse_coco_bbox_annotations(
    json: &serde_json::Value,
) -> Result<HashMap<u64, AnnotationRaw>, ImageLoaderError> {
    let mut annotations = HashMap::new();

    if let Some(json_annotations) = json["annotations"].as_array() {
        for annotation in json_annotations {
            let image_id = annotation["image_id"].as_u64().ok_or_else(|| {
                ImageLoaderError::ParsingError("Invalid image ID in annotation".into())
            })?;

            let class_id = annotation["category_id"]
                .as_u64()
                .ok_or_else(|| {
                    ImageLoaderError::ParsingError("Invalid class ID in annotations".to_string())
                })
                .and_then(|v| {
                    usize::try_from(v).map_err(|_| {
                        ImageLoaderError::ParsingError(
                            "Class ID in annotations out of usize range".to_string(),
                        )
                    })
                })?;

            let bbox_coords = annotation["bbox"]
                .as_array()
                .ok_or_else(|| ImageLoaderError::ParsingError("missing bbox array".to_string()))?
                .iter()
                .map(|v| {
                    v.as_f64()
                        .ok_or_else(|| {
                            ImageLoaderError::ParsingError("invalid bbox value".to_string())
                        })
                        .map(|val| val as f32)
                })
                .collect::<Result<Vec<f32>, _>>()?;

            if bbox_coords.len() < BBOX_MIN_NUM_VALUES {
                return Err(ImageLoaderError::ParsingError(format!(
                    "not enough bounding box coordinates in annotation for image {image_id}",
                )));
            }

            let bbox = BoundingBox {
                coords: [
                    bbox_coords[0],
                    bbox_coords[1],
                    bbox_coords[2],
                    bbox_coords[3],
                ],
                label: class_id,
            };

            annotations
                .entry(image_id)
                .and_modify(|entry| {
                    if let AnnotationRaw::BoundingBoxes(bboxes) = entry {
                        bboxes.push(bbox.clone());
                    }
                })
                .or_insert_with(|| AnnotationRaw::BoundingBoxes(vec![bbox]));
        }
    }

    if annotations.is_empty() {
        return Err(ImageLoaderError::ParsingError(
            "no annotations found".to_string(),
        ));
    }

    Ok(annotations)
}

/// Retrieve all available images from the COCO JSON
fn parse_coco_images<P: AsRef<Path>>(
    images_path: &P,
    mut annotations: HashMap<u64, AnnotationRaw>,
    json: &serde_json::Value,
) -> Result<Vec<ImageDatasetItemRaw>, ImageLoaderError> {
    let mut images = Vec::new();
    if let Some(json_images) = json["images"].as_array() {
        for image in json_images {
            let image_id = image["id"].as_u64().ok_or_else(|| {
                ImageLoaderError::ParsingError("Invalid image ID in image list".to_string())
            })?;

            let file_name = image["file_name"]
                .as_str()
                .ok_or_else(|| ImageLoaderError::ParsingError("Invalid image ID".to_string()))?
                .to_string();

            let mut image_path = images_path.as_ref().to_path_buf();
            image_path.push(file_name);

            if !image_path.exists() {
                return Err(ImageLoaderError::IOError(format!(
                    "Image {} not found",
                    image_path.display()
                )));
            }

            let annotation = annotations
                .remove(&image_id)
                .unwrap_or_else(|| AnnotationRaw::BoundingBoxes(Vec::new()));

            images.push(ImageDatasetItemRaw {
                annotation,
                image_path,
            });
        }
    }

    if images.is_empty() {
        return Err(ImageLoaderError::ParsingError(
            "No images found in annotations".to_string(),
        ));
    }

    Ok(images)
}

impl Mapper<ImageDatasetItemRaw, ImageDatasetItem> for PathToImageDatasetItem {
    /// Convert a raw image dataset item (path-like) to a 3D image array with a target label.
    fn map(&self, item: &ImageDatasetItemRaw) -> ImageDatasetItem {
        let annotation = parse_image_annotation(&item.annotation, &self.classes);

        // Load image from disk
        let image = image::open(&item.image_path).unwrap();

        // Save image dimensions for manipulation
        let img_width = image.width() as usize;
        let img_height = image.height() as usize;

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
            image_width: img_width,
            image_height: img_height,
            annotation,
            image_path: item.image_path.display().to_string(),
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

    /// Parsing error.
    #[error("Parsing error: `{0}`")]
    ParsingError(String),
}

type ImageDatasetMapper =
    MapperDataset<InMemDataset<ImageDatasetItemRaw>, PathToImageDatasetItem, ImageDatasetItemRaw>;

/// A generic dataset to load images from disk.
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
        // Glob all images with extensions
        let walker = globwalk::GlobWalkerBuilder::from_patterns(
            root.as_ref(),
            &[format!(
                "*.{{{}}}", // "*.{ext1,ext2,ext3}
                extensions
                    .iter()
                    .map(Self::check_extension)
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

            items.push(ImageDatasetItemRaw::new(
                image_path,
                AnnotationRaw::Label(label),
            ))
        }

        // Sort class names
        let mut classes = classes.into_iter().collect::<Vec<_>>();
        classes.sort();

        Self::with_items(items, &classes)
    }

    /// Create an image classification dataset with the specified items.
    ///
    /// # Arguments
    ///
    /// * `items` - List of dataset items, each item represented by a tuple `(image path, label)`.
    /// * `classes` - Dataset class names.
    ///
    /// # Returns
    /// A new dataset instance.
    pub fn new_classification_with_items<P: AsRef<Path>, S: AsRef<str>>(
        items: Vec<(P, String)>,
        classes: &[S],
    ) -> Result<Self, ImageLoaderError> {
        // Parse items and check valid image extension types
        let items = items
            .into_iter()
            .map(|(path, label)| {
                // Map image path and label
                let path = path.as_ref();
                let label = AnnotationRaw::Label(label);

                Self::check_extension(&path.extension().unwrap().to_str().unwrap())?;

                Ok(ImageDatasetItemRaw::new(path, label))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Self::with_items(items, classes)
    }

    /// Create a multi-label image classification dataset with the specified items.
    ///
    /// # Arguments
    ///
    /// * `items` - List of dataset items, each item represented by a tuple `(image path, labels)`.
    /// * `classes` - Dataset class names.
    ///
    /// # Returns
    /// A new dataset instance.
    pub fn new_multilabel_classification_with_items<P: AsRef<Path>, S: AsRef<str>>(
        items: Vec<(P, Vec<String>)>,
        classes: &[S],
    ) -> Result<Self, ImageLoaderError> {
        // Parse items and check valid image extension types
        let items = items
            .into_iter()
            .map(|(path, labels)| {
                // Map image path and multi-label
                let path = path.as_ref();
                let labels = AnnotationRaw::MultiLabel(labels);

                Self::check_extension(&path.extension().unwrap().to_str().unwrap())?;

                Ok(ImageDatasetItemRaw::new(path, labels))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Self::with_items(items, classes)
    }

    /// Create an image segmentation dataset with the specified items.
    ///
    /// # Arguments
    ///
    /// * `items` - List of dataset items, each item represented by a tuple `(image path, annotation path)`.
    /// * `classes` - Dataset class names.
    ///
    /// # Returns
    /// A new dataset instance.
    pub fn new_segmentation_with_items<P: AsRef<Path>, S: AsRef<str>>(
        items: Vec<(P, P)>,
        classes: &[S],
    ) -> Result<Self, ImageLoaderError> {
        // Parse items and check valid image extension types
        let items = items
            .into_iter()
            .map(|(image_path, mask_path)| {
                // Map image path and segmentation mask path
                let image_path = image_path.as_ref();
                let annotation = AnnotationRaw::SegmentationMask(mask_path.as_ref().to_path_buf());

                Self::check_extension(&image_path.extension().unwrap().to_str().unwrap())?;

                Ok(ImageDatasetItemRaw::new(image_path, annotation))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Self::with_items(items, classes)
    }

    /// Create a COCO detection dataset based on the annotations JSON and image directory.
    ///
    /// # Arguments
    ///
    /// * `annotations_json` - Path to the JSON file containing annotations in COCO format (for
    ///   example instances_train2017.json).
    ///
    /// * `images_path` - Path containing the images matching the annotations JSON.
    ///
    /// # Returns
    /// A new dataset instance.
    pub fn new_coco_detection<A: AsRef<Path>, I: AsRef<Path>>(
        annotations_json: A,
        images_path: I,
    ) -> Result<Self, ImageLoaderError> {
        let file = fs::File::open(annotations_json)
            .map_err(|e| ImageLoaderError::IOError(format!("Failed to open annotations: {e}")))?;
        let json: Value = serde_json::from_reader(file).map_err(|e| {
            ImageLoaderError::ParsingError(format!("Failed to parse annotations: {e}"))
        })?;

        let classes = parse_coco_classes(&json)?;
        let annotations = parse_coco_bbox_annotations(&json)?;
        let items = parse_coco_images(&images_path, annotations, &json)?;
        let dataset = InMemDataset::new(items);
        let mapper = PathToImageDatasetItem { classes };
        let dataset = MapperDataset::new(dataset, mapper);

        Ok(Self { dataset })
    }

    /// Create an image dataset with the specified items.
    ///
    /// # Arguments
    ///
    /// * `items` - Raw dataset items.
    /// * `classes` - Dataset class names.
    ///
    /// # Returns
    /// A new dataset instance.
    fn with_items<S: AsRef<str>>(
        items: Vec<ImageDatasetItemRaw>,
        classes: &[S],
    ) -> Result<Self, ImageLoaderError> {
        // NOTE: right now we don't need to validate the supported image files since
        // the method is private. We assume it's already validated.
        let dataset = InMemDataset::new(items);

        // Class names to index map
        let classes = classes.iter().map(|c| c.as_ref()).collect::<Vec<_>>();
        let classes_map: HashMap<_, _> = classes
            .into_iter()
            .enumerate()
            .map(|(idx, cls)| (cls.to_string(), idx))
            .collect();

        let mapper = PathToImageDatasetItem {
            classes: classes_map,
        };
        let dataset = MapperDataset::new(dataset, mapper);

        Ok(Self { dataset })
    }

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
}

#[cfg(test)]
mod tests {
    use super::*;
    const DATASET_ROOT: &str = "tests/data/image_folder";
    const SEGMASK_ROOT: &str = "tests/data/segmask_folder";
    const COCO_JSON: &str = "tests/data/dataset_coco.json";
    const COCO_IMAGES: &str = "tests/data/image_folder_coco";

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
    pub fn image_folder_dataset_with_items_sizes() {
        let root = Path::new(DATASET_ROOT);
        let items = vec![
            (root.join("orange").join("dot.jpg"), "orange".to_string()),
            (root.join("red").join("dot.jpg"), "red".to_string()),
            (root.join("red").join("dot.png"), "red".to_string()),
        ];
        let dataset =
            ImageFolderDataset::new_classification_with_items(items, &["orange", "red"]).unwrap();

        // Dataset has 3 elements
        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.get(3), None);

        // Test item sizes

        assert_eq!(
            (
                dataset.get(0).unwrap().image_width,
                dataset.get(0).unwrap().image_height
            ),
            (1, 1)
        );
        assert_eq!(
            (
                dataset.get(1).unwrap().image_width,
                dataset.get(1).unwrap().image_height
            ),
            (1, 1)
        );
        assert_eq!(
            (
                dataset.get(2).unwrap().image_width,
                dataset.get(2).unwrap().image_height
            ),
            (1, 1)
        );
    }

    #[test]
    pub fn image_folder_dataset_with_items() {
        let root = Path::new(DATASET_ROOT);
        let items = vec![
            (root.join("orange").join("dot.jpg"), "orange".to_string()),
            (root.join("red").join("dot.jpg"), "red".to_string()),
            (root.join("red").join("dot.png"), "red".to_string()),
        ];
        let dataset =
            ImageFolderDataset::new_classification_with_items(items, &["orange", "red"]).unwrap();

        // Dataset has 3 elements
        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.get(3), None);

        // Dataset elements should be: orange (0), red (1), red (1)
        assert_eq!(dataset.get(0).unwrap().annotation, Annotation::Label(0));
        assert_eq!(dataset.get(1).unwrap().annotation, Annotation::Label(1));
        assert_eq!(dataset.get(2).unwrap().annotation, Annotation::Label(1));
    }

    #[test]
    pub fn image_folder_dataset_multilabel() {
        let root = Path::new(DATASET_ROOT);
        let items = vec![
            (
                root.join("orange").join("dot.jpg"),
                vec!["dot".to_string(), "orange".to_string()],
            ),
            (
                root.join("red").join("dot.jpg"),
                vec!["dot".to_string(), "red".to_string()],
            ),
            (
                root.join("red").join("dot.png"),
                vec!["dot".to_string(), "red".to_string()],
            ),
        ];
        let dataset = ImageFolderDataset::new_multilabel_classification_with_items(
            items,
            &["dot", "orange", "red"],
        )
        .unwrap();

        // Dataset has 3 elements
        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.get(3), None);

        // Dataset elements should be: [dot, orange] (0, 1), [dot, red] (0, 2), [dot, red] (0, 2)
        assert_eq!(
            dataset.get(0).unwrap().annotation,
            Annotation::MultiLabel(vec![0, 1])
        );
        assert_eq!(
            dataset.get(1).unwrap().annotation,
            Annotation::MultiLabel(vec![0, 2])
        );
        assert_eq!(
            dataset.get(2).unwrap().annotation,
            Annotation::MultiLabel(vec![0, 2])
        );
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

    #[test]
    pub fn parse_image_annotation_label_string() {
        let classes = HashMap::from([("0".to_string(), 0_usize), ("1".to_string(), 1_usize)]);
        let anno = AnnotationRaw::Label("0".to_string());
        assert_eq!(
            parse_image_annotation(&anno, &classes),
            Annotation::Label(0)
        );
    }

    #[test]
    pub fn parse_image_annotation_multilabel_string() {
        let classes = HashMap::from([
            ("0".to_string(), 0_usize),
            ("1".to_string(), 1_usize),
            ("2".to_string(), 2_usize),
        ]);
        let anno = AnnotationRaw::MultiLabel(vec!["0".to_string(), "2".to_string()]);
        assert_eq!(
            parse_image_annotation(&anno, &classes),
            Annotation::MultiLabel(vec![0, 2])
        );
    }

    #[test]
    pub fn segmask_image_path_to_vec_usize() {
        let root = Path::new(SEGMASK_ROOT);

        // checkerboard mask
        const TEST_CHECKERBOARD_MASK_PATTERN: [u8; 64] = [
            1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2,
            1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1,
            2, 1, 2, 1, 2, 1,
        ];
        assert_eq!(
            TEST_CHECKERBOARD_MASK_PATTERN
                .iter()
                .map(|&x| x as usize)
                .collect::<Vec<usize>>(),
            segmentation_mask_to_vec_usize(&root.join("annotations").join("mask_checkerboard.png")),
        );

        // random 2 colors mask
        const TEST_RANDOM2COLORS_MASK_PATTERN: [u8; 64] = [
            1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2,
            2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 2, 1,
            1, 1, 1, 1, 1, 1,
        ];
        assert_eq!(
            TEST_RANDOM2COLORS_MASK_PATTERN
                .iter()
                .map(|&x| x as usize)
                .collect::<Vec<usize>>(),
            segmentation_mask_to_vec_usize(
                &root.join("annotations").join("mask_random_2colors.png")
            ),
        );
        // random 3 colors mask
        const TEST_RANDOM3COLORS_MASK_PATTERN: [u8; 64] = [
            3, 1, 3, 3, 1, 1, 3, 2, 3, 3, 3, 3, 1, 3, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 3, 3,
            3, 2, 3, 2, 2, 3, 2, 3, 3, 1, 3, 1, 3, 3, 1, 1, 3, 2, 1, 2, 2, 2, 1, 2, 1, 2, 3, 3, 1,
            3, 3, 2, 1, 2, 2,
        ];
        assert_eq!(
            TEST_RANDOM3COLORS_MASK_PATTERN
                .iter()
                .map(|&x| x as usize)
                .collect::<Vec<usize>>(),
            segmentation_mask_to_vec_usize(
                &root.join("annotations").join("mask_random_3colors.png")
            ),
        );
    }

    #[test]
    pub fn segmask_folder_dataset() {
        let root = Path::new(SEGMASK_ROOT);

        let items = vec![
            (
                root.join("images").join("image_checkerboard.png"),
                root.join("annotations").join("mask_checkerboard.png"),
            ),
            (
                root.join("images").join("image_random_2colors.png"),
                root.join("annotations").join("mask_random_2colors.png"),
            ),
            (
                root.join("images").join("image_random_3colors.png"),
                root.join("annotations").join("mask_random_3colors.png"),
            ),
        ];
        let dataset = ImageFolderDataset::new_segmentation_with_items(
            items,
            &[
                "foo", // 0
                "bar", // 1
                "baz", // 2
                "qux", // 3
            ],
        )
        .unwrap();

        // Dataset has 3 elements; each (image, annotation) is a single item
        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.get(3), None);

        // checkerboard mask
        const TEST_CHECKERBOARD_MASK_PATTERN: [u8; 64] = [
            1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2,
            1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1,
            2, 1, 2, 1, 2, 1,
        ];
        assert_eq!(
            dataset.get(0).unwrap().annotation,
            Annotation::SegmentationMask(SegmentationMask {
                mask: TEST_CHECKERBOARD_MASK_PATTERN
                    .iter()
                    .map(|&x| x as usize)
                    .collect()
            })
        );
        // random 2 colors mask
        const TEST_RANDOM2COLORS_MASK_PATTERN: [u8; 64] = [
            1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2,
            2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 2, 1,
            1, 1, 1, 1, 1, 1,
        ];
        assert_eq!(
            dataset.get(1).unwrap().annotation,
            Annotation::SegmentationMask(SegmentationMask {
                mask: TEST_RANDOM2COLORS_MASK_PATTERN
                    .iter()
                    .map(|&x| x as usize)
                    .collect()
            })
        );
        // random 3 colors mask
        const TEST_RANDOM3COLORS_MASK_PATTERN: [u8; 64] = [
            3, 1, 3, 3, 1, 1, 3, 2, 3, 3, 3, 3, 1, 3, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 3, 3,
            3, 2, 3, 2, 2, 3, 2, 3, 3, 1, 3, 1, 3, 3, 1, 1, 3, 2, 1, 2, 2, 2, 1, 2, 1, 2, 3, 3, 1,
            3, 3, 2, 1, 2, 2,
        ];
        assert_eq!(
            dataset.get(2).unwrap().annotation,
            Annotation::SegmentationMask(SegmentationMask {
                mask: TEST_RANDOM3COLORS_MASK_PATTERN
                    .iter()
                    .map(|&x| x as usize)
                    .collect()
            })
        );
    }

    #[test]
    pub fn coco_detection_dataset() {
        let dataset = ImageFolderDataset::new_coco_detection(COCO_JSON, COCO_IMAGES).unwrap();
        assert_eq!(dataset.len(), 3); // we have only three images defined
        assert_eq!(dataset.get(3), None);

        const TWO_DOTS_AND_TRIANGLE_B1: BoundingBox = BoundingBox {
            coords: [3.125_172, 18.090_784, 10.960_11, 10.740_027],
            label: 0,
        };

        const TWO_DOTS_AND_TRIANGLE_B2: BoundingBox = BoundingBox {
            coords: [3.257_221_5, 3.037_139, 10.563_961, 10.828_06],
            label: 0,
        };

        const TWO_DOTS_AND_TRIANGLE_B3: BoundingBox = BoundingBox {
            coords: [15.097_662, 3.389_271, 12.632_737, 11.180_193],
            label: 1,
        };

        const DOTS_TRIANGLE_B1: BoundingBox = BoundingBox {
            coords: [3.125_172, 17.914_719, 10.828_06, 11.004_127],
            label: 0,
        };

        const DOTS_TRIANGLE_B2: BoundingBox = BoundingBox {
            coords: [15.273_727, 3.301_238, 12.192_573, 11.708_39],
            label: 1,
        };

        const ONE_DOT_B1: BoundingBox = BoundingBox {
            coords: [10.079_78, 9.595_598, 10.960_11, 11.356_258],
            label: 0,
        };

        for item in dataset.iter() {
            let file_name = Path::new(&item.image_path).file_name().unwrap();
            match item.annotation {
                // check if the number of bounding boxes is correct
                Annotation::BoundingBoxes(v) => {
                    if file_name == "two_dots_and_triangle.jpg" {
                        assert_eq!(v.len(), 3);
                        assert!(v.contains(&TWO_DOTS_AND_TRIANGLE_B1));
                        assert!(v.contains(&TWO_DOTS_AND_TRIANGLE_B2));
                        assert!(v.contains(&TWO_DOTS_AND_TRIANGLE_B3));
                    } else if file_name == "dot_triangle.jpg" {
                        assert_eq!(v.len(), 2);
                        assert!(v.contains(&DOTS_TRIANGLE_B1));
                        assert!(v.contains(&DOTS_TRIANGLE_B2));
                    } else if file_name == "one_dot.jpg" {
                        assert_eq!(v.len(), 1);
                        assert!(v.contains(&ONE_DOT_B1));
                    } else {
                        panic!("{}", format!("unexpected image name: {}", item.image_path));
                    }
                }
                _ => panic!("unexpected annotation"),
            }
        }
    }
}
