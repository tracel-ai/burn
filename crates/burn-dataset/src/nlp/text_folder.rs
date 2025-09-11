use crate::transform::{Mapper, MapperDataset};
use crate::{Dataset, InMemDataset};

use encoding_rs::{GB18030, GBK, UTF_8, UTF_16BE, UTF_16LE};
use globwalk::{self, DirEntry};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use thiserror::Error;

const SUPPORTED_FILES: [&str; 1] = ["txt"];

/// Text data type.
#[derive(Debug, Clone, PartialEq)]
pub struct TextData {
    /// The text content.
    pub text: String,

    /// Original text source.
    pub text_path: String,
}

/// Text dataset item.
#[derive(Debug, Clone, PartialEq)]
pub struct TextDatasetItem {
    /// Text content.
    pub text: TextData,

    /// Label for the text.
    pub label: usize,
}

/// Raw text dataset item.
#[derive(Debug, Clone)]
struct TextDatasetItemRaw {
    /// Text path.
    text_path: PathBuf,

    /// Text label.
    label: String,
}

impl TextDatasetItemRaw {
    fn new<P: AsRef<Path>>(text_path: P, label: String) -> TextDatasetItemRaw {
        TextDatasetItemRaw {
            text_path: text_path.as_ref().to_path_buf(),
            label,
        }
    }
}

struct PathToTextDatasetItem {
    classes: HashMap<String, usize>,
}

/// Parse the text content from file with auto-detection of encoding.
fn parse_text_content(text_path: &PathBuf) -> String {
    // Read raw bytes from disk
    let mut file = fs::File::open(text_path).unwrap();
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).unwrap();

    // Try to detect encoding and decode text
    // First try UTF-8 with BOM
    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) && bytes.len() >= 3 {
        let (result, _, had_errors) = UTF_8.decode(&bytes[3..]);
        if !had_errors {
            return result.into_owned();
        }
    }

    // Try UTF-8 without BOM
    let (result, _, had_errors) = UTF_8.decode(&bytes);
    if !had_errors {
        return result.into_owned();
    }

    // Try UTF-16LE with BOM
    if bytes.starts_with(&[0xFF, 0xFE]) && bytes.len() >= 2 {
        let (result, had_errors) = UTF_16LE.decode_with_bom_removal(&bytes[2..]);
        if !had_errors {
            return result.into_owned();
        }
    }

    // Try UTF-16BE with BOM
    if bytes.starts_with(&[0xFE, 0xFF]) && bytes.len() >= 2 {
        let (result, had_errors) = UTF_16BE.decode_with_bom_removal(&bytes[2..]);
        if !had_errors {
            return result.into_owned();
        }
    }

    // Try GB18030 encoding
    let (result, _, had_errors) = GB18030.decode(&bytes);
    if !had_errors {
        return result.into_owned();
    }

    // Try GBK encoding
    let (result, _, had_errors) = GBK.decode(&bytes);
    if !had_errors {
        return result.into_owned();
    }

    // Default fallback - use from_utf8_lossy for any remaining cases
    String::from_utf8_lossy(&bytes).to_string()
}

impl Mapper<TextDatasetItemRaw, TextDatasetItem> for PathToTextDatasetItem {
    /// Convert a raw text dataset item (path-like) to text content with a target label.
    fn map(&self, item: &TextDatasetItemRaw) -> TextDatasetItem {
        let label = *self.classes.get(&item.label).unwrap();

        // Load text from disk
        let text_content = parse_text_content(&item.text_path);

        let text_data = TextData {
            text: text_content,
            text_path: item.text_path.display().to_string(),
        };

        TextDatasetItem {
            text: text_data,
            label,
        }
    }
}

/// Error type for [TextFolderDataset](TextFolderDataset).
#[derive(Error, Debug)]
pub enum TextLoaderError {
    /// Unknown error.
    #[error("unknown: `{0}`")]
    Unknown(String),

    /// I/O operation error.
    #[error("I/O error: `{0}`")]
    IOError(String),

    /// Invalid file error.
    #[error("Invalid file extension: `{0}`")]
    InvalidFileExtensionError(String),

    /// Encoding error.
    #[error("Encoding error: `{0}`")]
    EncodingError(String),
}

type TextDatasetMapper =
    MapperDataset<InMemDataset<TextDatasetItemRaw>, PathToTextDatasetItem, TextDatasetItemRaw>;

/// A generic dataset to load texts from disk.
pub struct TextFolderDataset {
    dataset: TextDatasetMapper,
}

impl Dataset<TextDatasetItem> for TextFolderDataset {
    fn get(&self, index: usize) -> Option<TextDatasetItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl TextFolderDataset {
    /// Create a text classification dataset from the root folder.
    ///
    /// # Arguments
    ///
    /// * `root` - Dataset root folder.
    ///
    /// # Returns
    /// A new dataset instance.
    pub fn new_classification<P: AsRef<Path>>(root: P) -> Result<Self, TextLoaderError> {
        // New dataset containing any of the supported file types
        TextFolderDataset::new_classification_with(root, &SUPPORTED_FILES)
    }

    /// Create a text classification dataset from the root folder.
    /// The included texts are filtered based on the provided extensions.
    ///
    /// # Arguments
    ///
    /// * `root` - Dataset root folder.
    /// * `extensions` - List of allowed extensions.
    ///
    /// # Returns
    /// A new dataset instance.
    pub fn new_classification_with<P, S>(root: P, extensions: &[S]) -> Result<Self, TextLoaderError>
    where
        P: AsRef<Path>,
        S: AsRef<str>,
    {
        // Glob all texts with extensions
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
        .map_err(|err| TextLoaderError::Unknown(format!("{err:?}")))?
        .filter_map(Result::ok);

        // Get all dataset items
        let mut items = Vec::new();
        let mut classes = HashSet::new();
        for text in walker {
            let text_path = text.path();

            // Label name is represented by the parent folder name
            let label = text_path
                .parent()
                .ok_or_else(|| {
                    TextLoaderError::IOError("Could not resolve text parent folder".to_string())
                })?
                .file_name()
                .ok_or_else(|| {
                    TextLoaderError::IOError(
                        "Could not resolve text parent folder name".to_string(),
                    )
                })?
                .to_string_lossy()
                .into_owned();

            classes.insert(label.clone());

            items.push(TextDatasetItemRaw::new(text_path, label))
        }

        // Sort class names
        let mut classes = classes.into_iter().collect::<Vec<_>>();
        classes.sort();

        Self::with_items(items, &classes)
    }

    /// Create a text classification dataset with the specified items.
    ///
    /// # Arguments
    ///
    /// * `items` - List of dataset items, each item represented by a tuple `(text path, label)`.
    /// * `classes` - Dataset class names.
    ///
    /// # Returns
    /// A new dataset instance.
    pub fn new_classification_with_items<P: AsRef<Path>, S: AsRef<str>>(
        items: Vec<(P, String)>,
        classes: &[S],
    ) -> Result<Self, TextLoaderError> {
        // Parse items and check valid text extension types
        let items = items
            .into_iter()
            .map(|(path, label)| {
                // Map text path and label
                let path = path.as_ref();
                let label = label;

                Self::check_extension(&path.extension().unwrap().to_str().unwrap())?;

                Ok(TextDatasetItemRaw::new(path, label))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Self::with_items(items, classes)
    }

    /// Create a text dataset with the specified items.
    ///
    /// # Arguments
    ///
    /// * `items` - Raw dataset items.
    /// * `classes` - Dataset class names.
    ///
    /// # Returns
    /// A new dataset instance.
    fn with_items<S: AsRef<str>>(
        items: Vec<TextDatasetItemRaw>,
        classes: &[S],
    ) -> Result<Self, TextLoaderError> {
        // NOTE: right now we don't need to validate the supported text files since
        // the method is private. We assume it's already validated.
        let dataset = InMemDataset::new(items);

        // Class names to index map
        let classes = classes.iter().map(|c| c.as_ref()).collect::<Vec<_>>();
        let classes_map: HashMap<_, _> = classes
            .into_iter()
            .enumerate()
            .map(|(idx, cls)| (cls.to_string(), idx))
            .collect();

        let mapper = PathToTextDatasetItem {
            classes: classes_map,
        };
        let dataset = MapperDataset::new(dataset, mapper);

        Ok(Self { dataset })
    }

    /// Check if extension is supported.
    fn check_extension<S: AsRef<str>>(extension: &S) -> Result<String, TextLoaderError> {
        let extension = extension.as_ref();
        if !SUPPORTED_FILES.contains(&extension) {
            Err(TextLoaderError::InvalidFileExtensionError(
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
    use std::path::Path;

    const TEXT_ROOT: &str = "tests/data/text_folder";

    #[test]
    fn test_text_folder_dataset() {
        let dataset = TextFolderDataset::new_classification(TEXT_ROOT).unwrap();

        // Dataset should have 4 elements (2 positive + 2 negative)
        assert_eq!(dataset.len(), 4);
        assert_eq!(dataset.get(4), None);

        // Check that we have items from both classes
        let mut found_positive = false;
        let mut found_negative = false;

        for i in 0..dataset.len() {
            let item = dataset.get(i).unwrap();
            if item.label == 0 {
                found_negative = true;
                // Check that the text content is loaded correctly
                assert!(!item.text.text.is_empty());
                assert!(item.text.text_path.contains("negative"));
            } else if item.label == 1 {
                found_positive = true;
                // Check that the text content is loaded correctly
                assert!(!item.text.text.is_empty());
                assert!(item.text.text_path.contains("positive"));
            }
        }

        // Verify we found items from both classes
        assert!(found_positive);
        assert!(found_negative);
    }

    #[test]
    fn test_text_folder_dataset_with_invalid_extension() {
        // Try to create a dataset with an unsupported extension
        let result = TextFolderDataset::new_classification_with(TEXT_ROOT, &["invalid"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_text_folder_dataset_with_items() {
        // Create the dataset
        let root = Path::new(TEXT_ROOT);
        let items = vec![
            (
                root.join("positive").join("sample1.txt"),
                "positive".to_string(),
            ),
            (
                root.join("negative").join("sample2.txt"),
                "negative".to_string(),
            ),
        ];
        let classes = vec!["positive", "negative"];
        let dataset = TextFolderDataset::new_classification_with_items(items, &classes).unwrap();

        // Dataset should have 2 elements
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.get(2), None);

        // Get items
        let item0 = dataset.get(0).unwrap();
        let item1 = dataset.get(1).unwrap();

        // Check item0
        assert!(compare_item(
            &item0,
            &(
                "This is a positive text sample for testing the text folder dataset functionality."
                    .to_string(),
                0
            )
        ));

        // Check item1
        assert_eq!(item1.label, 1);
        assert!(item1.text.text_path.contains("negative"));
        assert!(compare_item(
            &item1,
            &(
                "另一个负面文本样本，用以确保数据集能够处理同一类别中的多个文件。".to_string(),
                1
            )
        ));
    }

    fn compare_item(item: &TextDatasetItem, target: &(String, usize)) -> bool {
        item.text.text == target.0 && item.label == target.1
    }
}
