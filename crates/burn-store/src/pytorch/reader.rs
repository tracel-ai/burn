//! PyTorch file reader implementation.

use crate::TensorSnapshot;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use burn_tensor::TensorData;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use super::pickle_reader::{Error as PickleError, Object, read_pickle, read_pickle_with_data};

/// Error type for PyTorch file operations
#[derive(Debug)]
pub enum Error {
    /// IO error
    Io(std::io::Error),
    /// Pickle parsing error
    Pickle(PickleError),
    /// Zip archive error
    Zip(zip::result::ZipError),
    /// Invalid file format
    InvalidFormat(String),
    /// Key not found
    KeyNotFound(String),
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<PickleError> for Error {
    fn from(e: PickleError) -> Self {
        Error::Pickle(e)
    }
}

impl From<zip::result::ZipError> for Error {
    fn from(e: zip::result::ZipError) -> Self {
        Error::Zip(e)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "IO error: {}", e),
            Error::Pickle(e) => write!(f, "Pickle error: {}", e),
            Error::Zip(e) => write!(f, "Zip error: {}", e),
            Error::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            Error::KeyNotFound(key) => write!(f, "Key not found: {}", key),
        }
    }
}

impl std::error::Error for Error {}

type Result<T> = std::result::Result<T, Error>;

/// PyTorch checkpoint reader
pub struct PytorchReader {
    tensors: HashMap<String, TensorSnapshot>,
}

impl PytorchReader {
    /// Load a PyTorch checkpoint file (.pt or .pth)
    pub fn new(path: &Path) -> Result<Self> {
        let tensors = load_pytorch_file(path)?;
        Ok(Self { tensors })
    }

    /// Load from a reader with an optional top-level key
    pub fn from_reader<R: Read>(reader: R, top_level_key: Option<&str>) -> Result<Self> {
        let tensors = load_from_reader(reader, top_level_key)?;
        Ok(Self { tensors })
    }

    /// Get all tensor names
    pub fn keys(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    /// Get a tensor by name
    pub fn get(&self, name: &str) -> Option<&TensorSnapshot> {
        self.tensors.get(name)
    }

    /// Get all tensors
    pub fn tensors(&self) -> &HashMap<String, TensorSnapshot> {
        &self.tensors
    }

    /// Take ownership of all tensors
    pub fn into_tensors(self) -> HashMap<String, TensorSnapshot> {
        self.tensors
    }

    /// Load with a specific top-level key (e.g., "state_dict")
    pub fn with_top_level_key(path: &Path, key: &str) -> Result<Self> {
        let tensors = load_pytorch_file_with_key(path, Some(key))?;
        Ok(Self { tensors })
    }
}

/// Load a PyTorch file and return the tensors
pub fn load_pytorch_file(path: &Path) -> Result<HashMap<String, TensorSnapshot>> {
    load_pytorch_file_with_key(path, None)
}

/// Load a PyTorch file with an optional top-level key
pub fn load_pytorch_file_with_key(
    path: &Path,
    top_level_key: Option<&str>,
) -> Result<HashMap<String, TensorSnapshot>> {
    // First, try to read as a zip file
    if let Ok(file) = File::open(path)
        && let Ok(mut archive) = zip::ZipArchive::new(BufReader::new(file))
    {
        // PyTorch saves the main data in various locations within the zip
        let mut pickle_data = Vec::new();
        let mut pickle_found = false;

        // Try different common pickle file locations
        let possible_pickle_paths = [
            "data.pkl",
            "archive/data.pkl",
            // Look for any .pkl file in the root or first-level directories
        ];

        for pickle_path in &possible_pickle_paths {
            if archive.by_name(pickle_path).is_ok() {
                let mut pickle_file = archive.by_name(pickle_path)?;
                pickle_file.read_to_end(&mut pickle_data)?;
                pickle_found = true;
                break;
            }
        }

        // If not found in standard locations, search for any .pkl file
        if !pickle_found {
            for i in 0..archive.len() {
                let file = archive.by_index(i)?;
                let name = file.name().to_string();
                drop(file); // Release the borrow

                if name.ends_with("data.pkl") {
                    let mut file = archive.by_index(i)?;
                    file.read_to_end(&mut pickle_data)?;
                    pickle_found = true;
                    break;
                }
            }
        }

        if !pickle_found {
            return Err(Error::InvalidFormat(
                "No pickle file found in zip".to_string(),
            ));
        }

        // Also read the data files for tensor storage
        let mut data_files: HashMap<String, Vec<u8>> = HashMap::new();
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            let name = file.name().to_string();

            // Look for data files - they can be in various locations
            let is_data_file = name.contains("/data/")
                || name.starts_with("data/")
                || name.starts_with("archive/data/");

            if is_data_file && !name.ends_with(".pkl") && !name.ends_with("/") {
                let mut contents = Vec::new();
                file.read_to_end(&mut contents)?;
                data_files.insert(name, contents);
            }
        }

        // Parse the pickle data with data files
        let mut pickle_reader = BufReader::new(pickle_data.as_slice());
        let obj = read_pickle_with_data(&mut pickle_reader, &data_files)?;

        // Extract tensors with their data
        return extract_tensors_with_data(obj, top_level_key);
    }

    // If not a zip or zip reading failed, try reading as a plain pickle file
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let obj = read_pickle(&mut reader)?;
    extract_tensors_with_data(obj, top_level_key)
}

/// Load from a reader
fn load_from_reader<R: Read>(
    reader: R,
    top_level_key: Option<&str>,
) -> Result<HashMap<String, TensorSnapshot>> {
    let mut buf_reader = BufReader::new(reader);
    let obj = read_pickle(&mut buf_reader)?;
    extract_tensors_with_data(obj, top_level_key)
}

/// Extract tensors from a parsed pickle object
fn extract_tensors_with_data(
    obj: Object,
    top_level_key: Option<&str>,
) -> Result<HashMap<String, TensorSnapshot>> {
    let dict = match obj {
        Object::Dict(dict) => {
            if let Some(key) = top_level_key {
                // Extract the nested dictionary if a top-level key is specified
                match dict.get(key) {
                    Some(Object::Dict(nested)) => nested.clone(),
                    _ => {
                        return Err(Error::KeyNotFound(format!(
                            "Top level key '{}' not found or not a dict",
                            key
                        )));
                    }
                }
            } else {
                dict
            }
        }
        _ => {
            return Err(Error::InvalidFormat(
                "Expected a dictionary at the root".to_string(),
            ));
        }
    };

    let mut tensors = HashMap::new();
    extract_tensors_recursive(&Object::Dict(dict), String::new(), &mut tensors);
    Ok(tensors)
}

/// Recursively extract tensors from an object
fn extract_tensors_recursive(
    obj: &Object,
    path: String,
    tensors: &mut HashMap<String, TensorSnapshot>,
) {
    match obj {
        Object::Dict(dict) => {
            for (key, value) in dict {
                let new_path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{}.{}", path, key)
                };
                extract_tensors_recursive(value, new_path, tensors);
            }
        }
        Object::TorchParam(snapshot) => {
            // The TensorSnapshot already contains the data loading closure
            tensors.insert(path, snapshot.clone());
        }
        _ => {}
    }
}

/// High-level function to read all tensors from a PyTorch file
pub fn read_pytorch_file(
    path: &Path,
    top_level_key: Option<&str>,
) -> Result<HashMap<String, TensorSnapshot>> {
    load_pytorch_file_with_key(path, top_level_key)
}

/// Read tensors from a PyTorch file into memory
/// This is a convenience function that materializes all tensor data
pub fn read_pytorch_tensors(
    path: &Path,
    top_level_key: Option<&str>,
) -> Result<HashMap<String, TensorData>> {
    let snapshots = read_pytorch_file(path, top_level_key)?;
    let mut tensors = HashMap::new();

    for (key, snapshot) in snapshots {
        tensors.insert(key, snapshot.to_data());
    }

    Ok(tensors)
}
