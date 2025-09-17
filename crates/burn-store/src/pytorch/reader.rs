//! PyTorch file reader implementation.
//!
//! This module provides support for reading PyTorch checkpoint files (.pt/.pth).
//!
//! # Supported Formats
//!
//! ## 1. Modern ZIP Format (PyTorch 1.6+)
//! Files are ZIP archives containing:
//! - `data.pkl` or `archive/data.pkl`: Pickled tensor metadata
//! - `data/` directory: Binary tensor data files
//!
//! ## 2. Legacy Pickle Format (PyTorch 0.1.10 - 1.5)
//! Sequential pickle streams with the structure:
//! - Magic number pickle (0x1950a86a20f9469cfc6c)
//! - Protocol version pickle (e.g., 1001)
//! - System info pickle (endianness, type sizes)
//! - Model data pickle (state_dict or full model)
//!
//! ## 3. Simple Pickle Format
//! Direct pickle file with a dictionary at the root, commonly used for
//! manually saved state_dicts.
//!
//! # Compatibility
//!
//! The reader handles backward compatibility by detecting the file format
//! automatically. Files from PyTorch 0.1.10 through current versions are
//! supported, though full model saves (vs state_dict) may have limitations
//! as they contain Python code references.

use crate::TensorSnapshot;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use burn_tensor::TensorData;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
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
            Error::Pickle(e) => write!(
                f,
                "Pickle parsing error: {}. This may indicate an unsupported PyTorch file format or corrupted file.",
                e
            ),
            Error::Zip(e) => write!(f, "Zip archive error: {}", e),
            Error::InvalidFormat(msg) => write!(f, "Invalid PyTorch file format: {}", msg),
            Error::KeyNotFound(key) => write!(
                f,
                "Key '{}' not found in PyTorch file. Available keys may be listed with the keys() method.",
                key
            ),
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
                "No data.pkl file found in ZIP archive. Expected PyTorch 1.6+ format with data.pkl or archive/data.pkl".to_string(),
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
    let mut file = File::open(path)?;

    // Check for PyTorch legacy format (starts with magic number as pickled integer)
    let mut header = [0u8; 15];
    // Use read() instead of read_exact() to handle files smaller than 15 bytes
    let bytes_read = file.read(&mut header)?;
    file.seek(std::io::SeekFrom::Start(0))?;

    // Only check for legacy format if we have enough bytes
    let is_legacy_format = bytes_read >= 15
        && header[0] == 0x80
        && header[1] == 0x02
        && header[2] == 0x8a
        && header[3] == 0x0a
        && header[14] == 0x2e;

    // PyTorch legacy format detection (PyTorch 0.1.10 - 1.3)
    // These files use sequential pickle streams with metadata before the actual data.
    // Format structure:
    //   1. Magic number (0x1950a86a20f9469cfc6c) stored as LONG1 pickle
    //   2. Protocol version (e.g., 1001)
    //   3. System info dict (protocol_version, little_endian, type_sizes)
    //   4. Actual model data (state_dict or full model)
    //   5. Storage keys list (pickle)
    //   6. Raw binary data for each storage
    //
    // The pattern is: 0x80 0x02 0x8a 0x0a (PROTO 2, LONG1 with 10 bytes)
    // followed by 10 bytes of magic number, then 0x2e (STOP)
    if is_legacy_format {
        return load_legacy_pytorch_file(path, top_level_key);
    }

    // Standard pickle file
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
                            "Top-level key '{}' not found or is not a dictionary. Available top-level keys in file: {:?}",
                            key,
                            dict.keys().collect::<Vec<_>>()
                        )));
                    }
                }
            } else {
                dict
            }
        }
        _ => {
            return Err(Error::InvalidFormat(
                "Expected a dictionary at the root of the PyTorch file, but found a different type. The file may be a full model save rather than a state_dict.".to_string(),
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

/// Load a legacy PyTorch file with embedded storage data
fn load_legacy_pytorch_file(
    path: &Path,
    top_level_key: Option<&str>,
) -> Result<HashMap<String, TensorSnapshot>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Skip metadata pickles
    // 1. Magic number
    let _ = read_pickle(&mut reader)?;

    // 2. Protocol version
    let _ = read_pickle(&mut reader)?;

    // 3. System info
    let _ = read_pickle(&mut reader)?;

    // Save position before main pickle
    let main_pickle_pos = reader.stream_position()?;

    // 4. Skip main object for now
    let _ = read_pickle(&mut reader)?;

    // 5. Storage keys list (sorted keys as written by PyTorch)
    let storage_keys = match read_pickle(&mut reader) {
        Ok(Object::List(keys)) => keys
            .into_iter()
            .filter_map(|obj| match obj {
                Object::String(s) => Some(s),
                _ => None,
            })
            .collect::<Vec<_>>(),
        _ => vec![],
    };

    // 6. Raw binary data starts here
    let data_start_pos = reader.stream_position()?;
    let file_size = reader.seek(SeekFrom::End(0))?;
    let data_size = file_size - data_start_pos;

    // Read all storage data
    reader.seek(SeekFrom::Start(data_start_pos))?;
    let mut all_storage_data = vec![0u8; data_size as usize];
    reader.read_exact(&mut all_storage_data)?;

    // Create storage map
    // In PyTorch legacy format, all storage data is concatenated.
    // Each tensor references a storage by key and has an offset within that storage.
    // Since we can't determine exact storage boundaries without torch, we give
    // each unique storage access to all the data. The storage_offset in each
    // tensor will handle finding the correct position.
    let mut storage_map = HashMap::new();
    let unique_keys: std::collections::HashSet<_> = storage_keys.iter().cloned().collect();

    // This approach uses more memory but is correct
    for key in unique_keys {
        storage_map.insert(format!("data/{}", key), all_storage_data.clone());
    }

    // Now re-read the main pickle with storage data available
    reader.seek(SeekFrom::Start(main_pickle_pos))?;
    let main_obj = read_pickle_with_data(&mut reader, &storage_map)?;

    // Extract tensors normally
    extract_tensors_with_data(main_obj, top_level_key)
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
