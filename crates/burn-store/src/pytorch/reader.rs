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
//! ## 2. TAR Format (older torchvision models like AlexNet, SqueezeNet)
//! TAR archives containing:
//! - `sys_info`: System info pickle (endianness, type sizes)
//! - `pickle`: OrderedDict mapping tensor names to storage keys
//! - `tensors`: Tensor metadata (unused, metadata is in pickle)
//! - `storages`: Count pickle + sequential (metadata, num_elements, raw data)
//!
//! ## 3. Legacy Pickle Format (PyTorch 0.1.10 - 1.5)
//! Sequential pickle streams with the structure:
//! - Magic number pickle (0x1950a86a20f9469cfc6c)
//! - Protocol version pickle (e.g., 1001)
//! - System info pickle (endianness, type sizes)
//! - Model data pickle (state_dict or full model)
//!
//! ## 4. Simple Pickle Format
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
use burn_core::record::serde::{adapter::DefaultAdapter, data::NestedValue, de::Deserializer};
use serde::de::DeserializeOwned;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use super::lazy_data::LazyDataSource;
use super::pickle_reader::{Object, PickleError, read_pickle, read_pickle_with_data};
use std::sync::Arc;

/// Error type for PyTorch file operations
#[derive(Debug)]
pub enum PytorchError {
    /// IO error
    Io(std::io::Error),
    /// Pickle parsing error
    Pickle(PickleError),
    /// Zip archive error
    Zip(zip::result::ZipError),
    /// TAR archive error
    Tar(std::io::Error),
    /// Invalid file format
    InvalidFormat(String),
    /// Key not found
    KeyNotFound(String),
    /// Serde deserialization error
    Serde(burn_core::record::serde::error::Error),
}

impl From<std::io::Error> for PytorchError {
    fn from(e: std::io::Error) -> Self {
        PytorchError::Io(e)
    }
}

impl From<PickleError> for PytorchError {
    fn from(e: PickleError) -> Self {
        PytorchError::Pickle(e)
    }
}

impl From<zip::result::ZipError> for PytorchError {
    fn from(e: zip::result::ZipError) -> Self {
        PytorchError::Zip(e)
    }
}

impl From<burn_core::record::serde::error::Error> for PytorchError {
    fn from(e: burn_core::record::serde::error::Error) -> Self {
        PytorchError::Serde(e)
    }
}

impl std::fmt::Display for PytorchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PytorchError::Io(e) => write!(f, "IO error: {}", e),
            PytorchError::Pickle(e) => write!(
                f,
                "Pickle parsing error: {}. This may indicate an unsupported PyTorch file format or corrupted file.",
                e
            ),
            PytorchError::Zip(e) => write!(f, "Zip archive error: {}", e),
            PytorchError::Tar(e) => write!(f, "TAR archive error: {}", e),
            PytorchError::InvalidFormat(msg) => write!(f, "Invalid PyTorch file format: {}", msg),
            PytorchError::KeyNotFound(key) => write!(
                f,
                "Key '{}' not found in PyTorch file. Available keys may be listed with the keys() method.",
                key
            ),
            PytorchError::Serde(e) => write!(f, "Serde deserialization error: {}", e),
        }
    }
}

impl std::error::Error for PytorchError {}

type Result<T> = std::result::Result<T, PytorchError>;

/// Metadata about a PyTorch file
///
/// Contains information about the file format, version, and other properties
/// that can be useful for debugging or compatibility checking.
#[derive(Debug, Clone)]
pub struct PytorchMetadata {
    /// Format version (e.g., "1.0" for modern ZIP format)
    pub format_version: Option<String>,
    /// File format type (ZIP, Legacy, or Pickle)
    pub format_type: FileFormat,
    /// Byte order (endianness) - currently only LittleEndian is supported
    pub byte_order: ByteOrder,
    /// Whether the file has storage alignment information
    pub has_storage_alignment: bool,
    /// PyTorch version that saved the file (if available)
    pub pytorch_version: Option<String>,
    /// Number of tensors in the file
    pub tensor_count: usize,
    /// Total size of tensor data in bytes (if available)
    pub total_data_size: Option<usize>,
}

impl PytorchMetadata {
    /// Check if this is a modern format file (ZIP-based, PyTorch 1.6+)
    pub fn is_modern_format(&self) -> bool {
        matches!(self.format_type, FileFormat::Zip)
    }

    /// Check if this is a legacy format file (PyTorch 0.1.10 - 1.5)
    pub fn is_legacy_format(&self) -> bool {
        matches!(self.format_type, FileFormat::Legacy)
    }
}

/// File format type
#[derive(Debug, Clone, PartialEq)]
pub enum FileFormat {
    /// ZIP-based format (PyTorch 1.6+)
    Zip,
    /// TAR-based format (older torchvision models)
    Tar,
    /// Legacy format (PyTorch 0.1.10 - 1.5)
    Legacy,
    /// Simple pickle file
    Pickle,
}

/// Byte order (endianness)
#[derive(Debug, Clone, PartialEq)]
pub enum ByteOrder {
    LittleEndian,
    BigEndian,
}

/// PyTorch checkpoint reader
///
/// This is the main interface for reading PyTorch checkpoint files (.pt/.pth).
/// It supports multiple PyTorch formats including modern ZIP-based format (1.6+),
/// legacy format (0.1.10-1.5), and simple pickle files.
///
/// # Example
/// ```rust,no_run
/// # use burn_store::pytorch::PytorchReader;
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Load a checkpoint file
/// let reader = PytorchReader::new("model.pt")?;
///
/// // Get tensor names
/// let keys = reader.keys();
///
/// // Access a specific tensor
/// if let Some(tensor) = reader.get("conv1.weight") {
///     let data = tensor.to_data(); // Materializes the tensor
/// }
///
/// // Check file metadata
/// println!("Format: {:?}", reader.metadata().format_type);
/// println!("Tensor count: {}", reader.metadata().tensor_count);
/// # Ok(())
/// # }
/// ```
pub struct PytorchReader {
    tensors: HashMap<String, TensorSnapshot>,
    metadata: PytorchMetadata,
}

impl PytorchReader {
    /// Load a PyTorch checkpoint file
    ///
    /// # Arguments
    /// * `path` - Path to the PyTorch file (.pt or .pth)
    ///
    /// # Returns
    /// A `PytorchReader` with lazy-loaded tensors and metadata
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let (tensors, metadata) = load_pytorch_file_with_metadata(path.as_ref(), None)?;
        Ok(Self { tensors, metadata })
    }

    /// Load a PyTorch checkpoint with a specific top-level key
    ///
    /// Many PyTorch checkpoints store the model weights under a specific key
    /// like "state_dict", "model", or "model_state_dict".
    ///
    /// # Arguments
    /// * `path` - Path to the PyTorch file
    /// * `key` - Top-level key to extract (e.g., "state_dict")
    ///
    /// # Example
    /// ```rust,no_run
    /// # use burn_store::pytorch::PytorchReader;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let reader = PytorchReader::with_top_level_key("checkpoint.pt", "state_dict")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_top_level_key<P: AsRef<Path>>(path: P, key: &str) -> Result<Self> {
        let (tensors, metadata) = load_pytorch_file_with_metadata(path.as_ref(), Some(key))?;
        Ok(Self { tensors, metadata })
    }

    /// Load from a reader
    ///
    /// This method is useful when loading from non-file sources like memory buffers.
    /// Note: Metadata detection is limited when loading from a reader.
    ///
    /// # Arguments
    /// * `reader` - Any type implementing `Read`
    /// * `top_level_key` - Optional key to extract
    pub fn from_reader<R: Read>(reader: R, top_level_key: Option<&str>) -> Result<Self> {
        // For reader-based loading, we don't have full metadata access
        let tensors = load_from_reader(reader, top_level_key)?;
        let metadata = PytorchMetadata {
            format_version: None,
            format_type: FileFormat::Pickle, // Default assumption
            byte_order: ByteOrder::LittleEndian,
            has_storage_alignment: false,
            pytorch_version: None,
            tensor_count: tensors.len(),
            total_data_size: None,
        };
        Ok(Self { tensors, metadata })
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

    /// Get metadata about the loaded file
    ///
    /// Provides information about the file format, version, endianness, etc.
    pub fn metadata(&self) -> &PytorchMetadata {
        &self.metadata
    }

    /// Get the number of tensors in the file
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Check if the file contains no tensors
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Read raw pickle data from a PyTorch file
    ///
    /// This is useful for extracting configuration or metadata that isn't tensor data.
    /// Returns a simplified JSON-like structure that can be easily converted to other formats.
    ///
    /// # Arguments
    /// * `path` - Path to the PyTorch file
    /// * `top_level_key` - Optional key to extract from the top-level dictionary
    ///
    /// # Returns
    /// A `PickleValue` representing the pickle data structure
    pub fn read_pickle_data<P: AsRef<Path>>(
        path: P,
        top_level_key: Option<&str>,
    ) -> Result<PickleValue> {
        read_pickle_as_value(path.as_ref(), top_level_key)
    }

    /// Load and deserialize configuration data from a PyTorch file
    ///
    /// This method reads configuration or metadata stored in PyTorch checkpoint files
    /// and deserializes it into the specified type. It's particularly useful for
    /// extracting model configurations that might be saved alongside model weights.
    ///
    /// # Arguments
    /// * `path` - Path to the PyTorch file (.pt or .pth)
    /// * `top_level_key` - Optional key to extract specific data within the pickle file.
    ///   If `None`, the entire content is deserialized.
    ///
    /// # Type Parameters
    /// * `D` - The target type to deserialize into. Must implement `DeserializeOwned`.
    ///
    /// # Returns
    /// A `Result` containing the deserialized configuration data, or an `Error` if
    /// reading or deserialization fails.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use burn_store::pytorch::PytorchReader;
    /// # use serde::Deserialize;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// #[derive(Debug, Deserialize)]
    /// struct ModelConfig {
    ///     hidden_size: usize,
    ///     num_layers: usize,
    /// }
    ///
    /// let config: ModelConfig = PytorchReader::load_config("model.pth", Some("config"))?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn load_config<D, P>(path: P, top_level_key: Option<&str>) -> Result<D>
    where
        D: DeserializeOwned,
        P: AsRef<Path>,
    {
        // Read the PyTorch file and extract the pickle data
        let pickle_value = Self::read_pickle_data(path, top_level_key)?;

        // Convert PickleValue to NestedValue
        let nested_value = convert_pickle_to_nested_value(pickle_value)?;

        // Create a deserializer with the default adapter
        let deserializer = Deserializer::<DefaultAdapter>::new(nested_value, false);

        // Deserialize the nested value into the target type
        let value = D::deserialize(deserializer)?;
        Ok(value)
    }
}

/// Simplified representation of pickle data
///
/// This enum provides a JSON-like structure that's easier to work with
/// than the internal pickle Object type.
#[derive(Debug, Clone, PartialEq)]
pub enum PickleValue {
    /// None/null value
    None,
    /// Boolean value
    Bool(bool),
    /// Integer value
    Int(i64),
    /// Floating point value
    Float(f64),
    /// String value
    String(String),
    /// List/array of values
    List(Vec<PickleValue>),
    /// Dictionary/map of string keys to values
    Dict(HashMap<String, PickleValue>),
    /// Binary data
    Bytes(Vec<u8>),
}

/// Internal function to load a PyTorch file with metadata
fn load_pytorch_file_with_metadata(
    path: &Path,
    top_level_key: Option<&str>,
) -> Result<(HashMap<String, TensorSnapshot>, PytorchMetadata)> {
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
            return Err(PytorchError::InvalidFormat(
                "No data.pkl file found in ZIP archive. Expected PyTorch 1.6+ format with data.pkl or archive/data.pkl".to_string(),
            ));
        }

        // Check for format version (optional)
        let format_version = if let Ok(mut version_file) = archive.by_name(".format_version") {
            let mut version_data = Vec::new();
            version_file.read_to_end(&mut version_data)?;
            let version_str = String::from_utf8_lossy(&version_data);
            let version = version_str.trim().to_string();
            Some(version)
        } else {
            None
        };

        // Check for byteorder file to detect endianness
        let is_big_endian = if let Ok(mut byteorder_file) = archive.by_name("byteorder") {
            let mut byteorder_data = Vec::new();
            byteorder_file.read_to_end(&mut byteorder_data)?;
            let byteorder_str = String::from_utf8_lossy(&byteorder_data);
            byteorder_str.trim() == "big"
        } else {
            false // Default to little-endian if no byteorder file
        };

        if is_big_endian {
            // Big-endian files are not yet supported as they require different byte order conversion
            // TODO: To support big-endian files, we need to:
            // 1. Pass endianness info through to pickle_reader
            // 2. Use from_be_bytes instead of from_le_bytes for tensor data
            // 3. Handle byte swapping for all numeric types (f32, f64, i32, etc.)
            return Err(PytorchError::InvalidFormat(
                "Big-endian PyTorch files are not yet supported. The file was saved on a big-endian system and requires byte order conversion.".to_string()
            ));
        }

        // Check for storage alignment file
        let has_storage_alignment = archive.by_name(".storage_alignment").is_ok();

        // Check for PyTorch version (if saved)
        let pytorch_version = if let Ok(mut version_file) = archive.by_name("version") {
            let mut version_data = Vec::new();
            version_file.read_to_end(&mut version_data)?;
            Some(String::from_utf8_lossy(&version_data).trim().to_string())
        } else {
            None
        };

        // Create a lazy data source instead of loading all data upfront
        let data_source = Arc::new(LazyDataSource::from_zip(path)?);

        // Calculate total data size without loading
        let mut total_data_size = 0usize;
        for i in 0..archive.len() {
            let file = archive.by_index(i)?;
            let name = file.name();

            // Look for data files - they can be in various locations
            let is_data_file = (name.contains("/data/")
                || name.starts_with("data/")
                || name.starts_with("archive/data/"))
                && !name.ends_with(".pkl")
                && !name.ends_with("/");

            if is_data_file {
                total_data_size += file.size() as usize;
            }
        }

        // Parse the pickle data with lazy data source
        let mut pickle_reader = BufReader::new(pickle_data.as_slice());
        let obj = read_pickle_with_data(&mut pickle_reader, data_source)?;

        // Extract tensors with their data
        let tensors = extract_tensors_with_data(obj, top_level_key)?;

        // Create metadata
        let metadata = PytorchMetadata {
            format_version,
            format_type: FileFormat::Zip,
            byte_order: if is_big_endian {
                ByteOrder::BigEndian
            } else {
                ByteOrder::LittleEndian
            },
            has_storage_alignment,
            pytorch_version,
            tensor_count: tensors.len(),
            total_data_size: Some(total_data_size),
        };

        return Ok((tensors, metadata));
    }

    // If not a zip or zip reading failed, try TAR format
    if is_tar_file(path) {
        return load_tar_pytorch_file_with_metadata(path, top_level_key);
    }

    // Try reading as a plain pickle file
    let mut file = File::open(path)?;

    // Check for PyTorch legacy format (starts with magic number as pickled integer)
    let mut header = [0u8; 15];
    // Use read() instead of read_exact() to handle files smaller than 15 bytes
    let bytes_read = file.read(&mut header)?;
    file.seek(std::io::SeekFrom::Start(0))?;

    // Only check for legacy format if we have enough bytes
    // PyTorch legacy format detection (PyTorch 0.1.10 - 1.3)
    // Reference: https://github.com/pytorch/pytorch/blob/main/torch/serialization.py#L65
    //
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
    // followed by 10 bytes of magic number (little-endian), then 0x2e (STOP)
    let is_legacy_format = bytes_read >= 15
        && header[0] == 0x80  // PROTO opcode
        && header[1] == 0x02  // Protocol version 2
        && header[2] == 0x8a  // LONG1 opcode
        && header[3] == 0x0a  // 10 bytes follow
        // Magic number 0x1950a86a20f9469cfc6c in little-endian
        && header[4] == 0x6c
        && header[5] == 0xfc
        && header[6] == 0x9c
        && header[7] == 0x46
        && header[8] == 0xf9
        && header[9] == 0x20
        && header[10] == 0x6a
        && header[11] == 0xa8
        && header[12] == 0x50
        && header[13] == 0x19
        && header[14] == 0x2e; // STOP opcode

    if is_legacy_format {
        return load_legacy_pytorch_file_with_metadata(path, top_level_key);
    }

    // Standard pickle file
    // This might be a pickle with tensor references, so we need to handle that case
    // For plain pickle files without a separate data section, we can't use lazy loading
    // so we'll just create empty placeholder tensors for the structure
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Try reading without data source first
    match read_pickle(&mut reader) {
        Ok(obj) => {
            let tensors = extract_tensors_with_data(obj, top_level_key)?;
            let tensor_count = tensors.len();
            Ok((
                tensors,
                PytorchMetadata {
                    format_version: None,
                    format_type: FileFormat::Pickle,
                    byte_order: ByteOrder::LittleEndian,
                    has_storage_alignment: false,
                    pytorch_version: None,
                    tensor_count,
                    total_data_size: None,
                },
            ))
        }
        Err(e)
            if e.to_string()
                .contains("Cannot load tensor data without a data source") =>
        {
            // This pickle file contains tensor data but we're trying to read it without
            // providing a data source. This shouldn't happen in normal usage as PyTorch
            // files with actual tensor data should be in ZIP or legacy format.
            Err(PytorchError::InvalidFormat(
                "Pickle file contains tensor data but no data source is available. This file should be loaded as ZIP or legacy format.".to_string()
            ))
        }
        Err(e) => Err(PytorchError::Pickle(e)),
    }
}

/// Load from a reader
fn load_from_reader<R: Read>(
    reader: R,
    top_level_key: Option<&str>,
) -> Result<HashMap<String, TensorSnapshot>> {
    let mut buf_reader = BufReader::new(reader);

    // Try reading without data source
    match read_pickle(&mut buf_reader) {
        Ok(obj) => extract_tensors_with_data(obj, top_level_key),
        Err(e)
            if e.to_string()
                .contains("Cannot load tensor data without a data source") =>
        {
            // This reader contains tensor data but we can't load it without a file path
            Err(PytorchError::InvalidFormat(
                "Reader contains tensor data but no data source is available. Use file-based loading instead.".to_string()
            ))
        }
        Err(e) => Err(PytorchError::Pickle(e)),
    }
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
                        return Err(PytorchError::KeyNotFound(format!(
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
            return Err(PytorchError::InvalidFormat(
                "Expected a dictionary at the root of the PyTorch file, but found a different type. The file may be a full model save rather than a state_dict.".to_string(),
            ));
        }
    };

    let mut tensors = HashMap::new();
    let mut path = Vec::new();
    extract_tensors_recursive(&Object::Dict(dict), &mut path, &mut tensors);
    Ok(tensors)
}

/// Recursively extract tensors from an object
fn extract_tensors_recursive<'a>(
    obj: &'a Object,
    path: &mut Vec<&'a str>,
    tensors: &mut HashMap<String, TensorSnapshot>,
) {
    match obj {
        Object::Dict(dict) => {
            for (key, value) in dict {
                path.push(key);
                extract_tensors_recursive(value, path, tensors);
                path.pop();
            }
        }
        Object::TorchParam(snapshot) => {
            // The TensorSnapshot already contains the data loading closure
            // Only allocate the string here when we actually insert
            tensors.insert(path.join("."), snapshot.clone());
        }
        _ => {}
    }
}

/// Load a legacy PyTorch file with metadata
fn load_legacy_pytorch_file_with_metadata(
    path: &Path,
    top_level_key: Option<&str>,
) -> Result<(HashMap<String, TensorSnapshot>, PytorchMetadata)> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Skip metadata pickles
    // 1. Magic number
    let _ = read_pickle(&mut reader).map_err(|e| {
        PytorchError::InvalidFormat(format!(
            "Failed to read magic number from legacy format: {}",
            e
        ))
    })?;

    // 2. Protocol version
    let _ = read_pickle(&mut reader).map_err(|e| {
        PytorchError::InvalidFormat(format!(
            "Failed to read protocol version from legacy format: {}",
            e
        ))
    })?;

    // 3. System info
    let _ = read_pickle(&mut reader).map_err(|e| {
        PytorchError::InvalidFormat(format!(
            "Failed to read system info from legacy format: {}",
            e
        ))
    })?;

    // Save position before main pickle
    let main_pickle_pos = reader.stream_position()?;

    // 4. Skip main object - it might contain tensors so we can't parse it yet
    // We'll re-read it with a data source later
    use crate::pytorch::pickle_reader::skip_pickle;
    skip_pickle(&mut reader).map_err(|e| {
        PytorchError::InvalidFormat(format!(
            "Failed to skip main object in legacy format: {}",
            e
        ))
    })?;

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

    // 6. Skip 8-byte header before raw binary data
    // PyTorch legacy format has an 8-byte header (possibly protocol version or alignment)
    // between the storage keys list and the actual tensor data
    let mut header = [0u8; 8];
    if reader.read(&mut header).is_ok() {
        // Header read successfully, data starts after this
    }

    // 7. Raw binary data starts here
    let data_start_pos = reader.stream_position()?;
    let file_size = reader.seek(SeekFrom::End(0))?;
    let data_size = file_size - data_start_pos;

    // Create a lazy data source for legacy multi-storage format
    let data_source = Arc::new(LazyDataSource::from_legacy_multi_storage(
        path,
        data_start_pos,
        data_size,
    ));

    // Set storage keys BEFORE parsing the main pickle
    // This is critical because track_storage_usage() is called during parsing
    // and it needs storage_keys to build the storage map
    if let LazyDataSource::LegacyMultiStorage(ref source) = *data_source
        && !storage_keys.is_empty()
    {
        let source = source
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        source.set_storage_keys(storage_keys.clone());
    }

    // Now re-read the main pickle with lazy data source
    reader.seek(SeekFrom::Start(main_pickle_pos))?;
    let main_obj = read_pickle_with_data(&mut reader, data_source.clone())?;

    // Extract tensors normally
    let tensors = extract_tensors_with_data(main_obj, top_level_key)?;

    // Create metadata for legacy format
    let metadata = PytorchMetadata {
        format_version: None, // Legacy format doesn't have version files
        format_type: FileFormat::Legacy,
        byte_order: ByteOrder::LittleEndian, // Legacy format is little-endian
        has_storage_alignment: false,
        pytorch_version: None, // Could parse from protocol version, but not reliable
        tensor_count: tensors.len(),
        total_data_size: Some(data_size as usize),
    };

    Ok((tensors, metadata))
}

/// Check if a file is a TAR archive
fn is_tar_file(path: &Path) -> bool {
    if let Ok(mut file) = File::open(path) {
        // TAR files have "ustar" magic at offset 257
        let mut header = [0u8; 263];
        if file.read_exact(&mut header).is_ok() {
            // Check for "ustar" magic at offset 257
            return &header[257..262] == b"ustar";
        }
    }
    false
}

/// Load a TAR format PyTorch file with metadata
fn load_tar_pytorch_file_with_metadata(
    path: &Path,
    top_level_key: Option<&str>,
) -> Result<(HashMap<String, TensorSnapshot>, PytorchMetadata)> {
    use tar::Archive;

    let file = File::open(path)?;
    let mut archive = Archive::new(BufReader::new(file));

    // Extract the main entries from the TAR archive
    let mut sys_info_data: Option<Vec<u8>> = None;
    let mut pickle_data: Option<Vec<u8>> = None;
    let mut tensors_data: Option<Vec<u8>> = None;
    let mut storages_data: Option<Vec<u8>> = None;

    for entry in archive.entries().map_err(PytorchError::Tar)? {
        let mut entry = entry.map_err(PytorchError::Tar)?;
        let entry_path = entry
            .path()
            .map_err(PytorchError::Tar)?
            .to_string_lossy()
            .to_string();

        // Skip PAX headers
        if entry_path.contains("@PaxHeader") {
            continue;
        }

        // Normalize path (remove ./ prefix if present)
        let normalized = entry_path.trim_start_matches("./");

        match normalized {
            "sys_info" => {
                let mut data = Vec::new();
                entry.read_to_end(&mut data).map_err(PytorchError::Tar)?;
                sys_info_data = Some(data);
            }
            "pickle" => {
                let mut data = Vec::new();
                entry.read_to_end(&mut data).map_err(PytorchError::Tar)?;
                pickle_data = Some(data);
            }
            "tensors" => {
                let mut data = Vec::new();
                entry.read_to_end(&mut data).map_err(PytorchError::Tar)?;
                tensors_data = Some(data);
            }
            "storages" => {
                let mut data = Vec::new();
                entry.read_to_end(&mut data).map_err(PytorchError::Tar)?;
                storages_data = Some(data);
            }
            _ => {}
        }
    }

    // Validate required entries
    let pickle_data = pickle_data.ok_or_else(|| {
        PytorchError::InvalidFormat("TAR file missing 'pickle' entry".to_string())
    })?;
    let tensors_data = tensors_data.ok_or_else(|| {
        PytorchError::InvalidFormat("TAR file missing 'tensors' entry".to_string())
    })?;
    let storages_data = storages_data.ok_or_else(|| {
        PytorchError::InvalidFormat("TAR file missing 'storages' entry".to_string())
    })?;

    // Parse sys_info to check endianness
    let is_little_endian = if let Some(ref data) = sys_info_data {
        parse_tar_sys_info(data)?
    } else {
        true // Default to little-endian
    };

    if !is_little_endian {
        return Err(PytorchError::InvalidFormat(
            "Big-endian TAR PyTorch files are not supported".to_string(),
        ));
    }

    // Create TarSource for lazy loading
    let data_source = Arc::new(LazyDataSource::from_tar(
        &tensors_data,
        &storages_data,
    )?);

    // Parse the pickle (OrderedDict of name -> storage_key)
    let mut pickle_reader = BufReader::new(pickle_data.as_slice());
    let obj = read_pickle_with_data(&mut pickle_reader, data_source)?;

    // Extract tensors
    let tensors = extract_tensors_with_data(obj, top_level_key)?;

    let metadata = PytorchMetadata {
        format_version: None,
        format_type: FileFormat::Tar,
        byte_order: ByteOrder::LittleEndian,
        has_storage_alignment: false,
        pytorch_version: None,
        tensor_count: tensors.len(),
        total_data_size: Some(storages_data.len()),
    };

    Ok((tensors, metadata))
}

/// Parse sys_info pickle from TAR format to extract endianness
fn parse_tar_sys_info(data: &[u8]) -> Result<bool> {
    let mut reader = BufReader::new(data);
    let obj = read_pickle(&mut reader)?;

    if let Object::Dict(dict) = obj {
        if let Some(Object::Bool(little_endian)) = dict.get("little_endian") {
            return Ok(*little_endian);
        }
    }

    Ok(true) // Default assumption
}

/// Read pickle data from a PyTorch file as a simplified value
fn read_pickle_as_value(path: &Path, top_level_key: Option<&str>) -> Result<PickleValue> {
    use crate::pytorch::lazy_data::LazyDataSource;
    use crate::pytorch::pickle_reader::{read_pickle, read_pickle_with_data};
    use std::sync::Arc;

    // Try to open as ZIP first
    if let Ok(file) = File::open(path)
        && let Ok(mut archive) = zip::ZipArchive::new(BufReader::new(file))
    {
        // Read pickle data from ZIP
        let mut pickle_data = Vec::new();

        // Try standard locations
        for pickle_path in &["data.pkl", "archive/data.pkl"] {
            if let Ok(mut pickle_file) = archive.by_name(pickle_path) {
                pickle_file.read_to_end(&mut pickle_data)?;
                break;
            }
        }

        // If not found, search for any .pkl file
        if pickle_data.is_empty() {
            for i in 0..archive.len() {
                let file = archive.by_index(i)?;
                let name = file.name().to_string();
                drop(file);

                if name.ends_with("data.pkl") {
                    let mut file = archive.by_index(i)?;
                    file.read_to_end(&mut pickle_data)?;
                    break;
                }
            }
        }

        if !pickle_data.is_empty() {
            // Create a data source for the ZIP file
            let data_source = LazyDataSource::from_zip(path)?;
            let data_source_arc = Arc::new(data_source);

            let mut reader = BufReader::new(pickle_data.as_slice());
            let obj = read_pickle_with_data(&mut reader, data_source_arc)?;
            return convert_object_to_value(obj, top_level_key);
        }
    }

    // Try as plain pickle file
    // First attempt without data source (for pure metadata files)
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    match read_pickle(&mut reader) {
        Ok(obj) => convert_object_to_value(obj, top_level_key),
        Err(e)
            if e.to_string()
                .contains("Cannot load tensor data without a data source") =>
        {
            // File contains tensors, need to use full PytorchReader
            // Use the regular reader to get proper tensor handling
            let reader = PytorchReader::new(path)?;

            // Convert tensors to PickleValue structure
            let mut result = std::collections::HashMap::new();
            for key in reader.keys() {
                // For pickle value extraction, we just need the structure, not the actual data
                result.insert(
                    key.clone(),
                    PickleValue::String(format!("<Tensor:{}>", key)),
                );
            }

            if let Some(key) = top_level_key {
                Ok(PickleValue::Dict(
                    [(key.to_string(), PickleValue::Dict(result))]
                        .into_iter()
                        .collect(),
                ))
            } else {
                Ok(PickleValue::Dict(result))
            }
        }
        Err(e) => Err(PytorchError::Pickle(e)),
    }
}

/// Convert internal Object to public PickleValue
fn convert_object_to_value(obj: Object, top_level_key: Option<&str>) -> Result<PickleValue> {
    use crate::pytorch::pickle_reader::Object;

    // If a top-level key is specified, extract it first
    if let Some(key) = top_level_key
        && let Object::Dict(dict) = obj
    {
        if let Some(value) = dict.get(key) {
            return object_to_pickle_value(value.clone());
        } else {
            return Err(PytorchError::KeyNotFound(format!(
                "Key '{}' not found in pickle data",
                key
            )));
        }
    }

    object_to_pickle_value(obj)
}

/// Convert Object to PickleValue
fn object_to_pickle_value(obj: Object) -> Result<PickleValue> {
    use crate::pytorch::pickle_reader::Object;

    Ok(match obj {
        Object::None => PickleValue::None,
        Object::Bool(b) => PickleValue::Bool(b),
        Object::Int(i) => PickleValue::Int(i),
        Object::Float(f) => PickleValue::Float(f),
        Object::String(s) => PickleValue::String(s),
        Object::Persistent(data) => {
            // Persistent data is raw bytes
            PickleValue::Bytes(data)
        }
        Object::PersistentTuple(tuple) => {
            // Convert persistent tuples to lists
            let mut values = Vec::new();
            for item in tuple {
                values.push(object_to_pickle_value(item)?);
            }
            PickleValue::List(values)
        }
        Object::List(list) => {
            let mut values = Vec::new();
            for item in list {
                values.push(object_to_pickle_value(item)?);
            }
            PickleValue::List(values)
        }
        Object::Dict(dict) => {
            let mut map = HashMap::new();
            for (k, v) in dict {
                map.insert(k, object_to_pickle_value(v)?);
            }
            PickleValue::Dict(map)
        }
        Object::Tuple(tuple) => {
            // Convert tuples to lists in the public API
            let mut values = Vec::new();
            for item in tuple {
                values.push(object_to_pickle_value(item)?);
            }
            PickleValue::List(values)
        }
        Object::TorchParam(_) => {
            // Skip tensor parameters in config reading
            PickleValue::None
        }
        Object::Class { .. } | Object::Build { .. } | Object::Reduce { .. } => {
            // Complex objects are represented as None for simplicity
            PickleValue::None
        }
    })
}

/// Convert PickleValue to NestedValue for deserialization
fn convert_pickle_to_nested_value(value: PickleValue) -> Result<NestedValue> {
    Ok(match value {
        PickleValue::None => NestedValue::Default(None),
        PickleValue::Bool(b) => NestedValue::Bool(b),
        PickleValue::Int(i) => NestedValue::I64(i),
        PickleValue::Float(f) => NestedValue::F64(f),
        PickleValue::String(s) => NestedValue::String(s),
        PickleValue::List(list) => {
            let mut vec = Vec::new();
            for item in list {
                vec.push(convert_pickle_to_nested_value(item)?);
            }
            NestedValue::Vec(vec)
        }
        PickleValue::Dict(dict) => {
            let mut map = HashMap::new();
            for (k, v) in dict {
                map.insert(k, convert_pickle_to_nested_value(v)?);
            }
            NestedValue::Map(map)
        }
        PickleValue::Bytes(data) => {
            // Convert bytes to a list of u8 values
            let vec: Vec<NestedValue> = data.into_iter().map(NestedValue::U8).collect();
            NestedValue::Vec(vec)
        }
    })
}
