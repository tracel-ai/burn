//! PyTorch file reader implementation.
//!
//! This module provides support for reading PyTorch checkpoint files (.pt/.pth).
//!
//! # Supported Formats
//!
//! ## 1. Modern ZIP Format (PyTorch 1.6+)
//! Files are ZIP archives holding, under a root directory (usually named after the file):
//! - `data.pkl`: Pickled tensor metadata
//! - `data/`: One binary file per tensor storage
//! - `version`, `byteorder`, and other small text entries
//!
//! ## 2. TAR Format (PyTorch before 0.1.10, e.g. early torchvision models)
//! TAR archives containing:
//! - `sys_info`: System info pickle (endianness, type sizes)
//! - `storages`: Count pickle, then per storage a metadata pickle, element count and bytes
//! - `tensors`: Count pickle, then per tensor a metadata pickle and binary shape, stride
//!   and storage offset
//! - `pickle`: The saved object, referencing tensors by persistent id
//!
//! ## 3. Legacy Pickle Format (PyTorch 0.1.10 - 1.5)
//! Sequential pickle streams with the structure:
//! - Magic number pickle (0x1950a86a20f9469cfc6c)
//! - Protocol version pickle (e.g., 1001)
//! - System info pickle (endianness, type sizes)
//! - Model data pickle (state_dict or full model)
//! - Storage key list, then each storage as an `i64` element count followed by its bytes
//!
//! ## 4. Simple Pickle Format
//! Direct pickle file with a dictionary at the root, commonly used for
//! manually saved configuration.
//!
//! # Compatibility
//!
//! The reader detects the file format automatically. Files from the earliest PyTorch
//! releases (TAR) through current versions are supported. Full model saves (as opposed to
//! a state_dict) are refused, as are checkpoints holding sparse, quantized or nested
//! tensors, since those cannot be represented. Only little-endian files are supported.

use crate::nested::{adapter::DefaultAdapter, data::NestedValue, de::Deserializer};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use burn_core::tensor::DType;
use burn_pack::Tensor as PackTensor;
use serde::de::DeserializeOwned;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use super::pickle_reader::{
    Object, PersistentIds, PickleError, StorageRef, build_tensor, extract_tensors, key_string,
    non_negative, read_pickle, storage_type_to_dtype,
};
use super::storage::{LegacySource, StorageSource, TarSource, ZipSource};
use byteorder::{LittleEndian, ReadBytesExt};

/// Error type for PyTorch file operations
#[derive(Debug)]
#[non_exhaustive]
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
    Serde(crate::nested::error::Error),
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

impl From<crate::nested::error::Error> for PytorchError {
    fn from(e: crate::nested::error::Error) -> Self {
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
            PytorchError::KeyNotFound(msg) => write!(f, "Key not found in PyTorch file: {}", msg),
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
    /// Contents of the `.format_version` entry (e.g. `"1"`), if the archive has one
    pub format_version: Option<String>,
    /// File format type (ZIP, TAR, Legacy, or Pickle)
    pub format_type: FileFormat,
    /// Byte order. Always `LittleEndian`: big-endian files are rejected at load.
    pub byte_order: ByteOrder,
    /// Whether the file has storage alignment information
    pub has_storage_alignment: bool,
    /// Contents of the `version` entry (e.g. `"3"`): the serialized-file format version,
    /// not the PyTorch release that wrote the file
    pub pytorch_version: Option<String>,
    /// Number of tensors in the file
    pub tensor_count: usize,
    /// Approximate size of the storage section in bytes (if available)
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

    fn for_format(format_type: FileFormat) -> Self {
        Self {
            format_version: None,
            format_type,
            byte_order: ByteOrder::LittleEndian,
            has_storage_alignment: false,
            pytorch_version: None,
            tensor_count: 0,
            total_data_size: None,
        }
    }
}

/// File format type
#[derive(Debug, Clone, PartialEq)]
pub enum FileFormat {
    /// ZIP-based format (PyTorch 1.6+)
    Zip,
    /// TAR-based format (PyTorch before 0.1.10)
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
/// legacy format (0.1.10-1.5), the early TAR format, and simple pickle files.
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
///     let data = burn_store::bridge::to_data(tensor)?; // Materializes the tensor
/// }
///
/// // Check file metadata
/// println!("Format: {:?}", reader.metadata().format_type);
/// println!("Tensor count: {}", reader.metadata().tensor_count);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct PytorchReader {
    tensors: HashMap<String, PackTensor>,
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
        Self::open(path.as_ref(), None)
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
        Self::open(path.as_ref(), Some(key))
    }

    fn open(path: &Path, top_level_key: Option<&str>) -> Result<Self> {
        let Loaded { root, mut metadata } = load_file(path)?;
        let tensors = extract_tensors_at(root, top_level_key)?;
        metadata.tensor_count = tensors.len();
        Ok(Self { tensors, metadata })
    }

    /// Load from a reader
    ///
    /// This method is useful when loading from non-file sources like memory buffers.
    /// The reader must hold a plain pickle: tensor data lives outside the pickle in every
    /// PyTorch container, so a checkpoint with tensors must be loaded from a file.
    ///
    /// # Arguments
    /// * `reader` - Any type implementing `Read`
    /// * `top_level_key` - Optional key to extract
    pub fn from_reader<R: Read>(reader: R, top_level_key: Option<&str>) -> Result<Self> {
        let root = read_pickle(&mut BufReader::new(reader), &PersistentIds::Unavailable)?;
        let tensors = extract_tensors_at(root, top_level_key)?;
        let mut metadata = PytorchMetadata::for_format(FileFormat::Pickle);
        metadata.tensor_count = tensors.len();
        Ok(Self { tensors, metadata })
    }

    /// Get all tensor names
    pub fn keys(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    /// Get a tensor by name
    pub fn get(&self, name: &str) -> Option<&PackTensor> {
        self.tensors.get(name)
    }

    /// Get all tensors
    pub fn tensors(&self) -> &HashMap<String, PackTensor> {
        &self.tensors
    }

    /// Take ownership of all tensors
    pub fn into_tensors(self) -> HashMap<String, PackTensor> {
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
    /// Tensors and Python objects the reader does not interpret appear as
    /// [`PickleValue::None`].
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
        let Loaded { root, .. } = load_file(path.as_ref())?;
        let value = select_top_level(root, top_level_key)?;
        Ok(to_pickle_value(value))
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
        let pickle_value = Self::read_pickle_data(path, top_level_key)?;
        let nested_value = to_nested_value(pickle_value);
        let deserializer = Deserializer::<DefaultAdapter>::new(nested_value, false);
        Ok(D::deserialize(deserializer)?)
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

// ---------------------------------------------------------------------------------------------
// Format detection and loading
// ---------------------------------------------------------------------------------------------

/// A parsed file: the root pickle object plus what the container says about itself.
struct Loaded {
    root: Object,
    /// `tensor_count` is filled in once tensors have been extracted.
    metadata: PytorchMetadata,
}

/// The first pickle of a legacy file: `PROTO 2`, `LONG1` of the 10-byte magic number
/// 0x1950a86a20f9469cfc6c (little-endian), `STOP`.
const LEGACY_MAGIC: [u8; 15] = [
    0x80, 0x02, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19, 0x2e,
];

/// Offset of the `ustar` magic within a TAR header.
const TAR_MAGIC_OFFSET: usize = 257;

fn detect_format(path: &Path) -> Result<FileFormat> {
    let mut header = Vec::new();
    File::open(path)?
        .take((TAR_MAGIC_OFFSET + 5) as u64)
        .read_to_end(&mut header)?;

    if header.starts_with(b"PK\x03\x04") || header.starts_with(b"PK\x05\x06") {
        Ok(FileFormat::Zip)
    } else if header.get(TAR_MAGIC_OFFSET..TAR_MAGIC_OFFSET + 5) == Some(b"ustar") {
        Ok(FileFormat::Tar)
    } else if header.starts_with(&LEGACY_MAGIC) {
        Ok(FileFormat::Legacy)
    } else {
        Ok(FileFormat::Pickle)
    }
}

fn load_file(path: &Path) -> Result<Loaded> {
    match detect_format(path)? {
        FileFormat::Zip => load_zip(path),
        FileFormat::Tar => load_tar(path),
        FileFormat::Legacy => load_legacy(path),
        FileFormat::Pickle => load_plain_pickle(path),
    }
}

fn load_zip(path: &Path) -> Result<Loaded> {
    let source = ZipSource::open(path)?;

    if source.read_text("byteorder")?.as_deref() == Some("big") {
        return Err(big_endian_error());
    }
    let mut metadata = PytorchMetadata::for_format(FileFormat::Zip);
    metadata.format_version = source.read_text(".format_version")?;
    metadata.pytorch_version = source.read_text("version")?;
    metadata.has_storage_alignment = source.has_entry(".storage_alignment");
    metadata.total_data_size = Some(source.data_size()? as usize);

    let pickle = source.pickle()?;
    let ids = PersistentIds::Storages(Arc::new(StorageSource::Zip(source)));
    let root = read_pickle(&mut Cursor::new(pickle), &ids)?;

    Ok(Loaded { root, metadata })
}

fn load_legacy(path: &Path) -> Result<Loaded> {
    let mut reader = BufReader::new(File::open(path)?);
    reader.seek(SeekFrom::Start(LEGACY_MAGIC.len() as u64))?;

    let read_header = |reader: &mut BufReader<File>, what: &str| {
        read_pickle(reader, &PersistentIds::Unavailable).map_err(|e| {
            PytorchError::InvalidFormat(format!("Failed to read {what} from legacy format: {e}"))
        })
    };
    // PyTorch refuses anything but 1001 here.
    let protocol_version = read_header(&mut reader, "protocol version")?;
    if !matches!(protocol_version, Object::Int(1001)) {
        return Err(PytorchError::InvalidFormat(format!(
            "Unsupported legacy protocol version {protocol_version:?}, expected 1001"
        )));
    }
    let sys_info = read_header(&mut reader, "system info")?;
    check_little_endian(&sys_info)?;

    let source = Arc::new(StorageSource::Legacy(LegacySource::new(path)));
    let root = read_pickle(&mut reader, &PersistentIds::Storages(source.clone()))?;

    // The storage keys, in the order their bytes follow.
    let storage_keys = match read_header(&mut reader, "storage key list")? {
        Object::List(keys) => keys
            .iter()
            .map(|key| key_string(key, "legacy storage key"))
            .collect::<std::result::Result<Vec<_>, _>>()?,
        other => {
            return Err(PytorchError::InvalidFormat(format!(
                "legacy storage key list must be a list, got {other:?}"
            )));
        }
    };

    let data_start = reader.stream_position()?;
    let file_len = reader.seek(SeekFrom::End(0))?;
    if let StorageSource::Legacy(legacy) = &*source {
        legacy
            .finish(&storage_keys, data_start, file_len)
            .map_err(|e| PytorchError::InvalidFormat(e.to_string()))?;
    }

    let mut metadata = PytorchMetadata::for_format(FileFormat::Legacy);
    metadata.total_data_size = Some((file_len - data_start) as usize);
    Ok(Loaded { root, metadata })
}

fn load_plain_pickle(path: &Path) -> Result<Loaded> {
    let mut reader = BufReader::new(File::open(path)?);
    let root = read_pickle(&mut reader, &PersistentIds::Unavailable)?;
    Ok(Loaded {
        root,
        metadata: PytorchMetadata::for_format(FileFormat::Pickle),
    })
}

fn load_tar(path: &Path) -> Result<Loaded> {
    let mut archive = tar::Archive::new(BufReader::new(File::open(path)?));
    let mut entries: HashMap<String, Vec<u8>> = HashMap::new();

    for entry in archive.entries().map_err(PytorchError::Tar)? {
        let mut entry = entry.map_err(PytorchError::Tar)?;
        let name = entry
            .path()
            .map_err(PytorchError::Tar)?
            .to_string_lossy()
            .trim_start_matches("./")
            .to_string();
        if matches!(
            name.as_str(),
            "sys_info" | "storages" | "tensors" | "pickle"
        ) {
            let mut data = Vec::new();
            entry.read_to_end(&mut data).map_err(PytorchError::Tar)?;
            entries.insert(name, data);
        }
    }

    if let Some(sys_info) = entries.remove("sys_info") {
        let sys_info = read_pickle(&mut Cursor::new(sys_info), &PersistentIds::Unavailable)?;
        check_little_endian(&sys_info)?;
    }
    let mut take = |name: &str| {
        entries
            .remove(name)
            .ok_or_else(|| PytorchError::InvalidFormat(format!("TAR file missing '{name}' entry")))
    };
    let storages = take("storages")?;
    let tensors = take("tensors")?;
    let pickle = take("pickle")?;

    let total_data_size = storages.len();
    let (tar_source, storage_info) = parse_tar_storages(storages)?;
    let source = Arc::new(StorageSource::Tar(tar_source));
    let tensors = parse_tar_tensors(&tensors, &storage_info, &source)?;
    let root = read_pickle(&mut Cursor::new(pickle), &PersistentIds::Tensors(tensors))?;

    let mut metadata = PytorchMetadata::for_format(FileFormat::Tar);
    metadata.total_data_size = Some(total_data_size);
    Ok(Loaded { root, metadata })
}

/// Per storage key, the element type and byte length a TAR `storages` entry declares.
type TarStorageInfo = HashMap<String, (DType, usize)>;

/// Parse the `storages` entry: a count, then per storage `(key, location, storage type)`,
/// an `i64` element count and the bytes, then a list of storage views.
fn parse_tar_storages(blob: Vec<u8>) -> Result<(TarSource, TarStorageInfo)> {
    let mut cursor = Cursor::new(blob.as_slice());
    let count = read_tar_count(&mut cursor, "storage")?;

    // Every storage costs at least a pickle and a count, so the count bounds the blob.
    let plausible = count.min(blob.len() / 8);
    let mut layout: HashMap<String, (usize, usize)> = HashMap::with_capacity(plausible);
    let mut info = TarStorageInfo::with_capacity(plausible);

    for _ in 0..count {
        let meta = read_pickle(&mut cursor, &PersistentIds::Unavailable)?;
        let [key, _location, storage_type] = tar_fields(&meta, "storage metadata")?;
        // The tables key storages by int `_cdata`; the main pickle's persistent ids are
        // `str(_cdata)`. Both are normalized to strings.
        let key = key_string(key, "TAR storage key")?;
        let dtype = match storage_type {
            Object::Class { name, .. } => storage_type_to_dtype(name.as_str())?,
            other => {
                return Err(PytorchError::InvalidFormat(format!(
                    "TAR storage type must be a class, got {other:?}"
                )));
            }
        };

        let numel = read_tar_usize(&mut cursor, "storage element count")?;
        let byte_len = numel.checked_mul(dtype.size()).ok_or_else(|| {
            PytorchError::InvalidFormat(format!("TAR storage '{key}' byte length overflows usize"))
        })?;
        let offset = cursor.position() as usize;
        let end = offset
            .checked_add(byte_len)
            .filter(|&end| end <= blob.len())
            .ok_or_else(|| {
                PytorchError::InvalidFormat(format!(
                    "TAR storage '{key}' ({byte_len} bytes) extends beyond the storages entry"
                ))
            })?;

        layout.insert(key.clone(), (offset, byte_len));
        info.insert(key, (dtype, byte_len));
        cursor.set_position(end as u64);
    }

    // Views of root storages: (view key, root key, element offset, element count).
    // PyTorch reads this list unconditionally, so its absence means a truncated entry.
    {
        let views = read_pickle(&mut cursor, &PersistentIds::Unavailable).map_err(|e| {
            PytorchError::InvalidFormat(format!("TAR storage views list missing: {e}"))
        })?;
        let Object::List(views) = views else {
            return Err(PytorchError::InvalidFormat(format!(
                "TAR storage views must be a list, got {views:?}"
            )));
        };
        for view in views {
            let [target, root, offset, numel] = tar_fields(&view, "storage view")?;
            let target = key_string(target, "TAR storage view key")?;
            let root = key_string(root, "TAR storage view root")?;
            let offset = non_negative(offset, "TAR storage view offset")?;
            let numel = non_negative(numel, "TAR storage view element count")?;
            let &(root_offset, root_len) = layout.get(&root).ok_or_else(|| {
                PytorchError::InvalidFormat(format!(
                    "TAR storage view '{target}' refers to unknown storage '{root}'"
                ))
            })?;
            let (dtype, _) = info[&root];
            let element_size = dtype.size();
            let (start, len) = offset
                .checked_mul(element_size)
                .zip(numel.checked_mul(element_size))
                .filter(|&(start, len)| start.checked_add(len).is_some_and(|end| end <= root_len))
                .ok_or_else(|| {
                    PytorchError::InvalidFormat(format!(
                        "TAR storage view '{target}' lies outside storage '{root}'"
                    ))
                })?;
            layout.insert(target.clone(), (root_offset + start, len));
            info.insert(target, (dtype, len));
        }
    }

    Ok((TarSource::new(blob, layout), info))
}

/// Parse the `tensors` entry: a count, then per tensor `(key, storage key, tensor type)`,
/// an `i32` rank, 4 unused bytes, `rank` sizes, `rank` strides and the storage offset.
fn parse_tar_tensors(
    blob: &[u8],
    storage_info: &TarStorageInfo,
    source: &Arc<StorageSource>,
) -> Result<HashMap<String, PackTensor>> {
    let mut cursor = Cursor::new(blob);
    let count = read_tar_count(&mut cursor, "tensor")?;
    let mut tensors = HashMap::with_capacity(count.min(blob.len() / 8));

    for _ in 0..count {
        let meta = read_pickle(&mut cursor, &PersistentIds::Unavailable)?;
        let [key, storage_key, _tensor_type] = tar_fields(&meta, "tensor metadata")?;
        let key = key_string(key, "TAR tensor key")?;
        let storage_key = key_string(storage_key, "TAR tensor storage key")?;

        let rank = cursor.read_i32::<LittleEndian>()?;
        let rank = usize::try_from(rank)
            .ok()
            .filter(|&rank| rank <= 64)
            .ok_or_else(|| {
                PytorchError::InvalidFormat(format!("TAR tensor '{key}' has invalid rank {rank}"))
            })?;
        cursor.read_i32::<LittleEndian>()?; // Padding: the rank was once written as 8 bytes.
        let shape = (0..rank)
            .map(|_| read_tar_usize(&mut cursor, "tensor shape"))
            .collect::<Result<Vec<_>>>()?;
        let stride = (0..rank)
            .map(|_| read_tar_usize(&mut cursor, "tensor stride"))
            .collect::<Result<Vec<_>>>()?;
        let storage_offset = read_tar_usize(&mut cursor, "tensor storage offset")?;

        let &(dtype, byte_len) = storage_info.get(&storage_key).ok_or_else(|| {
            PytorchError::InvalidFormat(format!(
                "TAR tensor '{key}' refers to unknown storage '{storage_key}'"
            ))
        })?;
        let storage = StorageRef {
            source: source.clone(),
            key: storage_key,
            dtype: Some(dtype),
            byte_len,
            view_offset: 0,
        };
        let tensor = build_tensor(storage, dtype, storage_offset, shape, stride)?;
        tensors.insert(key, tensor);
    }

    Ok(tensors)
}

fn read_tar_count(cursor: &mut Cursor<&[u8]>, what: &str) -> Result<usize> {
    let count = read_pickle(cursor, &PersistentIds::Unavailable)?;
    Ok(non_negative(&count, &format!("TAR {what} count"))?)
}

/// Read one of the `i64` fields in the TAR binary tables as a `usize`.
fn read_tar_usize(cursor: &mut Cursor<&[u8]>, what: &str) -> Result<usize> {
    let value = cursor.read_i64::<LittleEndian>()?;
    usize::try_from(value).map_err(|_| {
        PytorchError::InvalidFormat(format!("TAR {what} must be non-negative, got {value}"))
    })
}

/// View a TAR metadata pickle as a tuple of exactly `N` fields.
fn tar_fields<'a, const N: usize>(obj: &'a Object, what: &str) -> Result<&'a [Object; N]> {
    match obj {
        Object::Tuple(fields) => <&[Object; N]>::try_from(fields.as_slice()).map_err(|_| {
            PytorchError::InvalidFormat(format!(
                "TAR {what} must have {N} fields, got {}",
                fields.len()
            ))
        }),
        other => Err(PytorchError::InvalidFormat(format!(
            "TAR {what} must be a tuple, got {other:?}"
        ))),
    }
}

fn check_little_endian(sys_info: &Object) -> Result<()> {
    if let Object::Dict(dict) = sys_info
        && let Some(Object::Bool(false)) = dict.get("little_endian")
    {
        return Err(big_endian_error());
    }
    Ok(())
}

fn big_endian_error() -> PytorchError {
    PytorchError::InvalidFormat(
        "Big-endian PyTorch files are not supported. The file was saved on a big-endian system and requires byte order conversion."
            .to_string(),
    )
}

// ---------------------------------------------------------------------------------------------
// Root object handling
// ---------------------------------------------------------------------------------------------

/// Return `root`, or the entry under `key` when one is given.
fn select_top_level(root: Object, key: Option<&str>) -> Result<Object> {
    let Some(key) = key else {
        return Ok(root);
    };
    let Object::Dict(mut dict) = root else {
        return Err(PytorchError::InvalidFormat(format!(
            "Cannot select top-level key '{key}': the file's root is not a dictionary"
        )));
    };
    dict.remove(key).ok_or_else(|| {
        let mut available: Vec<_> = dict.keys().cloned().collect();
        available.sort();
        PytorchError::KeyNotFound(format!(
            "'{key}' not found. Available top-level keys: {available:?}"
        ))
    })
}

fn extract_tensors_at(
    root: Object,
    top_level_key: Option<&str>,
) -> Result<HashMap<String, PackTensor>> {
    let Object::Dict(dict) = select_top_level(root, top_level_key)? else {
        return Err(PytorchError::InvalidFormat(match top_level_key {
            Some(key) => format!("Top-level key '{key}' does not hold a dictionary"),
            None => "Expected a dictionary at the root of the PyTorch file, but found a different type. The file may be a full model save rather than a state_dict.".to_string(),
        }));
    };
    Ok(extract_tensors(dict))
}

/// Convert an internal object to the public [`PickleValue`].
///
/// Tuples become lists, and anything without a JSON-like counterpart (classes, tensors,
/// storages, uninterpreted objects) becomes `None`.
fn to_pickle_value(obj: Object) -> PickleValue {
    match obj {
        Object::None => PickleValue::None,
        Object::Bool(b) => PickleValue::Bool(b),
        Object::Int(i) => PickleValue::Int(i),
        Object::Float(f) => PickleValue::Float(f),
        Object::String(s) => PickleValue::String(s),
        Object::Bytes(b) => PickleValue::Bytes(b),
        Object::List(items) | Object::Tuple(items) => {
            PickleValue::List(items.into_iter().map(to_pickle_value).collect())
        }
        Object::Dict(dict) => PickleValue::Dict(
            dict.into_iter()
                .map(|(k, v)| (k, to_pickle_value(v)))
                .collect(),
        ),
        Object::Class { .. } | Object::Storage(_) | Object::Opaque | Object::Tensor(_) => {
            PickleValue::None
        }
    }
}

/// Convert a [`PickleValue`] to a [`NestedValue`] for deserialization.
fn to_nested_value(value: PickleValue) -> NestedValue {
    match value {
        PickleValue::None => NestedValue::Default(None),
        PickleValue::Bool(b) => NestedValue::Bool(b),
        PickleValue::Int(i) => NestedValue::I64(i),
        PickleValue::Float(f) => NestedValue::F64(f),
        PickleValue::String(s) => NestedValue::String(s),
        PickleValue::List(list) => {
            NestedValue::Vec(list.into_iter().map(to_nested_value).collect())
        }
        PickleValue::Dict(dict) => NestedValue::Map(
            dict.into_iter()
                .map(|(k, v)| (k, to_nested_value(v)))
                .collect(),
        ),
        PickleValue::Bytes(data) => {
            NestedValue::Vec(data.into_iter().map(NestedValue::U8).collect())
        }
    }
}
