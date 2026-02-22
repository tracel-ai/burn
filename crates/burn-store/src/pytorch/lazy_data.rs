//! Lazy data loading support for PyTorch files.
//!
//! This module provides abstractions for lazy loading of tensor data from PyTorch files,
//! avoiding the need to load all data into memory upfront.

use alloc::string::String;
use alloc::vec::Vec;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use zip::ZipArchive;

/// A data source that can lazily load tensor data.
#[derive(Clone)]
pub enum LazyDataSource {
    /// ZIP archive with lazy loading
    Zip(Arc<Mutex<ZipSource>>),
    /// TAR archive format (older torchvision models)
    Tar(Arc<Mutex<TarSource>>),
    /// Legacy format with multiple storages in single blob
    LegacyMultiStorage(Arc<Mutex<LegacyMultiStorageSource>>),
}

/// ZIP archive source for lazy loading
pub struct ZipSource {
    path: PathBuf,
    // Cache the file list to avoid reopening archive repeatedly
    file_list: Vec<(String, u64, u64)>, // (name, offset, compressed_size)
}

/// TAR archive source for lazy loading (older torchvision models like AlexNet, SqueezeNet)
///
/// Older PyTorch/torchvision models (pre-1.6) use TAR format instead of ZIP.
/// The TAR archive contains:
/// - `sys_info`: System info pickle (endianness, type sizes)
/// - `pickle`: OrderedDict mapping tensor names to storage keys
/// - `tensors`: Tensor metadata pickles (unused, metadata is embedded in pickle)
/// - `storages`: Storage count + sequential (metadata pickle, element count, raw data)
pub struct TarSource {
    /// Cached storage map: storage_key -> (offset_in_storages, size_bytes)
    storage_map: HashMap<String, (usize, usize)>,
    /// The raw storages data (kept in memory for TAR format)
    storages_data: Vec<u8>,
}

/// Legacy multi-storage source for old PyTorch format (0.1.10 - 1.5)
///
/// Legacy format stores tensor data as concatenated raw binary without explicit
/// storage boundaries. This source tracks storage usage during tensor parsing
/// to build a storage map for lazy loading.
///
/// ## Storage Layout
/// - Pickle metadata with tensor definitions
/// - List of storage keys (determines concatenation order)
/// - Raw binary blob with all storages concatenated
pub struct LegacyMultiStorageSource {
    path: PathBuf,
    data_offset: u64,
    #[allow(dead_code)]
    data_size: u64,
    // Map of storage_key -> (offset_in_blob, size)
    storage_map: RwLock<Option<HashMap<String, (u64, u64)>>>,
    // Storage keys in order (for boundary calculation)
    storage_keys: RwLock<Option<Vec<String>>>,
    // Track storage usage as tensors are accessed
    storage_usage: RwLock<HashMap<String, usize>>, // key -> max_bytes_needed
}

impl ZipSource {
    /// Create a new ZIP source
    pub fn new(path: PathBuf) -> std::io::Result<Self> {
        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let mut archive = ZipArchive::new(reader)?;

        // Cache file metadata
        let mut file_list = Vec::new();
        for i in 0..archive.len() {
            let file = archive.by_index(i)?;
            let name = file.name().to_string();
            let offset = file.data_start();
            let compressed_size = file.compressed_size();
            file_list.push((
                name,
                offset.expect("should have an offset"),
                compressed_size,
            ));
        }

        Ok(Self { path, file_list })
    }

    /// Check if a file exists in the archive
    pub fn contains(&self, name: &str) -> bool {
        self.file_list.iter().any(|(n, _, _)| n == name)
    }

    /// Get list of data files (excluding pickle files)
    pub fn data_files(&self) -> Vec<String> {
        self.file_list
            .iter()
            .filter(|(name, _, _)| name.starts_with("data/") || name.contains("/data/"))
            .filter(|(name, _, _)| !name.ends_with(".pkl") && !name.ends_with("/"))
            .map(|(name, _, _)| name.clone())
            .collect()
    }

    /// Read a specific file from the archive
    pub fn read_file(&self, name: &str) -> std::io::Result<Vec<u8>> {
        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);
        let mut archive = ZipArchive::new(reader)?;

        let mut file = archive.by_name(name)?;
        let mut contents = Vec::with_capacity(file.size() as usize);
        file.read_to_end(&mut contents)?;
        Ok(contents)
    }

    /// Read a portion of a file
    pub fn read_file_range(
        &self,
        name: &str,
        offset: usize,
        length: usize,
    ) -> std::io::Result<Vec<u8>> {
        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);
        let mut archive = ZipArchive::new(reader)?;

        let mut file = archive.by_name(name)?;
        let mut buffer = vec![0u8; length];

        // Skip to offset
        let mut skip_buffer = vec![0u8; offset.min(8192)];
        let mut skipped = 0;
        while skipped < offset {
            let to_skip = (offset - skipped).min(skip_buffer.len());
            file.read_exact(&mut skip_buffer[..to_skip])?;
            skipped += to_skip;
        }

        // Read the requested data
        file.read_exact(&mut buffer)?;
        Ok(buffer)
    }
}

impl LegacyMultiStorageSource {
    /// Create a new legacy multi-storage source
    pub fn new(path: PathBuf, data_offset: u64, data_size: u64) -> Self {
        Self {
            path,
            data_offset,
            data_size,
            storage_map: RwLock::new(None),
            storage_keys: RwLock::new(None),
            storage_usage: RwLock::new(HashMap::new()),
        }
    }

    /// Set the ordered storage keys from the pickle
    pub fn set_storage_keys(&self, keys: Vec<String>) {
        let mut storage_keys = self
            .storage_keys
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *storage_keys = Some(keys);
    }

    /// Track storage usage from tensor access
    /// This is called from within tensor loading closures
    pub fn track_storage_usage(&self, storage_key: &str, offset: usize, size: usize) {
        let mut usage = self
            .storage_usage
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let max_extent = offset + size;
        usage
            .entry(storage_key.to_string())
            .and_modify(|current| *current = (*current).max(max_extent))
            .or_insert(max_extent);

        // Try to build storage map if we have enough information
        drop(usage);
        self.try_build_storage_map();
    }

    /// Try to build the storage map from tracked usage
    fn try_build_storage_map(&self) {
        // Only build if we don't already have a map
        if self
            .storage_map
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_some()
        {
            return;
        }

        // Check if we have storage keys
        let keys_guard = self
            .storage_keys
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(ref keys) = *keys_guard {
            let usage = self
                .storage_usage
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());

            // Only build if we have usage info for all storages
            if keys.iter().all(|k| usage.contains_key(k)) {
                let mut map = HashMap::new();
                let mut current_offset = 0u64;

                for key in keys {
                    if let Some(&size) = usage.get(key) {
                        map.insert(key.clone(), (current_offset, size as u64));
                        current_offset += size as u64;
                    }
                }

                // Set the storage map
                drop(keys_guard);
                drop(usage);
                let mut storage_map = self
                    .storage_map
                    .write()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                *storage_map = Some(map);
            }
        }
    }

    /// Read data for a specific storage key
    /// Only loads the specific storage portion, never the entire blob
    pub fn read(&self, key: &str) -> std::io::Result<Vec<u8>> {
        // Extract numeric key from paths like "data/0" or just "0"
        let storage_key = key.split('/').next_back().unwrap_or(key);

        // Get storage map - must be available for lazy loading to work
        let storage_map = self
            .storage_map
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        if let Some(ref map) = *storage_map
            && let Some(&(offset, size)) = map.get(storage_key)
        {
            // Load only this specific storage
            let mut file = File::open(&self.path)?;
            file.seek(std::io::SeekFrom::Start(self.data_offset + offset))?;

            let mut buffer = vec![0u8; size as usize];
            file.read_exact(&mut buffer)?;
            return Ok(buffer);
        }

        // NO FALLBACK! If we don't have storage boundaries, we cannot load data lazily
        // The storage map MUST be built from tensor metadata for lazy loading to work
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Storage boundaries not available for key '{}'. Cannot perform lazy loading.",
                storage_key
            ),
        ))
    }
}

impl TarSource {
    /// Create a new TAR source by parsing storages data.
    ///
    /// # Arguments
    /// * `_tensors_data` - Unused; tensor metadata is embedded in the pickle entry
    /// * `storages_data` - Raw storages blob with structure:
    ///   - Count pickle (number of storages)
    ///   - For each storage: metadata pickle + u64 num_elements + raw binary data
    pub fn new(_tensors_data: &[u8], storages_data: Vec<u8>) -> std::io::Result<Self> {
        use super::pickle_reader::read_pickle;
        use std::io::Cursor;

        let mut storage_map = HashMap::new();
        let mut pos = 0usize;

        // First, read the count of storages
        let mut cursor = Cursor::new(&storages_data[pos..]);
        let storage_count = if let Ok(super::pickle_reader::Object::Int(count)) =
            read_pickle(&mut cursor)
        {
            pos += cursor.position() as usize;
            count as usize
        } else {
            0
        };

        // Parse each storage entry
        for _i in 0..storage_count {
            if pos >= storages_data.len() {
                break;
            }

            // Read the storage metadata pickle: (storage_key, device, storage_type)
            let mut cursor = Cursor::new(&storages_data[pos..]);
            if let Ok(obj) = read_pickle(&mut cursor) {
                let pickle_size = cursor.position() as usize;
                pos += pickle_size;

                // Extract storage info from pickle tuple
                let (storage_key, storage_type) = match obj {
                    super::pickle_reader::Object::Tuple(tuple) if tuple.len() >= 3 => {
                        let key = match &tuple[0] {
                            super::pickle_reader::Object::Int(i) => i.to_string(),
                            super::pickle_reader::Object::String(s) => s.clone(),
                            _ => continue,
                        };
                        // tuple[1] is device (e.g., "cpu")
                        // tuple[2] is storage type class
                        let stype = match &tuple[2] {
                            super::pickle_reader::Object::Class { name, .. } => name.clone(),
                            _ => "FloatStorage".to_string(),
                        };
                        (key, stype)
                    }
                    _ => continue,
                };

                // Read the number of elements (u64 little-endian)
                if pos + 8 > storages_data.len() {
                    break;
                }
                let num_elements = u64::from_le_bytes([
                    storages_data[pos],
                    storages_data[pos + 1],
                    storages_data[pos + 2],
                    storages_data[pos + 3],
                    storages_data[pos + 4],
                    storages_data[pos + 5],
                    storages_data[pos + 6],
                    storages_data[pos + 7],
                ]) as usize;
                pos += 8;

                // Determine element size from storage type
                let element_size = if storage_type.contains("Double") || storage_type.contains("Long") {
                    8
                } else if storage_type.contains("Half") || storage_type.contains("Short") {
                    2
                } else if storage_type.contains("Byte") || storage_type.contains("Char") || storage_type.contains("Bool") {
                    1
                } else {
                    4 // Default to float (4 bytes)
                };

                let data_size = num_elements * element_size;

                // Store the offset to raw data and its size
                storage_map.insert(storage_key, (pos, data_size));

                // Skip the raw binary data
                pos += data_size;
            } else {
                break;
            }
        }

        Ok(Self {
            storage_map,
            storages_data,
        })
    }

    /// Read data for a specific storage key
    pub fn read(&self, key: &str) -> std::io::Result<Vec<u8>> {
        // Extract the storage key from paths like "data/0"
        let storage_key = key.split('/').next_back().unwrap_or(key);

        if let Some(&(offset, size)) = self.storage_map.get(storage_key) {
            if offset + size <= self.storages_data.len() {
                return Ok(self.storages_data[offset..offset + size].to_vec());
            }
        }

        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Storage key '{}' not found in TAR archive", storage_key),
        ))
    }

    /// Check if a storage key exists
    pub fn contains(&self, key: &str) -> bool {
        let storage_key = key.split('/').next_back().unwrap_or(key);
        self.storage_map.contains_key(storage_key)
    }

    /// Get list of storage keys
    pub fn keys(&self) -> Vec<String> {
        self.storage_map.keys().cloned().collect()
    }
}

impl LazyDataSource {
    /// Create from a ZIP file
    pub fn from_zip(path: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self::Zip(Arc::new(Mutex::new(ZipSource::new(
            path.as_ref().to_path_buf(),
        )?))))
    }

    /// Create from a TAR archive's tensors and storages data
    pub fn from_tar(tensors_data: &[u8], storages_data: &[u8]) -> std::io::Result<Self> {
        Ok(Self::Tar(Arc::new(Mutex::new(TarSource::new(
            tensors_data,
            storages_data.to_vec(),
        )?))))
    }

    /// Create from a legacy multi-storage file
    pub fn from_legacy_multi_storage(
        path: impl AsRef<Path>,
        data_offset: u64,
        data_size: u64,
    ) -> Self {
        Self::LegacyMultiStorage(Arc::new(Mutex::new(LegacyMultiStorageSource::new(
            path.as_ref().to_path_buf(),
            data_offset,
            data_size,
        ))))
    }

    /// Read data for a specific key
    pub fn read(&self, key: &str) -> std::io::Result<Vec<u8>> {
        match self {
            Self::Zip(source) => {
                let source = source
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                source.read_file(key)
            }
            Self::Tar(source) => {
                let source = source
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                source.read(key)
            }
            Self::LegacyMultiStorage(source) => {
                let source = source
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                source.read(key)
            }
        }
    }

    /// Read a portion of data for a specific key
    pub fn read_range(&self, key: &str, offset: usize, length: usize) -> std::io::Result<Vec<u8>> {
        match self {
            Self::Zip(source) => {
                let source = source
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                source.read_file_range(key, offset, length)
            }
            Self::Tar(source) => {
                // For TAR format, read the full data and slice it
                let source = source
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let data = source.read(key)?;
                let end = (offset + length).min(data.len());
                Ok(data[offset..end].to_vec())
            }
            Self::LegacyMultiStorage(source) => {
                // For legacy format, read only the requested range
                let storage_key = key.split('/').next_back().unwrap_or(key);
                let source = source
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());

                // Get storage boundaries
                let storage_map = source
                    .storage_map
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if let Some(ref map) = *storage_map
                    && let Some(&(storage_offset, storage_size)) = map.get(storage_key)
                {
                    // Calculate actual file position
                    let file_offset = source.data_offset + storage_offset + offset as u64;
                    let read_length = length.min((storage_size as usize).saturating_sub(offset));

                    // Read only the requested range
                    let mut file = File::open(&source.path)?;
                    file.seek(std::io::SeekFrom::Start(file_offset))?;

                    let mut buffer = vec![0u8; read_length];
                    file.read_exact(&mut buffer)?;
                    Ok(buffer)
                } else {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "Storage boundaries not available for key '{}'. Cannot perform lazy loading.",
                            storage_key
                        ),
                    ))
                }
            }
        }
    }

    /// Check if a key exists
    pub fn contains(&self, key: &str) -> bool {
        match self {
            Self::Zip(source) => {
                let source = source
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                source.contains(key)
            }
            Self::Tar(source) => {
                let source = source
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                source.contains(key)
            }
            Self::LegacyMultiStorage(_) => true, // Legacy format has all data
        }
    }

    /// Get list of available keys (for ZIP sources)
    pub fn keys(&self) -> Vec<String> {
        match self {
            Self::Zip(source) => {
                let source = source
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                source.data_files()
            }
            Self::Tar(source) => {
                let source = source
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                source.keys()
            }
            Self::LegacyMultiStorage(_) => vec![], // Legacy format doesn't have distinct keys
        }
    }
}
