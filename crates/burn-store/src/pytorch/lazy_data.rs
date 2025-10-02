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
    /// Legacy format with multiple storages in single blob
    LegacyMultiStorage(Arc<Mutex<LegacyMultiStorageSource>>),
}

/// ZIP archive source for lazy loading
pub struct ZipSource {
    path: PathBuf,
    // Cache the file list to avoid reopening archive repeatedly
    file_list: Vec<(String, u64, u64)>, // (name, offset, compressed_size)
}

/// Legacy multi-storage source for old PyTorch format (pre-1.6)
///
/// ## Format Analysis
///
/// Based on research into PyTorch's serialization.py and the legacy TAR format:
///
/// 1. **Storage Layout**: PyTorch legacy format (0.1.10-1.5) stores data as:
///    - Pickle metadata containing tensor definitions
///    - A list of storage keys in order
///    - Raw binary data with all storages concatenated
///
/// 2. **Boundary Detection Challenge**: After extensive research, I found that:
///    - PyTorch does NOT store explicit storage boundaries in the file
///    - Storages are concatenated in the order specified by the storage keys list
///    - Each tensor references its storage by key and specifies offset/size
///
/// 3. **Why True Lazy Loading is Difficult**:
///    - To determine storage boundaries, we would need to:
///      a. Parse ALL tensor metadata to find which storage each uses
///      b. Track the maximum extent of each storage based on tensor usage
///      c. Infer boundaries from the gaps between storages
///    - However, the TensorSnapshot abstraction hides storage keys in closures
///    - This would require deep modifications to the pickle parsing logic
///
/// ## Current Implementation
///
/// This implementation provides a best-effort approach:
/// - Supports setting a storage map if boundaries can be determined externally
/// - Falls back to loading the entire blob if boundaries are unknown
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
            file_list.push((name, offset, compressed_size));
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

impl LazyDataSource {
    /// Create from a ZIP file
    pub fn from_zip(path: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self::Zip(Arc::new(Mutex::new(ZipSource::new(
            path.as_ref().to_path_buf(),
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
            Self::LegacyMultiStorage(_) => vec![], // Legacy format doesn't have distinct keys
        }
    }
}
