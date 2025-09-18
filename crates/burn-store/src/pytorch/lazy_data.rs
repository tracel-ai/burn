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
    /// Direct file access for legacy format
    File(Arc<Mutex<FileSource>>),
    /// Legacy format with multiple storages in single blob
    LegacyMultiStorage(Arc<Mutex<LegacyMultiStorageSource>>),
}

/// ZIP archive source for lazy loading
pub struct ZipSource {
    path: PathBuf,
    // Cache the file list to avoid reopening archive repeatedly
    file_list: Vec<(String, u64, u64)>, // (name, offset, compressed_size)
}

/// File source for legacy format
pub struct FileSource {
    path: PathBuf,
    offset: u64,
    size: u64,
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
/// - Falls back to loading the entire blob and caching it
/// - Individual storages are cached after first access to minimize file I/O
pub struct LegacyMultiStorageSource {
    path: PathBuf,
    data_offset: u64,
    data_size: u64,
    // Map of storage_key -> (offset_in_blob, size)
    storage_map: RwLock<Option<HashMap<String, (u64, u64)>>>,
    // Cache individual storages after loading
    cached_storages: RwLock<HashMap<String, Arc<Vec<u8>>>>,
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

impl FileSource {
    /// Create a new file source
    pub fn new(path: PathBuf, offset: u64, size: u64) -> Self {
        Self { path, offset, size }
    }

    /// Read data from the file
    pub fn read(&self) -> std::io::Result<Vec<u8>> {
        let mut file = File::open(&self.path)?;
        file.seek(std::io::SeekFrom::Start(self.offset))?;

        let mut buffer = vec![0u8; self.size as usize];
        file.read_exact(&mut buffer)?;
        Ok(buffer)
    }

    /// Read a portion of the data
    pub fn read_range(&self, offset: usize, length: usize) -> std::io::Result<Vec<u8>> {
        let mut file = File::open(&self.path)?;
        file.seek(std::io::SeekFrom::Start(self.offset + offset as u64))?;

        let mut buffer = vec![0u8; length.min((self.size as usize).saturating_sub(offset))];
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
            cached_storages: RwLock::new(HashMap::new()),
        }
    }

    /// Set the storage map after analyzing tensor metadata
    /// This should be called after pickle parsing to enable efficient lazy loading
    pub fn set_storage_map(&self, map: HashMap<String, (u64, u64)>) {
        let mut storage_map = self.storage_map.write().unwrap();
        *storage_map = Some(map);
    }

    /// Read data for a specific storage key
    /// Uses storage map if available for efficient loading, otherwise loads entire blob
    pub fn read(&self, key: &str) -> std::io::Result<Vec<u8>> {
        // Extract numeric key from paths like "data/0" or just "0"
        let storage_key = key.split('/').next_back().unwrap_or(key);

        // Check cache first
        {
            let cache = self.cached_storages.read().unwrap();
            if let Some(data) = cache.get(storage_key) {
                return Ok(data.as_ref().clone());
            }
        }

        // Try to use storage map for efficient loading
        let storage_map = self.storage_map.read().unwrap();
        if let Some(ref map) = *storage_map
            && let Some(&(offset, size)) = map.get(storage_key)
        {
            // Load only this specific storage
            let mut file = File::open(&self.path)?;
            file.seek(std::io::SeekFrom::Start(self.data_offset + offset))?;

            let mut buffer = vec![0u8; size as usize];
            file.read_exact(&mut buffer)?;

            // Cache this storage
            let data_arc = Arc::new(buffer.clone());
            {
                let mut cache = self.cached_storages.write().unwrap();
                cache.insert(storage_key.to_string(), data_arc);
            }

            return Ok(buffer);
        }

        // Fallback: load entire blob (for compatibility)
        // This happens if storage map wasn't set or key not found
        let mut file = File::open(&self.path)?;
        file.seek(std::io::SeekFrom::Start(self.data_offset))?;

        let mut buffer = vec![0u8; self.data_size as usize];
        file.read_exact(&mut buffer)?;

        // Cache as single storage "0" for compatibility
        let data_arc = Arc::new(buffer.clone());
        {
            let mut cache = self.cached_storages.write().unwrap();
            cache.insert("0".to_string(), data_arc);
        }

        Ok(buffer)
    }
}

impl LazyDataSource {
    /// Create from a ZIP file
    pub fn from_zip(path: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self::Zip(Arc::new(Mutex::new(ZipSource::new(
            path.as_ref().to_path_buf(),
        )?))))
    }

    /// Create from a file with offset and size
    pub fn from_file(path: impl AsRef<Path>, offset: u64, size: u64) -> Self {
        Self::File(Arc::new(Mutex::new(FileSource::new(
            path.as_ref().to_path_buf(),
            offset,
            size,
        ))))
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
                let source = source.lock().unwrap();
                source.read_file(key)
            }
            Self::File(source) => {
                let source = source.lock().unwrap();
                source.read()
            }
            Self::LegacyMultiStorage(source) => {
                let source = source.lock().unwrap();
                source.read(key)
            }
        }
    }

    /// Read a portion of data for a specific key
    pub fn read_range(&self, key: &str, offset: usize, length: usize) -> std::io::Result<Vec<u8>> {
        match self {
            Self::Zip(source) => {
                let source = source.lock().unwrap();
                source.read_file_range(key, offset, length)
            }
            Self::File(source) => {
                let source = source.lock().unwrap();
                source.read_range(offset, length)
            }
            Self::LegacyMultiStorage(source) => {
                // For legacy format, read the entire blob then slice it
                let source = source.lock().unwrap();
                let data = source.read(key)?;
                let end = (offset + length).min(data.len());
                Ok(data[offset.min(data.len())..end].to_vec())
            }
        }
    }

    /// Check if a key exists
    pub fn contains(&self, key: &str) -> bool {
        match self {
            Self::Zip(source) => {
                let source = source.lock().unwrap();
                source.contains(key)
            }
            Self::File(_) => true, // File source always has its data
            Self::LegacyMultiStorage(_) => true, // Legacy format has all data
        }
    }

    /// Get list of available keys (for ZIP sources)
    pub fn keys(&self) -> Vec<String> {
        match self {
            Self::Zip(source) => {
                let source = source.lock().unwrap();
                source.data_files()
            }
            Self::File(_) => vec![],
            Self::LegacyMultiStorage(_) => vec![], // Legacy format doesn't have distinct keys
        }
    }
}
