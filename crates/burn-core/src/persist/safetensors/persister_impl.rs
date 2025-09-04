use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::convert::Into;
use hashbrown::HashMap;
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::persist::{
    KeyRemapper, ModulePersist, ModulePersister, PathFilter, TensorView, appliers::ApplyResult,
};
use crate::tensor::backend::Backend;
use burn_tensor::{Bytes, TensorData};

use super::format::{
    SafetensorsError, SafetensorsHeader, TensorInfo, read_header_size, write_header_size,
};

/// Configuration builder for SafetensorsPersister
pub struct SafetensorsPersisterConfig {
    // Core configuration fields
    filter: PathFilter,
    remapping: KeyRemapper,
    validate: bool,
    allow_partial: bool,

    // Safetensors-specific fields
    metadata: HashMap<String, String>,
}

impl Default for SafetensorsPersisterConfig {
    fn default() -> Self {
        Self {
            filter: PathFilter::all(), // Default to matching all tensors
            remapping: KeyRemapper::default(),
            validate: true,
            allow_partial: false,
            metadata: HashMap::new(),
        }
    }
}

impl SafetensorsPersisterConfig {
    /// Create a new SafetensorsPersisterConfig
    pub fn new() -> Self {
        Self::default()
    }

    /// Add metadata to the safetensors file
    pub fn with_metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set the tensor filter (replaces any existing filter)
    pub fn with_filter(mut self, filter: PathFilter) -> Self {
        self.filter = filter;
        self
    }

    /// Add regex patterns for filtering tensors (convenience method)
    #[cfg(target_has_atomic = "ptr")]
    pub fn filter_by_regex<I, S>(mut self, patterns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.filter = PathFilter::from_regex_patterns(patterns);
        self
    }

    /// Add specific tensor names to include (convenience method)
    pub fn filter_by_names<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.filter = PathFilter::from_paths(names);
        self
    }

    /// Add a predicate function for filtering (convenience method)
    pub fn filter_by_predicate(mut self, predicate: fn(&str, &str) -> bool) -> Self {
        self.filter = PathFilter::from_predicate(predicate);
        self
    }

    /// Add key remapping patterns
    #[cfg(target_has_atomic = "ptr")]
    pub fn with_remapping<S1: AsRef<str>, S2: AsRef<str>>(mut self, patterns: &[(S1, S2)]) -> Self {
        let patterns = patterns
            .iter()
            .map(|(p, r)| (p.as_ref().to_string(), r.as_ref().to_string()))
            .collect();
        self.remapping = KeyRemapper::from_patterns(patterns).expect("Invalid regex pattern");
        self
    }

    /// Enable or disable tensor validation
    pub fn with_validation(mut self, validate: bool) -> Self {
        self.validate = validate;
        self
    }

    /// Allow partial loading (missing tensors are skipped)
    pub fn allow_partial(mut self, allow: bool) -> Self {
        self.allow_partial = allow;
        self
    }

    /// Allow partial loading (missing tensors are skipped) - alias for allow_partial
    pub fn with_partial_loading(mut self, allow: bool) -> Self {
        self.allow_partial = allow;
        self
    }

    /// Build a SafetensorsPersister for a file
    pub fn build<P: AsRef<Path>>(self, path: P) -> Result<SafetensorsPersister, SafetensorsError> {
        Ok(SafetensorsPersister::File(FilePersister {
            path: path.as_ref().to_path_buf(),
            filter: self.filter,
            remapping: self.remapping,
            validate: self.validate,
            allow_partial: self.allow_partial,
            metadata: self.metadata,
        }))
    }

    /// Build an in-memory SafetensorsPersister (useful for testing)
    pub fn build_in_memory(self) -> SafetensorsMemoryPersister {
        SafetensorsMemoryPersister {
            data: None,
            filter: self.filter,
            remapping: self.remapping,
            validate: self.validate,
            allow_partial: self.allow_partial,
            metadata: self.metadata,
        }
    }
}

/// SafetensorsPersister implementation for file and memory-based persistence
pub enum SafetensorsPersister {
    File(FilePersister),
}

pub(crate) struct FilePersister {
    path: PathBuf,
    filter: PathFilter,
    remapping: KeyRemapper,
    validate: bool,
    allow_partial: bool,
    metadata: HashMap<String, String>,
}

impl FilePersister {
    /// Write tensors to file
    fn write_tensors(&self, views: Vec<TensorView>) -> Result<(), SafetensorsError> {
        // Convert Vec<TensorView> to HashMap for writing
        let mut views_map = HashMap::new();
        for view in views {
            let path = view.full_path();
            views_map.insert(path, view);
        }
        self.write_tensors_map(views_map)
    }

    fn write_tensors_map(
        &self,
        views: HashMap<String, TensorView>,
    ) -> Result<(), SafetensorsError> {
        let file = File::create(&self.path)
            .map_err(|e| SafetensorsError::Io(format!("Failed to create file: {}", e)))?;

        let mut writer = SafetensorsFileWriter {
            writer: file,
            header: SafetensorsHeader::new(),
            data_offset: 0,
            metadata: self.metadata.clone(),
        };

        // Add metadata to header
        for (key, value) in &self.metadata {
            writer.header = writer
                .header
                .clone()
                .with_metadata(key.clone(), value.clone());
        }

        // Collect tensor data
        let mut tensor_data: Vec<(String, TensorData)> = Vec::new();

        for (name, view) in views {
            let data = view.to_data();
            let start = writer.data_offset;
            let end = start + data.bytes.len();

            let info = TensorInfo::new(data.dtype, data.shape.clone(), start, end);
            writer.header.add_tensor(name.clone(), info);
            writer.data_offset = end;

            tensor_data.push((name, data));
        }

        // Write header
        writer.write_header()?;

        // Write tensor data
        for (_, data) in tensor_data {
            writer
                .writer
                .write_all(&data.bytes)
                .map_err(|e| SafetensorsError::Io(format!("Failed to write tensor data: {}", e)))?;
        }

        writer
            .writer
            .flush()
            .map_err(|e| SafetensorsError::Io(format!("Failed to flush: {}", e)))?;

        Ok(())
    }

    /// Read tensors from file
    fn read_tensors(&self) -> Result<Vec<TensorView>, SafetensorsError> {
        let mut file = File::open(&self.path)
            .map_err(|e| SafetensorsError::Io(format!("Failed to open file: {}", e)))?;

        // Read header size
        let mut size_bytes = [0u8; 8];
        file.read_exact(&mut size_bytes)
            .map_err(|e| SafetensorsError::Io(format!("Failed to read header size: {}", e)))?;

        let header_size = read_header_size(&size_bytes)? as usize;

        // Read header
        let mut header_bytes = vec![0u8; header_size];
        file.read_exact(&mut header_bytes)
            .map_err(|e| SafetensorsError::Io(format!("Failed to read header: {}", e)))?;

        let header = SafetensorsHeader::from_bytes(&header_bytes)?;

        // Validate offsets
        for (name, info) in &header.tensors {
            info.validate_offsets().map_err(|e| {
                SafetensorsError::InvalidFormat(format!(
                    "Invalid offsets for tensor {}: {}",
                    name, e
                ))
            })?;
        }

        let header_total_size = 8 + header_size;

        // Read all tensors
        let mut views = Vec::new();
        for (name, info) in header.tensors {
            let file_offset = header_total_size + info.data_offsets[0];
            let data_size = info.data_offsets[1] - info.data_offsets[0];

            // Seek and read tensor data
            file.seek(SeekFrom::Start(file_offset as u64))
                .map_err(|e| SafetensorsError::Io(format!("Failed to seek: {}", e)))?;

            let mut bytes = vec![0u8; data_size];
            file.read_exact(&mut bytes)
                .map_err(|e| SafetensorsError::Io(format!("Failed to read tensor data: {}", e)))?;

            let data = TensorData {
                bytes: Bytes::from_bytes_vec(bytes),
                shape: info.shape.clone(),
                dtype: info.to_burn_dtype(),
            };

            use crate::module::ParamId;
            let path_parts: Vec<String> = name.split('.').map(|s| s.to_string()).collect();
            let tensor_view = TensorView::from_data(
                data,
                path_parts,
                vec!["Safetensors".to_string()], // container stack
                ParamId::new(),                  // generate a new ID
            );
            views.push(tensor_view);
        }

        Ok(views)
    }

    /// Apply filtering to tensor views using the new PathFilter
    fn apply_filter(&self, views: Vec<TensorView>) -> Vec<TensorView> {
        views
            .into_iter()
            .filter(|view| {
                let path = view.full_path();
                self.filter.matches(&path)
            })
            .collect()
    }

    /// Apply key remapping to tensor views
    #[cfg(target_has_atomic = "ptr")]
    fn apply_remapping(&self, views: Vec<TensorView>) -> Vec<TensorView> {
        if self.remapping.is_empty() {
            return views;
        }

        // Apply remapping directly on the Vec
        let (remapped_views, _) = self.remapping.remap(views);
        remapped_views
    }

    #[cfg(not(target_has_atomic = "ptr"))]
    fn apply_remapping(&self, views: Vec<TensorView>) -> Vec<TensorView> {
        views
    }
}

impl ModulePersister for SafetensorsPersister {
    type Error = SafetensorsError;

    fn collect_from<B: Backend, M: ModulePersist<B>>(
        &mut self,
        module: &M,
    ) -> Result<(), Self::Error> {
        match self {
            SafetensorsPersister::File(persister) => {
                let mut views = module.collect();

                // Apply filtering
                views = persister.apply_filter(views);

                // Apply remapping for save
                views = persister.apply_remapping(views);

                // Write to file
                persister.write_tensors(views)?;

                Ok(())
            }
        }
    }

    fn apply_to<B: Backend, M: ModulePersist<B>>(
        &mut self,
        module: &mut M,
    ) -> Result<ApplyResult, Self::Error> {
        match self {
            SafetensorsPersister::File(persister) => {
                // Read from file
                let views = persister.read_tensors()?;

                // Apply the views to the module
                let result = module.apply(views);

                // Check for errors if validation is enabled
                if persister.validate && !result.errors.is_empty() {
                    return Err(SafetensorsError::ValidationFailed(format!(
                        "Import errors: {:?}",
                        result.errors
                    )));
                }

                // Check for missing tensors if partial loading is not allowed
                if !persister.allow_partial && !result.missing.is_empty() {
                    return Err(SafetensorsError::TensorNotFound(format!(
                        "Missing tensors: {:?}",
                        result.missing
                    )));
                }

                Ok(result)
            }
        }
    }
}

/// Memory-based persister for testing and in-memory operations
pub struct SafetensorsMemoryPersister {
    pub(crate) data: Option<Vec<u8>>,
    filter: PathFilter,
    remapping: KeyRemapper,
    validate: bool,
    allow_partial: bool,
    metadata: HashMap<String, String>,
}

impl SafetensorsMemoryPersister {
    /// Get the serialized data
    pub fn data(&self) -> Option<&Vec<u8>> {
        self.data.as_ref()
    }

    /// Set data for loading
    pub fn set_data(&mut self, data: Vec<u8>) {
        self.data = Some(data);
    }

    /// Write tensors to memory
    fn write_tensors(&mut self, views: Vec<TensorView>) -> Result<(), SafetensorsError> {
        // Convert Vec<TensorView> to HashMap for writing
        let mut views_map = HashMap::new();
        for view in views {
            let path = view.full_path();
            views_map.insert(path, view);
        }
        self.write_tensors_map(views_map)
    }

    fn write_tensors_map(
        &mut self,
        views: HashMap<String, TensorView>,
    ) -> Result<(), SafetensorsError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = SafetensorsFileWriter {
            writer: &mut buffer,
            header: SafetensorsHeader::new(),
            data_offset: 0,
            metadata: self.metadata.clone(),
        };

        // Add metadata to header
        for (key, value) in &self.metadata {
            writer.header = writer
                .header
                .clone()
                .with_metadata(key.clone(), value.clone());
        }

        // Collect tensor data
        let mut tensor_data: Vec<(String, TensorData)> = Vec::new();

        for (name, view) in views {
            let data = view.to_data();
            let start = writer.data_offset;
            let end = start + data.bytes.len();

            let info = TensorInfo::new(data.dtype, data.shape.clone(), start, end);
            writer.header.add_tensor(name.clone(), info);
            writer.data_offset = end;

            tensor_data.push((name, data));
        }

        // Write header
        writer.write_header()?;

        // Write tensor data
        for (_, data) in tensor_data {
            writer
                .writer
                .write_all(&data.bytes)
                .map_err(|e| SafetensorsError::Io(format!("Failed to write tensor data: {}", e)))?;
        }

        self.data = Some(buffer.into_inner());
        Ok(())
    }

    /// Read tensors from memory
    pub(crate) fn read_tensors(&self) -> Result<Vec<TensorView>, SafetensorsError> {
        let data = self
            .data
            .as_ref()
            .ok_or_else(|| SafetensorsError::InvalidFormat("No data available".to_string()))?;

        let mut cursor = Cursor::new(data);

        // Read header size
        let mut size_bytes = [0u8; 8];
        cursor
            .read_exact(&mut size_bytes)
            .map_err(|e| SafetensorsError::Io(format!("Failed to read header size: {}", e)))?;

        let header_size = read_header_size(&size_bytes)? as usize;

        // Read header
        let mut header_bytes = vec![0u8; header_size];
        cursor
            .read_exact(&mut header_bytes)
            .map_err(|e| SafetensorsError::Io(format!("Failed to read header: {}", e)))?;

        let header = SafetensorsHeader::from_bytes(&header_bytes)?;

        let header_total_size = 8 + header_size;

        // Read all tensors
        let mut views = Vec::new();
        for (name, info) in header.tensors {
            let file_offset = header_total_size + info.data_offsets[0];
            let data_size = info.data_offsets[1] - info.data_offsets[0];

            // Seek and read tensor data
            cursor
                .seek(SeekFrom::Start(file_offset as u64))
                .map_err(|e| SafetensorsError::Io(format!("Failed to seek: {}", e)))?;

            let mut bytes = vec![0u8; data_size];
            cursor
                .read_exact(&mut bytes)
                .map_err(|e| SafetensorsError::Io(format!("Failed to read tensor data: {}", e)))?;

            let tensor_data = TensorData {
                bytes: Bytes::from_bytes_vec(bytes),
                shape: info.shape.clone(),
                dtype: info.to_burn_dtype(),
            };

            use crate::module::ParamId;
            let path_parts: Vec<String> = name.split('.').map(|s| s.to_string()).collect();
            let tensor_view = TensorView::from_data(
                tensor_data,
                path_parts,
                vec!["SafetensorsMemory".to_string()], // container stack
                ParamId::new(),                        // generate a new ID
            );
            views.push(tensor_view);
        }

        Ok(views)
    }

    /// Apply filtering to tensor views using the new PathFilter
    fn apply_filter(&self, views: Vec<TensorView>) -> Vec<TensorView> {
        views
            .into_iter()
            .filter(|view| {
                let path = view.full_path();
                self.filter.matches(&path)
            })
            .collect()
    }

    /// Apply key remapping to tensor views
    #[cfg(target_has_atomic = "ptr")]
    fn apply_remapping(&self, views: Vec<TensorView>) -> Vec<TensorView> {
        if self.remapping.is_empty() {
            return views;
        }

        // Apply remapping directly on the Vec
        let (remapped_views, _) = self.remapping.remap(views);
        remapped_views
    }

    #[cfg(not(target_has_atomic = "ptr"))]
    fn apply_remapping(&self, views: Vec<TensorView>) -> Vec<TensorView> {
        views
    }
}

impl ModulePersister for SafetensorsMemoryPersister {
    type Error = SafetensorsError;

    fn collect_from<B: Backend, M: ModulePersist<B>>(
        &mut self,
        module: &M,
    ) -> Result<(), Self::Error> {
        let mut views = module.collect();

        // Apply filtering
        views = self.apply_filter(views);

        // Apply remapping for save
        views = self.apply_remapping(views);

        // Write to memory
        self.write_tensors(views)?;

        Ok(())
    }

    fn apply_to<B: Backend, M: ModulePersist<B>>(
        &mut self,
        module: &mut M,
    ) -> Result<ApplyResult, Self::Error> {
        // Read from memory
        let views = self.read_tensors()?;

        // Apply the views to the module
        let result = module.apply(views);

        // Check for errors if validation is enabled
        if self.validate && !result.errors.is_empty() {
            return Err(SafetensorsError::ValidationFailed(format!(
                "Import errors: {:?}",
                result.errors
            )));
        }

        // Check for missing tensors if partial loading is not allowed
        if !self.allow_partial && !result.missing.is_empty() {
            return Err(SafetensorsError::TensorNotFound(format!(
                "Missing tensors: {:?}",
                result.missing
            )));
        }

        Ok(result)
    }
}

/// Internal writer helper
struct SafetensorsFileWriter<W: Write + Seek> {
    writer: W,
    header: SafetensorsHeader,
    data_offset: usize,
    metadata: HashMap<String, String>,
}

impl<W: Write + Seek> SafetensorsFileWriter<W> {
    fn write_header(&mut self) -> Result<(), SafetensorsError> {
        // Serialize header to JSON
        let header_bytes = self.header.to_bytes()?;
        let header_size = header_bytes.len() as u64;

        // Write header size (8 bytes)
        self.writer
            .write_all(&write_header_size(header_size))
            .map_err(|e| SafetensorsError::Io(format!("Failed to write header size: {}", e)))?;

        // Write header JSON
        self.writer
            .write_all(&header_bytes)
            .map_err(|e| SafetensorsError::Io(format!("Failed to write header: {}", e)))?;

        Ok(())
    }
}
