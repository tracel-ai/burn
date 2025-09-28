#[cfg(feature = "std")]
use std::path::PathBuf;

use super::reader::BurnpackReader;
use super::writer::BurnpackWriter;
use crate::burnpack::base::BurnpackError;
use crate::{ModuleSnapshot, ModuleSnapshoter, PathFilter};
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// Store mode for BurnpackStore
enum StoreMode {
    #[cfg(feature = "std")]
    File(PathBuf),
    Bytes(Option<Vec<u8>>),
}

/// BurnpackStore - A Burn-specific file format store using MessagePack for metadata
pub struct BurnpackStore {
    /// Store mode - either file path or bytes
    mode: StoreMode,
    /// Optional filter for selective loading/saving
    filter: Option<PathFilter>,
    /// Additional metadata
    metadata: BTreeMap<String, String>,
    /// Allow partial loading (ignore missing tensors)
    allow_partial: bool,
    /// Writer for saving
    writer: Option<BurnpackWriter>,
    /// Reader for loading
    reader: Option<BurnpackReader>,
}

impl BurnpackStore {
    /// Create a new store from a file path
    #[cfg(feature = "std")]
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Self {
        Self {
            mode: StoreMode::File(path.as_ref().to_path_buf()),
            filter: None,
            metadata: BTreeMap::new(),
            allow_partial: false,
            writer: None,
            reader: None,
        }
    }

    /// Create a new store from bytes (for reading) or empty (for writing)
    pub fn from_bytes(bytes: Option<Vec<u8>>) -> Self {
        Self {
            mode: StoreMode::Bytes(bytes),
            filter: None,
            metadata: BTreeMap::new(),
            allow_partial: false,
            writer: None,
            reader: None,
        }
    }

    /// Add metadata key-value pair
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Allow partial loading (ignore missing tensors)
    pub fn allow_partial(mut self, allow: bool) -> Self {
        self.allow_partial = allow;
        self
    }

    /// Set path filter for selective loading/saving
    pub fn with_filter(mut self, filter: PathFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Add regex pattern to filter
    pub fn with_regex(mut self, pattern: &str) -> Self {
        let filter = self.filter.unwrap_or_default();
        self.filter = Some(filter.with_regex(pattern));
        self
    }

    /// Add exact path to filter
    pub fn with_full_path(mut self, path: impl Into<String>) -> Self {
        let filter = self.filter.unwrap_or_default();
        self.filter = Some(filter.with_full_path(path));
        self
    }

    /// Match all tensors (no filtering)
    pub fn match_all(mut self) -> Self {
        self.filter = Some(PathFilter::new().match_all());
        self
    }

    /// Get the bytes after writing (only valid for bytes mode after collecting)
    pub fn get_bytes(&self) -> Result<Vec<u8>, BurnpackError> {
        if let Some(writer) = &self.writer {
            return writer.to_bytes();
        }

        match &self.mode {
            StoreMode::Bytes(Some(bytes)) => Ok(bytes.clone()),
            _ => Err(BurnpackError::IoError("No bytes available".into())),
        }
    }
}

impl ModuleSnapshoter for BurnpackStore {
    type Error = BurnpackError;

    fn collect_from<B: burn_tensor::backend::Backend, M: ModuleSnapshot<B>>(
        &mut self,
        module: &M,
    ) -> Result<(), Self::Error> {
        // Collect snapshots from module
        let snapshots = module.collect(self.filter.clone(), None);

        // Initialize writer with snapshots
        let mut writer = BurnpackWriter::new(snapshots);

        // Add metadata using builder pattern
        for (key, value) in &self.metadata {
            writer = writer.with_metadata(key.as_str(), value.as_str());
        }

        // Store the writer for finalization
        self.writer = Some(writer);

        // Write to storage based on mode
        if let Some(writer) = &self.writer {
            match &mut self.mode {
                #[cfg(feature = "std")]
                StoreMode::File(path) => {
                    writer.write_to_file(path)?;
                }
                StoreMode::Bytes(bytes) => {
                    // Generate and store the bytes
                    *bytes = Some(writer.to_bytes()?);
                }
            }
        }

        Ok(())
    }

    fn apply_to<B: burn_tensor::backend::Backend, M: ModuleSnapshot<B>>(
        &mut self,
        module: &mut M,
    ) -> Result<crate::ApplyResult, Self::Error> {
        // Load reader if not already loaded
        if self.reader.is_none() {
            self.reader = Some(match &self.mode {
                #[cfg(feature = "std")]
                StoreMode::File(path) => BurnpackReader::from_file(path)?,
                StoreMode::Bytes(Some(bytes)) => BurnpackReader::from_bytes(bytes.clone())?,
                StoreMode::Bytes(None) => {
                    return Err(BurnpackError::IoError("No bytes to read from".into()));
                }
            });
        }

        let reader = self.reader.as_ref().unwrap();

        // Get all snapshots at once for efficient loading
        let snapshots = reader.get_snapshots();

        // TODO add remapping

        // Apply all snapshots at once to the module
        // The apply method returns the actual ApplyResult with details
        let result = module.apply(snapshots, self.filter.clone(), None);

        // TODO handle IO errors

        Ok(result)
    }
}
