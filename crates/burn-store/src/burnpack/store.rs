#[cfg(feature = "std")]
use std::path::PathBuf;

use super::reader::BurnpackReader;
use super::writer::BurnpackWriter;
#[cfg(feature = "std")]
use crate::KeyRemapper;
use crate::burnpack::base::BurnpackError;
use crate::{ModuleSnapshot, ModuleSnapshoter, PathFilter};
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use burn_core::prelude::Backend;

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
    /// Validate tensors during loading (check shapes and dtypes)
    validate: bool,
    /// Allow overwriting existing files (default: false)
    overwrite: bool,
    /// Automatically append .burnpack extension if not present (default: true)
    #[cfg(feature = "std")]
    auto_extension: bool,
    /// Key remapper for tensor name transformations
    #[cfg(feature = "std")]
    remapper: KeyRemapper,
    /// Writer for saving
    writer: Option<BurnpackWriter>,
    /// Reader for loading
    reader: Option<BurnpackReader>,
}

impl BurnpackStore {
    /// Get the default metadata that includes Burn framework information.
    ///
    /// This includes:
    /// - `format`: "burnpack"
    /// - `producer`: "burn"
    /// - `version`: The version of burn-store crate (from CARGO_PKG_VERSION)
    ///
    /// These metadata fields are automatically added to all saved models.
    pub fn default_metadata() -> BTreeMap<String, String> {
        let mut metadata = BTreeMap::new();
        metadata.insert("format".into(), "burnpack".into());
        metadata.insert("producer".into(), "burn".into());
        metadata.insert("version".into(), env!("CARGO_PKG_VERSION").into());
        metadata
    }
    /// Create a new store from a file path
    ///
    /// By default, automatically appends `.burnpack` extension if the path doesn't have one.
    /// Use `.auto_extension(false)` to disable this behavior.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use burn_store::BurnpackStore;
    /// // Automatically appends .burnpack
    /// let store = BurnpackStore::from_file("model");  // creates "model.burnpack"
    ///
    /// // Already has extension, no append
    /// let store = BurnpackStore::from_file("model.burnpack");  // uses "model.burnpack"
    /// let store = BurnpackStore::from_file("model.myext");  // uses "model.myext"
    ///
    /// // Disable auto-extension
    /// let store = BurnpackStore::from_file("model").auto_extension(false);  // uses "model"
    /// ```
    #[cfg(feature = "std")]
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Self {
        Self {
            mode: StoreMode::File(path.as_ref().to_path_buf()),
            filter: None,
            metadata: Self::default_metadata(),
            allow_partial: false,
            validate: true,
            overwrite: false,
            #[cfg(feature = "std")]
            auto_extension: true,
            #[cfg(feature = "std")]
            remapper: KeyRemapper::new(),
            writer: None,
            reader: None,
        }
    }

    /// Create a new store from bytes (for reading) or empty (for writing)
    pub fn from_bytes(bytes: Option<Vec<u8>>) -> Self {
        Self {
            mode: StoreMode::Bytes(bytes),
            filter: None,
            metadata: Self::default_metadata(),
            allow_partial: false,
            validate: true,
            overwrite: false,
            #[cfg(feature = "std")]
            auto_extension: false, // Not used for bytes mode
            #[cfg(feature = "std")]
            remapper: KeyRemapper::new(),
            writer: None,
            reader: None,
        }
    }

    /// Add metadata key-value pair
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Clear all metadata (including defaults)
    ///
    /// This removes all metadata including the default format, producer, and version fields.
    /// Use with caution as some tools may expect these fields to be present.
    pub fn clear_metadata(mut self) -> Self {
        self.metadata.clear();
        self
    }

    /// Allow partial loading (ignore missing tensors)
    ///
    /// When set to `true`, the store will not fail if some tensors are missing
    /// during loading. This is useful when loading a subset of a model's parameters.
    ///
    /// Default: `false`
    pub fn allow_partial(mut self, allow: bool) -> Self {
        self.allow_partial = allow;
        self
    }

    /// Enable or disable validation during loading
    ///
    /// When validation is enabled, the store will check that loaded tensors
    /// match the expected shapes and data types. Disabling validation can
    /// improve performance but may lead to runtime errors if data is corrupted.
    ///
    /// Default: `true`
    pub fn validate(mut self, validate: bool) -> Self {
        self.validate = validate;
        self
    }

    /// Allow overwriting existing files when saving
    ///
    /// When set to `false`, attempting to save to an existing file will result in an error.
    /// When set to `true`, existing files will be overwritten without warning.
    ///
    /// Default: `false`
    pub fn overwrite(mut self, overwrite: bool) -> Self {
        self.overwrite = overwrite;
        self
    }

    /// Enable or disable automatic .burnpack extension appending
    ///
    /// When enabled (default), automatically appends `.burnpack` to the file path
    /// if no extension is detected. If an extension is already present, it is preserved.
    ///
    /// When disabled, uses the exact path provided without modification.
    ///
    /// Default: `true`
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use burn_store::BurnpackStore;
    /// // With auto_extension enabled (default)
    /// let store = BurnpackStore::from_file("model");  // -> "model.burnpack"
    ///
    /// // With auto_extension disabled
    /// let store = BurnpackStore::from_file("model")
    ///     .auto_extension(false);  // -> "model"
    /// ```
    #[cfg(feature = "std")]
    pub fn auto_extension(mut self, enable: bool) -> Self {
        self.auto_extension = enable;
        self
    }

    /// Set path filter for selective loading/saving
    pub fn with_filter(mut self, filter: PathFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Add regex pattern to filter
    #[cfg(feature = "std")]
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

    /// Set key remapper for tensor name transformations during loading
    #[cfg(feature = "std")]
    pub fn remap(mut self, remapper: KeyRemapper) -> Self {
        self.remapper = remapper;
        self
    }

    /// Add a single regex pattern for key remapping
    #[cfg(feature = "std")]
    pub fn with_remap_pattern<S1, S2>(mut self, from: S1, to: S2) -> Self
    where
        S1: AsRef<str>,
        S2: Into<String>,
    {
        self.remapper = self
            .remapper
            .add_pattern(from.as_ref(), to.into())
            .expect("Invalid regex pattern");
        self
    }

    /// Set the path filter
    pub fn filter(mut self, filter: PathFilter) -> Self {
        self.filter = Some(filter);
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

    /// Process the file path with auto-extension logic
    #[cfg(feature = "std")]
    fn process_path(&self, path: &std::path::Path) -> PathBuf {
        if !self.auto_extension {
            return path.to_path_buf();
        }

        // Check if path already has an extension
        if path.extension().is_some() {
            // Has extension, use as-is
            return path.to_path_buf();
        }

        // No extension, append .burnpack
        let mut new_path = path.to_path_buf();
        new_path.set_extension("burnpack");
        new_path
    }
}

impl ModuleSnapshoter for BurnpackStore {
    type Error = BurnpackError;

    fn collect_from<B: Backend, M: ModuleSnapshot<B>>(
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
            match &self.mode {
                #[cfg(feature = "std")]
                StoreMode::File(path) => {
                    // Process path with auto-extension logic
                    let final_path = self.process_path(path);

                    // Check if file exists and overwrite is disabled
                    if final_path.exists() && !self.overwrite {
                        return Err(BurnpackError::IoError(format!(
                            "File already exists: {}. Use .overwrite(true) to overwrite.",
                            final_path.display()
                        )));
                    }
                    writer.write_to_file(&final_path)?;
                }
                StoreMode::Bytes(_) => {
                    // Generate and store the bytes - need to handle this separately due to mutability
                    let bytes_data = writer.to_bytes()?;
                    // Now update the mode with the bytes
                    if let StoreMode::Bytes(bytes_ref) = &mut self.mode {
                        *bytes_ref = Some(bytes_data);
                    }
                }
            }
        }

        Ok(())
    }

    fn apply_to<B: Backend, M: ModuleSnapshot<B>>(
        &mut self,
        module: &mut M,
    ) -> Result<crate::ApplyResult, Self::Error> {
        // Load reader if not already loaded
        if self.reader.is_none() {
            let reader = match &self.mode {
                #[cfg(feature = "std")]
                StoreMode::File(path) => {
                    // Process path with auto-extension logic
                    let final_path = self.process_path(path);
                    BurnpackReader::from_file(&final_path)?
                }
                StoreMode::Bytes(Some(bytes)) => BurnpackReader::from_bytes(bytes.clone())?,
                StoreMode::Bytes(None) => {
                    return Err(BurnpackError::IoError("No bytes to read from".into()));
                }
            };
            self.reader = Some(reader);
        }

        let reader = self
            .reader
            .as_ref()
            .ok_or_else(|| BurnpackError::IoError("Reader not initialized".into()))?;

        // Get all snapshots at once for efficient loading
        #[cfg(feature = "std")]
        let snapshots = if !self.remapper.patterns.is_empty() {
            let (remapped, _remapped_names) = self.remapper.remap(reader.get_snapshots());
            // TODO figure what to do with remapped names
            remapped
        } else {
            reader.get_snapshots()
        };

        #[cfg(not(feature = "std"))]
        let snapshots = reader.get_snapshots();

        // Apply all snapshots at once to the module
        let result = module.apply(snapshots, self.filter.clone(), None);

        // Validate if needed
        if self.validate && !result.errors.is_empty() {
            return Err(BurnpackError::ValidationError(format!(
                "Import errors: {:?}",
                result.errors
            )));
        }

        // Check for missing tensors if partial loading is not allowed
        if !self.allow_partial && !result.missing.is_empty() {
            return Err(BurnpackError::ValidationError(format!(
                "Missing tensors: {:?}",
                result.missing
            )));
        }

        Ok(result)
    }
}
