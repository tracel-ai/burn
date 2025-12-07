//! PyTorch store implementation for saving and loading models in PyTorch format.

use crate::{
    ApplyResult, KeyRemapper, ModuleSnapshot, ModuleStore, PathFilter, PyTorchToBurnAdapter,
    TensorSnapshot,
};

use alloc::collections::BTreeMap;

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use burn_tensor::backend::Backend;
use core::fmt;
use std::path::PathBuf;

use super::reader::{PytorchError as ReaderError, PytorchReader};

/// Errors that can occur during PyTorch operations.
#[derive(Debug)]
pub enum PytorchStoreError {
    /// Reader error.
    Reader(ReaderError),

    /// I/O error.
    Io(std::io::Error),

    /// Tensor not found.
    TensorNotFound(String),

    /// Validation failed.
    ValidationFailed(String),

    /// Other error.
    Other(String),
}

impl fmt::Display for PytorchStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reader(e) => write!(f, "PyTorch reader error: {}", e),
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::TensorNotFound(name) => write!(f, "Tensor not found: {}", name),
            Self::ValidationFailed(msg) => write!(f, "Validation failed: {}", msg),
            Self::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for PytorchStoreError {}

impl From<ReaderError> for PytorchStoreError {
    fn from(e: ReaderError) -> Self {
        PytorchStoreError::Reader(e)
    }
}

impl From<std::io::Error> for PytorchStoreError {
    fn from(e: std::io::Error) -> Self {
        PytorchStoreError::Io(e)
    }
}

/// PyTorch store for file-based storage only.
///
/// This store allows loading models from PyTorch checkpoint files (.pt/.pth)
/// with automatic weight transformation using `PyTorchToBurnAdapter`.
/// Linear weights are automatically transposed and normalization parameters
/// are renamed (gamma -> weight, beta -> bias).
///
/// Note that saving to PyTorch format is not yet supported.
pub struct PytorchStore {
    pub(crate) path: PathBuf,
    pub(crate) filter: PathFilter,
    pub(crate) remapper: KeyRemapper,
    pub(crate) validate: bool,
    pub(crate) allow_partial: bool,
    pub(crate) top_level_key: Option<String>,
    pub(crate) skip_enum_variants: bool,
    /// Cached tensor snapshots (parsed once, reused)
    snapshots_cache: Option<BTreeMap<String, TensorSnapshot>>,
}

impl PytorchStore {
    /// Create a store for loading from a PyTorch file.
    ///
    /// # Arguments
    /// * `path` - Path to the PyTorch checkpoint file (.pt or .pth)
    ///
    /// # Example
    /// ```rust,no_run
    /// use burn_store::PytorchStore;
    ///
    /// let store = PytorchStore::from_file("model.pth");
    /// ```
    pub fn from_file(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            filter: PathFilter::new(),
            remapper: KeyRemapper::new(),
            validate: true,
            allow_partial: false,
            top_level_key: None,
            // PyTorch models never include enum variant names in paths
            skip_enum_variants: true,
            snapshots_cache: None,
        }
    }

    /// Set a top-level key to extract tensors from.
    ///
    /// PyTorch files often contain nested dictionaries. Use this to extract
    /// tensors from a specific top-level key like "state_dict" or "model_state_dict".
    ///
    /// # Example
    /// ```rust,no_run
    /// # use burn_store::PytorchStore;
    /// let store = PytorchStore::from_file("checkpoint.pth")
    ///     .with_top_level_key("model_state_dict");
    /// ```
    pub fn with_top_level_key(mut self, key: impl Into<String>) -> Self {
        self.top_level_key = Some(key.into());
        self
    }

    /// Filter which tensors to load.
    pub fn filter(mut self, filter: PathFilter) -> Self {
        self.filter = filter;
        self
    }

    /// Add a regex pattern to filter tensors.
    ///
    /// Multiple patterns can be added and they work with OR logic.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use burn_store::PytorchStore;
    /// let store = PytorchStore::from_file("model.pth")
    ///     .with_regex(r"^encoder\..*")  // Match all encoder tensors
    ///     .with_regex(r".*\.weight$");   // OR match any weight tensors
    /// ```
    pub fn with_regex<S: AsRef<str>>(mut self, pattern: S) -> Self {
        self.filter = self.filter.with_regex(pattern);
        self
    }

    /// Add multiple regex patterns to filter tensors.
    pub fn with_regexes<I, S>(mut self, patterns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.filter = self.filter.with_regexes(patterns);
        self
    }

    /// Add an exact full path to match.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use burn_store::PytorchStore;
    /// let store = PytorchStore::from_file("model.pth")
    ///     .with_full_path("encoder.layer1.weight")
    ///     .with_full_path("decoder.output.bias");
    /// ```
    pub fn with_full_path<S: Into<String>>(mut self, path: S) -> Self {
        self.filter = self.filter.with_full_path(path);
        self
    }

    /// Add multiple exact full paths to match.
    pub fn with_full_paths<I, S>(mut self, paths: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.filter = self.filter.with_full_paths(paths);
        self
    }

    /// Add a predicate function for custom filtering logic.
    ///
    /// The predicate receives the tensor path and container path.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use burn_store::PytorchStore;
    /// let store = PytorchStore::from_file("model.pth")
    ///     .with_predicate(|path, _| path.starts_with("encoder.") || path.ends_with(".bias"));
    /// ```
    pub fn with_predicate(mut self, predicate: fn(&str, &str) -> bool) -> Self {
        self.filter = self.filter.with_predicate(predicate);
        self
    }

    /// Add multiple predicate functions.
    pub fn with_predicates<I>(mut self, predicates: I) -> Self
    where
        I: IntoIterator<Item = fn(&str, &str) -> bool>,
    {
        self.filter = self.filter.with_predicates(predicates);
        self
    }

    /// Set the filter to match all paths (disables filtering).
    pub fn match_all(mut self) -> Self {
        self.filter = self.filter.match_all();
        self
    }

    /// Remap tensor names during load.
    pub fn remap(mut self, remapper: KeyRemapper) -> Self {
        self.remapper = remapper;
        self
    }

    /// Add a regex pattern to remap tensor names during load.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use burn_store::PytorchStore;
    /// let store = PytorchStore::from_file("model.pth")
    ///     .with_key_remapping(r"^encoder\.", "transformer.encoder.")  // encoder.X -> transformer.encoder.X
    ///     .with_key_remapping(r"\.gamma$", ".weight");               // X.gamma -> X.weight
    /// ```
    pub fn with_key_remapping(
        mut self,
        from_pattern: impl AsRef<str>,
        to_pattern: impl Into<String>,
    ) -> Self {
        self.remapper = self
            .remapper
            .add_pattern(from_pattern, to_pattern)
            .expect("Invalid regex pattern");
        self
    }

    /// Set whether to validate tensors during loading (default: true).
    pub fn validate(mut self, validate: bool) -> Self {
        self.validate = validate;
        self
    }

    /// Allow partial loading of tensors (continue even if some tensors are missing).
    pub fn allow_partial(mut self, allow: bool) -> Self {
        self.allow_partial = allow;
        self
    }

    /// Skip enum variant names when matching tensor paths (default: true).
    ///
    /// When enabled, tensor paths from PyTorch that don't include enum variants
    /// can be matched against Burn module paths that do include them.
    /// For example, PyTorch path "feature.weight" can match Burn path "feature.BaseConv.weight".
    ///
    /// This defaults to `true` for PytorchStore since PyTorch models never include
    /// enum variant names in their parameter paths.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use burn_store::PytorchStore;
    /// // Disable enum variant skipping (not typical)
    /// let store = PytorchStore::from_file("model.pth")
    ///     .skip_enum_variants(false);
    /// ```
    pub fn skip_enum_variants(mut self, skip: bool) -> Self {
        self.skip_enum_variants = skip;
        self
    }

    /// Apply filter to tensor snapshots.
    fn apply_filter(&self, mut snapshots: Vec<TensorSnapshot>) -> Vec<TensorSnapshot> {
        if self.filter.is_empty() {
            return snapshots;
        }

        snapshots.retain(|snapshot| {
            let path = snapshot.full_path();
            self.filter.matches(&path)
        });

        snapshots
    }

    /// Apply remapping to tensor snapshots.
    fn apply_remapping(&self, snapshots: Vec<TensorSnapshot>) -> Vec<TensorSnapshot> {
        if self.remapper.is_empty() {
            return snapshots;
        }

        let (remapped, _) = self.remapper.remap(snapshots);
        remapped
    }

    /// Create a PytorchReader for the configured path and options.
    fn create_reader(&self) -> Result<PytorchReader, PytorchStoreError> {
        let reader = if let Some(ref key) = self.top_level_key {
            PytorchReader::with_top_level_key(&self.path, key)?
        } else {
            PytorchReader::new(&self.path)?
        };
        Ok(reader)
    }
}

impl ModuleStore for PytorchStore {
    type Error = PytorchStoreError;

    fn collect_from<B: Backend, M: ModuleSnapshot<B>>(
        &mut self,
        _module: &M,
    ) -> Result<(), Self::Error> {
        // Saving to PyTorch format is not yet supported
        Err(PytorchStoreError::Other(
            "Saving to PyTorch format is not yet supported. Use other formats for saving."
                .to_string(),
        ))
    }

    fn apply_to<B: Backend, M: ModuleSnapshot<B>>(
        &mut self,
        module: &mut M,
    ) -> Result<ApplyResult, Self::Error> {
        // Get snapshots from cache
        let snapshots: Vec<TensorSnapshot> = self.get_snapshots()?.values().cloned().collect();

        // Apply to module with PyTorchToBurnAdapter (always used for PyTorch files)
        // This adapter handles:
        // - Transposing linear weights from PyTorch format to Burn format
        // - Renaming normalization parameters (gamma -> weight, beta -> bias)
        let result = module.apply(
            snapshots,
            None,
            Some(Box::new(PyTorchToBurnAdapter)),
            self.skip_enum_variants,
        );

        // Validate if needed
        if self.validate && !result.errors.is_empty() {
            return Err(PytorchStoreError::ValidationFailed(format!(
                "Import errors:\n{}",
                result
            )));
        }

        if !self.allow_partial && !result.missing.is_empty() {
            return Err(PytorchStoreError::TensorNotFound(format!("\n{}", result)));
        }

        Ok(result)
    }

    fn get_snapshot(&mut self, name: &str) -> Result<Option<&TensorSnapshot>, Self::Error> {
        self.ensure_snapshots_cache()?;
        Ok(self.snapshots_cache.as_ref().unwrap().get(name))
    }

    fn get_snapshots(&mut self) -> Result<&BTreeMap<String, TensorSnapshot>, Self::Error> {
        self.ensure_snapshots_cache()?;
        Ok(self.snapshots_cache.as_ref().unwrap())
    }

    fn keys(&mut self) -> Result<Vec<String>, Self::Error> {
        // Use cache if available
        if self.snapshots_cache.is_some() {
            return Ok(self
                .snapshots_cache
                .as_ref()
                .unwrap()
                .keys()
                .cloned()
                .collect());
        }

        // Fast path: read keys without full parsing
        let reader = self.create_reader()?;
        Ok(reader.keys())
    }
}

impl PytorchStore {
    /// Ensure the snapshots cache is populated
    fn ensure_snapshots_cache(&mut self) -> Result<(), PytorchStoreError> {
        if self.snapshots_cache.is_some() {
            return Ok(());
        }

        let reader = self.create_reader()?;

        // Convert to tensor snapshots
        let mut snapshots: Vec<TensorSnapshot> = reader
            .into_tensors()
            .into_iter()
            .map(|(key, mut snapshot)| {
                // Parse the key into path parts (split by '.')
                let path_parts: Vec<String> = key.split('.').map(|s| s.to_string()).collect();

                // Set the path stack from the key
                snapshot.path_stack = Some(path_parts);
                snapshot.container_stack = None;
                snapshot.tensor_id = None;

                snapshot
            })
            .collect();

        // Apply filtering
        snapshots = self.apply_filter(snapshots);

        // Apply remapping
        snapshots = self.apply_remapping(snapshots);

        // Build cache as BTreeMap
        let cache: BTreeMap<String, TensorSnapshot> =
            snapshots.into_iter().map(|s| (s.full_path(), s)).collect();

        self.snapshots_cache = Some(cache);
        Ok(())
    }
}
