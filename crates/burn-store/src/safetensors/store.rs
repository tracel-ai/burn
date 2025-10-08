//! SafeTensors store implementation using the official safetensors crate.

use crate::{
    ApplyResult, IdentityAdapter, ModuleAdapter, ModuleSnapshot, ModuleSnapshoter, PathFilter,
    TensorSnapshot,
};

#[cfg(feature = "std")]
use crate::KeyRemapper;
use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use burn_core::module::ParamId;
use burn_tensor::backend::Backend;
use burn_tensor::{DType, TensorData};
use core::fmt;
use core::ops::Deref;
use hashbrown::HashMap;

// Arc is only available on targets with atomic pointers
#[cfg(target_has_atomic = "ptr")]
use alloc::sync::Arc;

// For targets without atomic pointers, we use Box instead
#[cfg(not(target_has_atomic = "ptr"))]
type Arc<T> = Box<T>;

/// Errors that can occur during SafeTensors operations.
#[derive(Debug)]
pub enum SafetensorsStoreError {
    /// SafeTensors crate error.
    Safetensors(safetensors::SafeTensorError),

    /// I/O error.
    #[cfg(feature = "std")]
    Io(std::io::Error),

    /// Tensor not found.
    TensorNotFound(String),

    /// Validation failed.
    ValidationFailed(String),

    /// Other error.
    Other(String),
}

impl fmt::Display for SafetensorsStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Safetensors(e) => write!(f, "SafeTensors error: {}", e),
            #[cfg(feature = "std")]
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::TensorNotFound(name) => write!(f, "Tensor not found: {}", name),
            Self::ValidationFailed(msg) => write!(f, "Validation failed: {}", msg),
            Self::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl core::error::Error for SafetensorsStoreError {}

impl From<safetensors::SafeTensorError> for SafetensorsStoreError {
    fn from(e: safetensors::SafeTensorError) -> Self {
        SafetensorsStoreError::Safetensors(e)
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for SafetensorsStoreError {
    fn from(e: std::io::Error) -> Self {
        SafetensorsStoreError::Io(e)
    }
}

/// SafeTensors store supporting both file and memory storage.
pub enum SafetensorsStore {
    /// File-based storage.
    #[cfg(feature = "std")]
    File(FileStore),

    /// Memory-based storage.
    Memory(MemoryStore),
}

impl Default for SafetensorsStore {
    /// Create a default memory-based store.
    fn default() -> Self {
        Self::from_bytes(None)
    }
}

impl SafetensorsStore {
    /// Get the default metadata that includes Burn framework information.
    ///
    /// This includes:
    /// - `format`: "safetensors"
    /// - `producer`: "burn"
    /// - `version`: The version of burn-store crate (from CARGO_PKG_VERSION)
    ///
    /// These metadata fields are automatically added to all saved models.
    pub fn default_metadata() -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "safetensors".to_string());
        metadata.insert("producer".to_string(), "burn".to_string());
        metadata.insert("version".to_string(), env!("CARGO_PKG_VERSION").to_string());
        metadata
    }

    /// Create a store for loading from or saving to a file.
    #[cfg(feature = "std")]
    pub fn from_file(path: impl Into<std::path::PathBuf>) -> Self {
        Self::File(FileStore {
            path: path.into(),
            filter: PathFilter::new(),
            remapper: KeyRemapper::new(),
            metadata: Self::default_metadata(),
            validate: true,
            allow_partial: false,
            from_adapter: Box::new(IdentityAdapter),
            to_adapter: Box::new(IdentityAdapter),
        })
    }

    /// Create a store for working with bytes in memory.
    pub fn from_bytes(bytes: Option<Vec<u8>>) -> Self {
        Self::Memory(MemoryStore {
            data: bytes.map(Arc::new),
            filter: PathFilter::new(),
            #[cfg(feature = "std")]
            remapper: KeyRemapper::new(),
            metadata: Self::default_metadata(),
            validate: true,
            allow_partial: false,
            from_adapter: Box::new(IdentityAdapter),
            to_adapter: Box::new(IdentityAdapter),
        })
    }

    /// Filter which tensors to load/save.
    pub fn filter(mut self, filter: PathFilter) -> Self {
        match &mut self {
            #[cfg(feature = "std")]
            Self::File(p) => p.filter = filter,
            Self::Memory(p) => p.filter = filter,
        }
        self
    }

    /// Add a regex pattern to filter tensors.
    ///
    /// Multiple patterns can be added and they work with OR logic.
    ///
    /// # Example
    /// ```rust,ignore
    /// let store = SafetensorsStore::from_file("model.safetensors")
    ///     .with_regex(r"^encoder\..*")  // Match all encoder tensors
    ///     .with_regex(r".*\.weight$");   // OR match any weight tensors
    /// ```
    #[cfg(feature = "std")]
    pub fn with_regex<S: AsRef<str>>(mut self, pattern: S) -> Self {
        match &mut self {
            #[cfg(feature = "std")]
            Self::File(p) => p.filter = p.filter.clone().with_regex(pattern),
            Self::Memory(p) => p.filter = p.filter.clone().with_regex(pattern),
        }
        self
    }

    /// Add multiple regex patterns to filter tensors.
    #[cfg(feature = "std")]
    pub fn with_regexes<I, S>(mut self, patterns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        match &mut self {
            #[cfg(feature = "std")]
            Self::File(p) => p.filter = p.filter.clone().with_regexes(patterns),
            Self::Memory(p) => p.filter = p.filter.clone().with_regexes(patterns),
        }
        self
    }

    /// Add an exact full path to match.
    ///
    /// # Example
    /// ```rust,ignore
    /// let store = SafetensorsStore::from_file("model.safetensors")
    ///     .with_full_path("encoder.layer1.weight")
    ///     .with_full_path("decoder.output.bias");
    /// ```
    pub fn with_full_path<S: Into<String>>(mut self, path: S) -> Self {
        match &mut self {
            #[cfg(feature = "std")]
            Self::File(p) => p.filter = p.filter.clone().with_full_path(path),
            Self::Memory(p) => p.filter = p.filter.clone().with_full_path(path),
        }
        self
    }

    /// Add multiple exact full paths to match.
    pub fn with_full_paths<I, S>(mut self, paths: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        match &mut self {
            #[cfg(feature = "std")]
            Self::File(p) => p.filter = p.filter.clone().with_full_paths(paths),
            Self::Memory(p) => p.filter = p.filter.clone().with_full_paths(paths),
        }
        self
    }

    /// Add a predicate function for custom filtering logic.
    ///
    /// The predicate receives the tensor path and container path.
    ///
    /// # Example
    /// ```rust,ignore
    /// let store = SafetensorsStore::from_file("model.safetensors")
    ///     .with_predicate(|path, _| path.starts_with("encoder.") || path.ends_with(".bias"));
    /// ```
    pub fn with_predicate(mut self, predicate: fn(&str, &str) -> bool) -> Self {
        match &mut self {
            #[cfg(feature = "std")]
            Self::File(p) => p.filter = p.filter.clone().with_predicate(predicate),
            Self::Memory(p) => p.filter = p.filter.clone().with_predicate(predicate),
        }
        self
    }

    /// Add multiple predicate functions.
    pub fn with_predicates<I>(mut self, predicates: I) -> Self
    where
        I: IntoIterator<Item = fn(&str, &str) -> bool>,
    {
        match &mut self {
            #[cfg(feature = "std")]
            Self::File(p) => p.filter = p.filter.clone().with_predicates(predicates),
            Self::Memory(p) => p.filter = p.filter.clone().with_predicates(predicates),
        }
        self
    }

    /// Set the filter to match all paths (disables filtering).
    pub fn match_all(mut self) -> Self {
        match &mut self {
            #[cfg(feature = "std")]
            Self::File(p) => p.filter = p.filter.clone().match_all(),
            Self::Memory(p) => p.filter = p.filter.clone().match_all(),
        }
        self
    }

    /// Remap tensor names during load/save.
    #[cfg(feature = "std")]
    pub fn remap(mut self, remapper: KeyRemapper) -> Self {
        match &mut self {
            Self::File(p) => p.remapper = remapper,
            Self::Memory(p) => p.remapper = remapper,
        }
        self
    }

    /// Add a regex pattern to remap tensor names during load/save.
    ///
    /// # Example
    /// ```rust,ignore
    /// let store = SafetensorsStore::from_file("model.safetensors")
    ///     .with_key_remapping(r"^encoder\.", "transformer.encoder.")  // encoder.X -> transformer.encoder.X
    ///     .with_key_remapping(r"\.gamma$", ".weight");               // X.gamma -> X.weight
    /// ```
    #[cfg(feature = "std")]
    pub fn with_key_remapping(
        mut self,
        from_pattern: impl AsRef<str>,
        to_pattern: impl Into<String>,
    ) -> Self {
        match &mut self {
            Self::File(p) => {
                p.remapper = p
                    .remapper
                    .clone()
                    .add_pattern(from_pattern, to_pattern)
                    .expect("Invalid regex pattern");
            }
            Self::Memory(p) => {
                p.remapper = p
                    .remapper
                    .clone()
                    .add_pattern(from_pattern, to_pattern)
                    .expect("Invalid regex pattern");
            }
        }
        self
    }

    /// Add metadata to be saved with the tensors.
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        let key = key.into();
        let value = value.into();
        match &mut self {
            #[cfg(feature = "std")]
            Self::File(p) => {
                p.metadata.insert(key, value);
            }
            Self::Memory(p) => {
                p.metadata.insert(key, value);
            }
        }
        self
    }

    /// Clear all metadata including the default Burn framework metadata.
    ///
    /// This removes the automatic `format`, `producer` and `version` fields.
    /// Use this when you need complete control over metadata or when
    /// saving models for use with other frameworks.
    pub fn clear_metadata(mut self) -> Self {
        match &mut self {
            #[cfg(feature = "std")]
            Self::File(p) => {
                p.metadata.clear();
            }
            Self::Memory(p) => {
                p.metadata.clear();
            }
        }
        self
    }

    /// Set whether to validate tensors during loading (default: true).
    pub fn validate(mut self, validate: bool) -> Self {
        match &mut self {
            #[cfg(feature = "std")]
            Self::File(p) => p.validate = validate,
            Self::Memory(p) => p.validate = validate,
        }
        self
    }

    /// Allow partial loading of tensors (continue even if some tensors are missing).
    pub fn allow_partial(mut self, allow: bool) -> Self {
        match &mut self {
            #[cfg(feature = "std")]
            Self::File(p) => p.allow_partial = allow,
            Self::Memory(p) => p.allow_partial = allow,
        }
        self
    }

    /// Set the adapter for loading tensors (converting from source format to Burn).
    pub fn with_from_adapter(mut self, adapter: impl ModuleAdapter + 'static) -> Self {
        match &mut self {
            #[cfg(feature = "std")]
            Self::File(p) => p.from_adapter = Box::new(adapter),
            Self::Memory(p) => p.from_adapter = Box::new(adapter),
        }
        self
    }

    /// Set the adapter for saving tensors (converting from Burn to target format).
    pub fn with_to_adapter(mut self, adapter: impl ModuleAdapter + 'static) -> Self {
        match &mut self {
            #[cfg(feature = "std")]
            Self::File(p) => p.to_adapter = Box::new(adapter),
            Self::Memory(p) => p.to_adapter = Box::new(adapter),
        }
        self
    }

    /// Get saved bytes from memory-based store.
    ///
    /// # Example
    /// ```rust,ignore
    /// let mut store = SafetensorsStore::from_bytes(None);
    /// model.collect_to(&mut store)?;
    /// let bytes = store.get_bytes()?;
    /// ```
    pub fn get_bytes(&self) -> Result<Vec<u8>, SafetensorsStoreError> {
        match self {
            #[cfg(feature = "std")]
            Self::File(_) => Err(SafetensorsStoreError::Other(
                "Cannot get bytes from file-based store".to_string(),
            )),
            Self::Memory(p) => p
                .data()
                .map(|arc| arc.as_ref().clone())
                .ok_or_else(|| SafetensorsStoreError::Other("No data available".to_string())),
        }
    }
}

/// File-based store.
#[cfg(feature = "std")]
pub struct FileStore {
    path: std::path::PathBuf,
    filter: PathFilter,
    remapper: KeyRemapper,
    metadata: HashMap<String, String>,
    validate: bool,
    allow_partial: bool,
    from_adapter: Box<dyn ModuleAdapter>,
    to_adapter: Box<dyn ModuleAdapter>,
}

/// Memory-based store.
pub struct MemoryStore {
    data: Option<Arc<Vec<u8>>>,
    filter: PathFilter,
    #[cfg(feature = "std")]
    remapper: KeyRemapper,
    metadata: HashMap<String, String>,
    validate: bool,
    allow_partial: bool,
    from_adapter: Box<dyn ModuleAdapter>,
    to_adapter: Box<dyn ModuleAdapter>,
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self {
            data: None,
            filter: PathFilter::new(),
            #[cfg(feature = "std")]
            remapper: KeyRemapper::new(),
            metadata: HashMap::new(),
            validate: true,
            allow_partial: false,
            from_adapter: Box::new(IdentityAdapter),
            to_adapter: Box::new(IdentityAdapter),
        }
    }
}

impl MemoryStore {
    #[cfg(test)]
    pub(crate) fn data(&self) -> Option<Arc<Vec<u8>>> {
        self.data.clone()
    }

    #[cfg(not(test))]
    fn data(&self) -> Option<Arc<Vec<u8>>> {
        self.data.clone()
    }

    #[cfg(test)]
    pub(crate) fn set_data(&mut self, data: Vec<u8>) {
        self.data = Some(Arc::new(data));
    }
}

// Adapter to use TensorSnapshot directly with safetensors
struct TensorSnapshotAdapter(TensorSnapshot);

impl safetensors::View for TensorSnapshotAdapter {
    fn dtype(&self) -> safetensors::Dtype {
        // Convert from burn dtype to safetensors dtype
        dtype_to_safetensors(self.0.dtype).unwrap_or(safetensors::Dtype::F32)
    }

    fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    fn data(&self) -> alloc::borrow::Cow<'_, [u8]> {
        // Only materialize data when actually needed for serialization
        let data = self
            .0
            .to_data()
            .unwrap_or_else(|e| panic!("Failed to get tensor data: {:?}", e));
        alloc::borrow::Cow::Owned(data.bytes.deref().to_vec())
    }

    fn data_len(&self) -> usize {
        // Use the efficient data_len method from TensorSnapshot
        self.0.data_len()
    }
}

impl ModuleSnapshoter for SafetensorsStore {
    type Error = SafetensorsStoreError;

    fn collect_from<B: Backend, M: ModuleSnapshot<B>>(
        &mut self,
        module: &M,
    ) -> Result<(), Self::Error> {
        // Collect tensor snapshots from module with adapter
        // The to_adapter converts from Burn format to target format for saving
        let to_adapter = match self {
            #[cfg(feature = "std")]
            Self::File(p) => p.to_adapter.clone(),
            Self::Memory(p) => p.to_adapter.clone(),
        };
        let mut snapshots = module.collect(None, Some(to_adapter));

        // Apply filtering
        snapshots = apply_filter(snapshots, self.get_filter());

        // Apply remapping
        #[cfg(feature = "std")]
        {
            snapshots = apply_remapping(snapshots, self.get_remapper());
        }

        // Get metadata (already includes format, producer and version from default_metadata)
        let metadata = self.get_metadata().clone();

        #[cfg(feature = "std")]
        let std_metadata: std::collections::HashMap<String, String> = metadata
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        // Write to storage
        match self {
            #[cfg(feature = "std")]
            Self::File(p) => {
                // Convert to safetensors format
                let tensors = snapshots_to_safetensors(snapshots)?;

                // Use serialize_to_file which streams directly to disk
                // This calls the lazy closures on-demand without buffering everything
                safetensors::serialize_to_file(tensors, Some(std_metadata), &p.path)?;
                Ok(())
            }
            Self::Memory(p) => {
                // For memory, we need to serialize to bytes
                let tensors = snapshots_to_safetensors(snapshots)?;
                // For no-std, serialize still needs std HashMap when std feature is enabled
                #[cfg(feature = "std")]
                let data = safetensors::serialize(tensors, Some(std_metadata))?;

                #[cfg(not(feature = "std"))]
                let data = safetensors::serialize(tensors, Some(metadata))?;
                p.data = Some(Arc::new(data));
                Ok(())
            }
        }
    }

    fn apply_to<B: Backend, M: ModuleSnapshot<B>>(
        &mut self,
        module: &mut M,
    ) -> Result<ApplyResult, Self::Error> {
        // Convert to tensor snapshots with lazy loading
        #[allow(unused_mut)]
        let mut snapshots = match self {
            #[cfg(feature = "std")]
            Self::File(p) => {
                // Use safetensors' built-in lazy loading mechanisms
                safetensors_to_snapshots_lazy_file(&p.path)?
            }
            Self::Memory(p) => {
                let data_arc = p
                    .data
                    .clone()
                    .ok_or_else(|| SafetensorsStoreError::Other("No data loaded".to_string()))?;
                safetensors_to_snapshots_lazy(data_arc)?
            }
        };

        // Apply remapping to loaded tensors
        #[cfg(feature = "std")]
        {
            snapshots = match self {
                Self::File(p) => apply_remapping(snapshots, &p.remapper),
                Self::Memory(p) => apply_remapping(snapshots, &p.remapper),
            };
        }

        // Get the adapter
        let adapter: Box<dyn ModuleAdapter> = match self {
            #[cfg(feature = "std")]
            Self::File(p) => p.from_adapter.clone(),
            Self::Memory(p) => p.from_adapter.clone(),
        };

        // Apply to module with adapter
        // The adapter will be applied during module traversal with proper container info
        let result = module.apply(snapshots, None, Some(adapter));

        // Validate if needed
        if self.get_validate() && !result.errors.is_empty() {
            return Err(SafetensorsStoreError::ValidationFailed(format!(
                "Import errors: {:?}",
                result.errors
            )));
        }

        if !self.get_allow_partial() && !result.missing.is_empty() {
            return Err(SafetensorsStoreError::TensorNotFound(format!(
                "Missing tensors: {:?}",
                result.missing
            )));
        }

        Ok(result)
    }
}

impl SafetensorsStore {
    fn get_filter(&self) -> &PathFilter {
        match self {
            #[cfg(feature = "std")]
            Self::File(p) => &p.filter,
            Self::Memory(p) => &p.filter,
        }
    }

    #[cfg(feature = "std")]
    fn get_remapper(&self) -> &KeyRemapper {
        match self {
            Self::File(p) => &p.remapper,
            Self::Memory(p) => &p.remapper,
        }
    }

    fn get_metadata(&self) -> &HashMap<String, String> {
        match self {
            #[cfg(feature = "std")]
            Self::File(p) => &p.metadata,
            Self::Memory(p) => &p.metadata,
        }
    }

    fn get_validate(&self) -> bool {
        match self {
            #[cfg(feature = "std")]
            Self::File(p) => p.validate,
            Self::Memory(p) => p.validate,
        }
    }

    fn get_allow_partial(&self) -> bool {
        match self {
            #[cfg(feature = "std")]
            Self::File(p) => p.allow_partial,
            Self::Memory(p) => p.allow_partial,
        }
    }
}

/// Apply filter to tensor snapshots.
fn apply_filter(mut snapshots: Vec<TensorSnapshot>, filter: &PathFilter) -> Vec<TensorSnapshot> {
    if filter.is_empty() {
        return snapshots;
    }

    snapshots.retain(|snapshot| {
        let path = snapshot.full_path();
        filter.matches(&path)
    });

    snapshots
}

/// Apply remapping to tensor snapshots.
#[cfg(feature = "std")]
fn apply_remapping(snapshots: Vec<TensorSnapshot>, remapper: &KeyRemapper) -> Vec<TensorSnapshot> {
    if remapper.is_empty() {
        return snapshots;
    }

    let (remapped, _) = remapper.remap(snapshots);
    remapped
}

/// Convert TensorSnapshots to safetensors format lazily.
fn snapshots_to_safetensors(
    snapshots: Vec<TensorSnapshot>,
) -> Result<Vec<(String, TensorSnapshotAdapter)>, SafetensorsStoreError> {
    let mut tensors = Vec::new();

    for snapshot in snapshots {
        let name = snapshot.full_path();
        // No need to materialize data - TensorSnapshot now has dtype and shape cached!
        tensors.push((name, TensorSnapshotAdapter(snapshot)));
    }

    Ok(tensors)
}

/// Convert safetensors to TensorSnapshots with lazy loading.
fn safetensors_to_snapshots_lazy(
    data_arc: Arc<Vec<u8>>,
) -> Result<Vec<TensorSnapshot>, SafetensorsStoreError> {
    // Parse to get metadata
    let tensors = safetensors::SafeTensors::deserialize(&data_arc)?;
    let mut snapshots = Vec::new();

    for (name, tensor_snapshot) in tensors.tensors() {
        // Extract metadata without materializing data
        let dtype = safetensor_dtype_to_burn(tensor_snapshot.dtype())?;
        let shape = tensor_snapshot.shape().to_vec();
        let path_parts: Vec<String> = name.split('.').map(|s| s.to_string()).collect();

        // Create a lazy closure that will deserialize only this tensor when needed
        #[cfg(target_has_atomic = "ptr")]
        let data_clone = Arc::clone(&data_arc);
        #[cfg(not(target_has_atomic = "ptr"))]
        let data_clone = data_arc.clone();
        let name_clone = name.to_string();
        let data_fn = alloc::rc::Rc::new(move || {
            // Re-deserialize when needed (this is cheap, just parsing header)
            let tensors = safetensors::SafeTensors::deserialize(&data_clone).map_err(|e| {
                crate::TensorSnapshotError::IoError(format!(
                    "Failed to re-deserialize safetensors: {}",
                    e
                ))
            })?;

            // Find our specific tensor
            let tensor = tensors.tensor(&name_clone).map_err(|e| {
                crate::TensorSnapshotError::DataError(format!(
                    "Tensor '{}' not found: {}",
                    name_clone, e
                ))
            })?;

            // Now materialize just this tensor's data
            let bytes = burn_tensor::Bytes::from_bytes_vec(tensor.data().to_vec());
            Ok(TensorData {
                bytes,
                shape: tensor.shape().to_vec(),
                dtype: safetensor_dtype_to_burn(tensor.dtype())
                    .map_err(|_| crate::TensorSnapshotError::DataError("Invalid dtype".into()))?,
            })
        });

        let snapshot = TensorSnapshot::from_closure(
            data_fn,
            dtype,
            shape,
            path_parts,
            vec![], // Empty container_stack - will be filled during module traversal
            ParamId::new(),
        );
        snapshots.push(snapshot);
    }

    Ok(snapshots)
}

/// Convert safetensors to TensorSnapshots with true on-demand loading from file.
/// This reads only the header initially, then loads tensor data on demand.
#[cfg(feature = "std")]
fn safetensors_to_snapshots_lazy_file(
    path: &std::path::Path,
) -> Result<Vec<TensorSnapshot>, SafetensorsStoreError> {
    // Always use memory mapping for the most efficient access
    use memmap2::MmapOptions;

    // Memory map the file for efficient access
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let mmap_arc = Arc::new(mmap);

    // Parse just to get metadata (safetensors won't copy data with mmap)
    let tensors = safetensors::SafeTensors::deserialize(&mmap_arc)?;
    let mut snapshots = Vec::new();

    for (name, tensor_snapshot) in tensors.tensors() {
        let dtype = safetensor_dtype_to_burn(tensor_snapshot.dtype())?;
        let shape = tensor_snapshot.shape().to_vec();
        let path_parts: Vec<String> = name.split('.').map(|s| s.to_string()).collect();

        // Create a lazy closure that accesses the mmap'd data
        let mmap_clone = Arc::clone(&mmap_arc);
        let name_clone = name.to_string();

        let data_fn = alloc::rc::Rc::new(move || {
            // Re-parse to get the tensor snapshot (this is cheap with mmap)
            let tensors = safetensors::SafeTensors::deserialize(&mmap_clone).map_err(|e| {
                crate::TensorSnapshotError::IoError(format!("Failed to deserialize: {}", e))
            })?;
            let tensor = tensors.tensor(&name_clone).map_err(|e| {
                crate::TensorSnapshotError::DataError(format!(
                    "Tensor '{}' not found: {}",
                    name_clone, e
                ))
            })?;

            // Only now do we actually copy the tensor data
            Ok(TensorData {
                bytes: burn_tensor::Bytes::from_bytes_vec(tensor.data().to_vec()),
                shape: tensor.shape().to_vec(),
                dtype: safetensor_dtype_to_burn(tensor.dtype())
                    .map_err(|_| crate::TensorSnapshotError::DataError("Invalid dtype".into()))?,
            })
        });

        let snapshot = TensorSnapshot::from_closure(
            data_fn,
            dtype,
            shape,
            path_parts,
            vec![], // Empty container_stack - will be filled during module traversal
            ParamId::new(),
        );
        snapshots.push(snapshot);
    }

    Ok(snapshots)
}

/// Helper to convert safetensors Dtype to burn DType.
fn safetensor_dtype_to_burn(dtype: safetensors::Dtype) -> Result<DType, SafetensorsStoreError> {
    use safetensors::Dtype;

    match dtype {
        Dtype::F64 => Ok(DType::F64),
        Dtype::F32 => Ok(DType::F32),
        Dtype::F16 => Ok(DType::F16),
        Dtype::BF16 => Ok(DType::BF16),
        Dtype::I64 => Ok(DType::I64),
        Dtype::I32 => Ok(DType::I32),
        Dtype::I16 => Ok(DType::I16),
        Dtype::I8 => Ok(DType::I8),
        Dtype::U64 => Ok(DType::U64),
        Dtype::U32 => Ok(DType::U32),
        Dtype::U8 => Ok(DType::U8),
        Dtype::BOOL => Ok(DType::Bool),
        _ => Err(SafetensorsStoreError::Other(format!(
            "Unsupported dtype: {:?}",
            dtype
        ))),
    }
}

/// Helper to convert DType to safetensors Dtype.
fn dtype_to_safetensors(dtype: DType) -> Result<safetensors::Dtype, SafetensorsStoreError> {
    use safetensors::Dtype;

    match dtype {
        DType::F64 => Ok(Dtype::F64),
        DType::F32 | DType::Flex32 => Ok(Dtype::F32), // Flex32 is stored as F32
        DType::F16 => Ok(Dtype::F16),
        DType::BF16 => Ok(Dtype::BF16),
        DType::I64 => Ok(Dtype::I64),
        DType::I32 => Ok(Dtype::I32),
        DType::I16 => Ok(Dtype::I16),
        DType::I8 => Ok(Dtype::I8),
        DType::U64 => Ok(Dtype::U64),
        DType::U32 => Ok(Dtype::U32),
        DType::U16 => Err(SafetensorsStoreError::Other(
            "U16 dtype not yet supported in safetensors".to_string(),
        )),
        DType::U8 => Ok(Dtype::U8),
        DType::Bool => Ok(Dtype::BOOL),
        DType::QFloat(_) => Err(SafetensorsStoreError::Other(
            "Quantized tensors not yet supported in safetensors".to_string(),
        )),
    }
}
