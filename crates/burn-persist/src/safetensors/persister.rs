//! SafeTensors persister implementation using the official safetensors crate.

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use burn_core::module::ParamId;
use burn_core::persist::{
    ApplyResult, KeyRemapper, ModulePersist, ModulePersister, PathFilter, TensorView,
};
use burn_core::tensor::backend::Backend;
use burn_tensor::{DType, TensorData};
use core::fmt;
use core::ops::Deref;
use hashbrown::HashMap;

/// Errors that can occur during SafeTensors operations.
#[derive(Debug)]
pub enum SafetensorsError {
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

impl fmt::Display for SafetensorsError {
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

impl core::error::Error for SafetensorsError {}

impl From<safetensors::SafeTensorError> for SafetensorsError {
    fn from(e: safetensors::SafeTensorError) -> Self {
        SafetensorsError::Safetensors(e)
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for SafetensorsError {
    fn from(e: std::io::Error) -> Self {
        SafetensorsError::Io(e)
    }
}

/// SafeTensors persister supporting both file and memory storage.
pub enum SafetensorsPersister {
    /// File-based storage.
    #[cfg(feature = "std")]
    File(FilePersister),

    /// Memory-based storage.
    Memory(MemoryPersister),
}

impl Default for SafetensorsPersister {
    /// Create a default memory-based persister.
    fn default() -> Self {
        Self::from_bytes(None)
    }
}

impl SafetensorsPersister {
    /// Create a persister for loading from or saving to a file.
    ///
    /// # Example
    /// ```ignore
    /// let persister = SafetensorsPersister::from_file("model.safetensors");
    /// model.save_with(&persister)?;
    /// ```
    #[cfg(feature = "std")]
    pub fn from_file(path: impl Into<std::path::PathBuf>) -> Self {
        Self::File(FilePersister {
            path: path.into(),
            filter: PathFilter::new(),
            remapper: KeyRemapper::new(),
            metadata: HashMap::new(),
            validate: true,
            allow_partial: false,
        })
    }

    /// Create a persister for working with bytes in memory.
    ///
    /// # Example
    /// ```ignore
    /// // Create empty persister for saving
    /// let persister = SafetensorsPersister::from_bytes(None);
    /// model.collect_to(&mut persister)?;
    /// let bytes = persister.get_bytes()?;
    ///
    /// // Create persister with bytes for loading
    /// let persister = SafetensorsPersister::from_bytes(Some(bytes));
    /// let model = Model::apply_from(&mut persister)?;
    /// ```
    pub fn from_bytes(bytes: Option<Vec<u8>>) -> Self {
        Self::Memory(MemoryPersister {
            data: bytes.map(alloc::sync::Arc::new),
            filter: PathFilter::new(),
            remapper: KeyRemapper::new(),
            metadata: HashMap::new(),
            validate: true,
            allow_partial: false,
        })
    }

    /// Convenience constructor - same as `from_file` (for backwards compatibility).
    #[cfg(feature = "std")]
    #[deprecated(since = "0.19.0", note = "Use `from_file` instead")]
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self::from_file(path)
    }

    /// Convenience constructor - same as `from_file`.
    #[cfg(feature = "std")]
    #[deprecated(since = "0.19.0", note = "Use `from_file` instead")]
    pub fn file(path: impl Into<std::path::PathBuf>) -> Self {
        Self::from_file(path)
    }

    /// Convenience constructor - same as `from_bytes(None)`.
    #[deprecated(since = "0.19.0", note = "Use `from_bytes(None)` instead")]
    pub fn memory() -> Self {
        Self::from_bytes(None)
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

    /// Remap tensor names during load/save.
    #[cfg(target_has_atomic = "ptr")]
    pub fn remap(mut self, remapper: KeyRemapper) -> Self {
        match &mut self {
            #[cfg(feature = "std")]
            Self::File(p) => p.remapper = remapper,
            Self::Memory(p) => p.remapper = remapper,
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

    /// List all tensor names in the file without loading any data.
    ///
    /// This only reads the header which contains tensor metadata.
    ///
    /// # Example
    /// ```ignore
    /// let persister = SafetensorsPersister::from_file("model.safetensors");
    /// let tensor_names = persister.list_tensors()?;
    /// println!("Model contains {} tensors", tensor_names.len());
    /// ```
    pub fn list_tensors(&self) -> Result<Vec<String>, SafetensorsError> {
        match self {
            #[cfg(feature = "std")]
            Self::File(p) => {
                #[cfg(feature = "memmap2")]
                {
                    use memmap2::MmapOptions;
                    let file = std::fs::File::open(&p.path)?;
                    let mmap = unsafe { MmapOptions::new().map(&file)? };
                    let tensors = safetensors::SafeTensors::deserialize(&mmap)?;
                    Ok(tensors.names().into_iter().map(|s| s.to_string()).collect())
                }
                #[cfg(not(feature = "memmap2"))]
                {
                    let buffer = std::fs::read(&p.path)?;
                    let tensors = safetensors::SafeTensors::deserialize(&buffer)?;
                    Ok(tensors.names().into_iter().map(|s| s.to_string()).collect())
                }
            }
            Self::Memory(p) => {
                let data = p
                    .data()
                    .ok_or_else(|| SafetensorsError::Other("No data available".to_string()))?;
                let tensors = safetensors::SafeTensors::deserialize(&data)?;
                Ok(tensors.names().into_iter().map(|s| s.to_string()).collect())
            }
        }
    }

    /// Get tensor metadata (shape, dtype) without loading the data.
    ///
    /// # Example
    /// ```ignore
    /// let persister = SafetensorsPersister::from_file("model.safetensors");
    /// if let Some((shape, dtype)) = persister.tensor_info("encoder.weight")? {
    ///     println!("Tensor shape: {:?}, dtype: {:?}", shape, dtype);
    /// }
    /// ```
    pub fn tensor_info(
        &self,
        name: &str,
    ) -> Result<Option<(Vec<usize>, burn_tensor::DType)>, SafetensorsError> {
        match self {
            #[cfg(feature = "std")]
            Self::File(p) => {
                #[cfg(feature = "memmap2")]
                {
                    use memmap2::MmapOptions;
                    let file = std::fs::File::open(&p.path)?;
                    let mmap = unsafe { MmapOptions::new().map(&file)? };
                    let tensors = safetensors::SafeTensors::deserialize(&mmap)?;
                    if let Ok(tensor) = tensors.tensor(name) {
                        let dtype = safetensor_dtype_to_burn(tensor.dtype())?;
                        Ok(Some((tensor.shape().to_vec(), dtype)))
                    } else {
                        Ok(None)
                    }
                }
                #[cfg(not(feature = "memmap2"))]
                {
                    let buffer = std::fs::read(&p.path)?;
                    let tensors = safetensors::SafeTensors::deserialize(&buffer)?;
                    if let Ok(tensor) = tensors.tensor(name) {
                        let dtype = safetensor_dtype_to_burn(tensor.dtype())?;
                        Ok(Some((tensor.shape().to_vec(), dtype)))
                    } else {
                        Ok(None)
                    }
                }
            }
            Self::Memory(p) => {
                let data = p
                    .data()
                    .ok_or_else(|| SafetensorsError::Other("No data available".to_string()))?;
                let tensors = safetensors::SafeTensors::deserialize(&data)?;
                if let Ok(tensor) = tensors.tensor(name) {
                    let dtype = safetensor_dtype_to_burn(tensor.dtype())?;
                    Ok(Some((tensor.shape().to_vec(), dtype)))
                } else {
                    Ok(None)
                }
            }
        }
    }

    /// Load only specific tensors by name.
    ///
    /// This is useful in distributed settings where different nodes need different tensors.
    /// Only the requested tensors are loaded into memory.
    ///
    /// # Example
    /// ```ignore
    /// let persister = SafetensorsPersister::from_file("model.safetensors");
    /// let tensors = persister.load_tensors(&["encoder.weight", "encoder.bias"])?;
    /// ```
    pub fn load_tensors(
        &self,
        names: &[&str],
    ) -> Result<Vec<(String, burn_tensor::TensorData)>, SafetensorsError> {
        match self {
            #[cfg(feature = "std")]
            Self::File(p) => {
                #[cfg(feature = "memmap2")]
                {
                    // Use memory mapping for efficient access - safetensors' recommended approach
                    use memmap2::MmapOptions;
                    let file = std::fs::File::open(&p.path)?;
                    let mmap = unsafe { MmapOptions::new().map(&file)? };
                    let tensors = safetensors::SafeTensors::deserialize(&mmap)?;

                    let mut result = Vec::new();
                    for name in names {
                        if let Ok(tensor) = tensors.tensor(name) {
                            let dtype = safetensor_dtype_to_burn(tensor.dtype())?;
                            let data = burn_tensor::TensorData {
                                bytes: burn_tensor::Bytes::from_bytes_vec(tensor.data().to_vec()),
                                shape: tensor.shape().to_vec(),
                                dtype,
                            };
                            result.push((name.to_string(), data));
                        }
                    }
                    Ok(result)
                }
                #[cfg(not(feature = "memmap2"))]
                {
                    // Fallback: read entire file if memmap2 is not available
                    let buffer = std::fs::read(&p.path)?;
                    let tensors = safetensors::SafeTensors::deserialize(&buffer)?;

                    let mut result = Vec::new();
                    for name in names {
                        if let Ok(tensor) = tensors.tensor(name) {
                            let dtype = safetensor_dtype_to_burn(tensor.dtype())?;
                            let data = burn_tensor::TensorData {
                                bytes: burn_tensor::Bytes::from_bytes_vec(tensor.data().to_vec()),
                                shape: tensor.shape().to_vec(),
                                dtype,
                            };
                            result.push((name.to_string(), data));
                        }
                    }
                    Ok(result)
                }
            }
            Self::Memory(p) => {
                let data = p
                    .data()
                    .ok_or_else(|| SafetensorsError::Other("No data available".to_string()))?;
                let tensors = safetensors::SafeTensors::deserialize(&data)?;

                let mut result = Vec::new();
                for name in names {
                    if let Ok(tensor) = tensors.tensor(name) {
                        let dtype = safetensor_dtype_to_burn(tensor.dtype())?;
                        let data = burn_tensor::TensorData {
                            bytes: burn_tensor::Bytes::from_bytes_vec(tensor.data().to_vec()),
                            shape: tensor.shape().to_vec(),
                            dtype,
                        };
                        result.push((name.to_string(), data));
                    }
                }
                Ok(result)
            }
        }
    }

    /// Get saved bytes from memory-based persister.
    ///
    /// # Example
    /// ```ignore
    /// let mut persister = SafetensorsPersister::from_bytes(None);
    /// model.collect_to(&mut persister)?;
    /// let bytes = persister.get_bytes()?;
    /// ```
    pub fn get_bytes(&self) -> Result<Vec<u8>, SafetensorsError> {
        match self {
            #[cfg(feature = "std")]
            Self::File(_) => Err(SafetensorsError::Other(
                "Cannot get bytes from file-based persister".to_string(),
            )),
            Self::Memory(p) => p
                .data()
                .map(|arc| arc.as_ref().clone())
                .ok_or_else(|| SafetensorsError::Other("No data available".to_string())),
        }
    }
}

/// File-based persister.
#[cfg(feature = "std")]
pub struct FilePersister {
    path: std::path::PathBuf,
    filter: PathFilter,
    remapper: KeyRemapper,
    metadata: HashMap<String, String>,
    validate: bool,
    allow_partial: bool,
}

/// Memory-based persister.
pub struct MemoryPersister {
    data: Option<alloc::sync::Arc<Vec<u8>>>,
    filter: PathFilter,
    remapper: KeyRemapper,
    metadata: HashMap<String, String>,
    validate: bool,
    allow_partial: bool,
}

impl Default for MemoryPersister {
    fn default() -> Self {
        Self {
            data: None,
            filter: PathFilter::new(),
            remapper: KeyRemapper::new(),
            metadata: HashMap::new(),
            validate: true,
            allow_partial: false,
        }
    }
}

impl MemoryPersister {
    /// Create a new memory-based persister.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the stored data.
    pub fn data(&self) -> Option<alloc::sync::Arc<Vec<u8>>> {
        self.data.clone()
    }

    /// Set data for loading.
    pub fn set_data(&mut self, data: Vec<u8>) {
        self.data = Some(alloc::sync::Arc::new(data));
    }
}

// Adapter to use TensorView directly with safetensors
struct TensorViewAdapter(TensorView);

impl safetensors::View for TensorViewAdapter {
    fn dtype(&self) -> safetensors::Dtype {
        // Convert from burn dtype to safetensors dtype
        dtype_to_safetensors(self.0.dtype).unwrap_or(safetensors::Dtype::F32)
    }

    fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    fn data(&self) -> alloc::borrow::Cow<'_, [u8]> {
        // Only materialize data when actually needed for serialization
        let data = self.0.to_data();
        alloc::borrow::Cow::Owned(data.bytes.deref().to_vec())
    }

    fn data_len(&self) -> usize {
        // Use the efficient data_len method from TensorView
        self.0.data_len()
    }
}

impl ModulePersister for SafetensorsPersister {
    type Error = SafetensorsError;

    fn collect_from<B: Backend, M: ModulePersist<B>>(
        &mut self,
        module: &M,
    ) -> Result<(), Self::Error> {
        // Collect tensor views from module
        let mut views = module.collect();

        // Apply filtering
        views = apply_filter(views, self.get_filter());

        // Apply remapping
        #[cfg(target_has_atomic = "ptr")]
        {
            views = apply_remapping(views, self.get_remapper());
        }

        // Convert to safetensors format
        let tensors = views_to_safetensors(views)?;

        // Add metadata
        let mut metadata = self.get_metadata().clone();
        metadata.insert("framework".to_string(), "burn".to_string());

        // Serialize using safetensors crate
        // safetensors always uses hashbrown::HashMap for metadata
        // FIXME copies data!!!
        let data = safetensors::serialize(tensors, Some(metadata))?;

        // Write to storage
        match self {
            #[cfg(feature = "std")]
            Self::File(p) => {
                std::fs::write(&p.path, data)?;
                Ok(())
            }
            Self::Memory(p) => {
                p.data = Some(alloc::sync::Arc::new(data));
                Ok(())
            }
        }
    }

    fn apply_to<B: Backend, M: ModulePersist<B>>(
        &mut self,
        module: &mut M,
    ) -> Result<ApplyResult, Self::Error> {
        // Convert to tensor views with lazy loading
        let views = match self {
            #[cfg(feature = "std")]
            Self::File(p) => {
                // Use safetensors' built-in lazy loading mechanisms
                safetensors_to_views_lazy_file(&p.path)?
            }
            Self::Memory(p) => {
                let data_arc = p
                    .data
                    .clone()
                    .ok_or_else(|| SafetensorsError::Other("No data loaded".to_string()))?;
                safetensors_to_views_lazy(data_arc)?
            }
        };

        // Apply to module
        let result = module.apply(views);

        // Validate if needed
        if self.get_validate() && !result.errors.is_empty() {
            return Err(SafetensorsError::ValidationFailed(format!(
                "Import errors: {:?}",
                result.errors
            )));
        }

        if !self.get_allow_partial() && !result.missing.is_empty() {
            return Err(SafetensorsError::TensorNotFound(format!(
                "Missing tensors: {:?}",
                result.missing
            )));
        }

        Ok(result)
    }
}

impl SafetensorsPersister {
    fn get_filter(&self) -> &PathFilter {
        match self {
            #[cfg(feature = "std")]
            Self::File(p) => &p.filter,
            Self::Memory(p) => &p.filter,
        }
    }

    fn get_remapper(&self) -> &KeyRemapper {
        match self {
            #[cfg(feature = "std")]
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

/// Apply filter to tensor views.
fn apply_filter(mut views: Vec<TensorView>, filter: &PathFilter) -> Vec<TensorView> {
    if filter.is_empty() {
        return views;
    }

    views.retain(|view| {
        let path = view.full_path();
        filter.matches(&path)
    });

    views
}

/// Apply remapping to tensor views.
#[cfg(target_has_atomic = "ptr")]
fn apply_remapping(views: Vec<TensorView>, remapper: &KeyRemapper) -> Vec<TensorView> {
    if remapper.is_empty() {
        return views;
    }

    let (remapped, _) = remapper.remap(views);
    remapped
}

/// Convert TensorViews to safetensors format lazily.
fn views_to_safetensors(
    views: Vec<TensorView>,
) -> Result<Vec<(String, TensorViewAdapter)>, SafetensorsError> {
    let mut tensors = Vec::new();

    for view in views {
        let name = view.full_path();
        // No need to materialize data - TensorView now has dtype and shape cached!
        tensors.push((name, TensorViewAdapter(view)));
    }

    Ok(tensors)
}

/// Convert safetensors to TensorViews with lazy loading.
fn safetensors_to_views_lazy(
    data_arc: alloc::sync::Arc<Vec<u8>>,
) -> Result<Vec<TensorView>, SafetensorsError> {
    // Parse to get metadata
    let tensors = safetensors::SafeTensors::deserialize(&data_arc)?;
    let mut views = Vec::new();

    for (name, tensor_view) in tensors.tensors() {
        // Extract metadata without materializing data
        let dtype = safetensor_dtype_to_burn(tensor_view.dtype())?;
        let shape = tensor_view.shape().to_vec();
        let path_parts: Vec<String> = name.split('.').map(|s| s.to_string()).collect();

        // Create a lazy closure that will deserialize only this tensor when needed
        let data_clone = alloc::sync::Arc::clone(&data_arc);
        let name_clone = name.to_string();
        let data_fn = Box::new(move || {
            // Re-deserialize when needed (this is cheap, just parsing header)
            let tensors = safetensors::SafeTensors::deserialize(&data_clone)
                .expect("Failed to re-deserialize safetensors");

            // Find our specific tensor
            let tensor = tensors.tensor(&name_clone).expect("Tensor should exist");

            // Now materialize just this tensor's data
            let bytes = burn_tensor::Bytes::from_bytes_vec(tensor.data().to_vec());
            TensorData {
                bytes,
                shape: tensor.shape().to_vec(),
                dtype: safetensor_dtype_to_burn(tensor.dtype()).expect("Valid dtype"),
            }
        });

        let view = TensorView::from_closure(
            data_fn,
            dtype,
            shape,
            path_parts,
            vec!["SafeTensor".to_string()],
            ParamId::new(),
        );
        views.push(view);
    }

    Ok(views)
}

/// Convert safetensors to TensorViews with true on-demand loading from file.
/// This reads only the header initially, then loads tensor data on demand.
#[cfg(feature = "std")]
fn safetensors_to_views_lazy_file(
    path: &std::path::Path,
) -> Result<Vec<TensorView>, SafetensorsError> {
    use alloc::sync::Arc;

    // Use memory mapping if available for the most efficient access
    #[cfg(feature = "memmap2")]
    {
        use memmap2::MmapOptions;

        // Memory map the file for efficient access
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let mmap_arc = Arc::new(mmap);

        // Parse just to get metadata (safetensors won't copy data with mmap)
        let tensors = safetensors::SafeTensors::deserialize(&mmap_arc)?;
        let mut views = Vec::new();

        for (name, tensor_view) in tensors.tensors() {
            let dtype = safetensor_dtype_to_burn(tensor_view.dtype())?;
            let shape = tensor_view.shape().to_vec();
            let path_parts: Vec<String> = name.split('.').map(|s| s.to_string()).collect();

            // Create a lazy closure that accesses the mmap'd data
            let mmap_clone = Arc::clone(&mmap_arc);
            let name_clone = name.to_string();

            let data_fn = Box::new(move || {
                // Re-parse to get the tensor view (this is cheap with mmap)
                let tensors = safetensors::SafeTensors::deserialize(&mmap_clone)
                    .expect("Failed to deserialize");
                let tensor = tensors.tensor(&name_clone).expect("Tensor should exist");

                // Only now do we actually copy the tensor data
                TensorData {
                    bytes: burn_tensor::Bytes::from_bytes_vec(tensor.data().to_vec()),
                    shape: tensor.shape().to_vec(),
                    dtype: safetensor_dtype_to_burn(tensor.dtype()).expect("Valid dtype"),
                }
            });

            let view = TensorView::from_closure(
                data_fn,
                dtype,
                shape,
                path_parts,
                vec!["SafeTensor".to_string()],
                ParamId::new(),
            );
            views.push(view);
        }

        Ok(views)
    }
    #[cfg(not(feature = "memmap2"))]
    {
        // Without mmap, we still want on-demand loading
        // Read the entire file but create closures that defer tensor data copying
        let buffer = std::fs::read(path)?;
        let buffer_arc = Arc::new(buffer);

        // Parse with safetensors (this doesn't copy tensor data, just creates views)
        let tensors = safetensors::SafeTensors::deserialize(&buffer_arc)?;
        let mut views = Vec::new();

        for (name, tensor_view) in tensors.tensors() {
            let dtype = safetensor_dtype_to_burn(tensor_view.dtype())?;
            let shape = tensor_view.shape().to_vec();
            let path_parts: Vec<String> = name.split('.').map(|s| s.to_string()).collect();

            // Create a lazy closure that copies data only when accessed
            let buffer_clone = Arc::clone(&buffer_arc);
            let name_clone = name.to_string();

            let data_fn = Box::new(move || {
                // Re-parse to get the tensor view
                let tensors = safetensors::SafeTensors::deserialize(&buffer_clone)
                    .expect("Failed to deserialize");
                let tensor = tensors.tensor(&name_clone).expect("Tensor should exist");

                // Copy tensor data only when this closure is called
                TensorData {
                    bytes: burn_tensor::Bytes::from_bytes_vec(tensor.data().to_vec()),
                    shape: tensor.shape().to_vec(),
                    dtype: safetensor_dtype_to_burn(tensor.dtype()).expect("Valid dtype"),
                }
            });

            let view = TensorView::from_closure(
                data_fn,
                dtype,
                shape,
                path_parts,
                vec!["SafeTensor".to_string()],
                ParamId::new(),
            );
            views.push(view);
        }

        Ok(views)
    }
}

/// Helper to convert safetensors Dtype to burn DType.
fn safetensor_dtype_to_burn(dtype: safetensors::Dtype) -> Result<DType, SafetensorsError> {
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
        _ => Err(SafetensorsError::Other(format!(
            "Unsupported dtype: {:?}",
            dtype
        ))),
    }
}

/// Helper to convert DType to safetensors Dtype.
fn dtype_to_safetensors(dtype: DType) -> Result<safetensors::Dtype, SafetensorsError> {
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
        DType::U16 => Err(SafetensorsError::Other(
            "U16 dtype not yet supported in safetensors".to_string(),
        )),
        DType::U8 => Ok(Dtype::U8),
        DType::Bool => Ok(Dtype::BOOL),
        DType::QFloat(_) => Err(SafetensorsError::Other(
            "Quantized tensors not yet supported in safetensors".to_string(),
        )),
    }
}
