use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::convert::Into;
use hashbrown::HashMap;

use crate::persist::{
    appliers::ApplyResult,
    KeyRemapper, ModulePersist, ModulePersister, TensorView,
};
use crate::tensor::backend::Backend;

use super::{
    format::SafetensorsError,
    reader::{MemmapSafetensorsReader, SafetensorsReader},
    writer::SafetensorsWriter,
};

/// Configuration builder for SafetensorsPersister
///
/// This struct provides a fluent interface for configuring safetensors persistence
/// operations with filtering, remapping, and other options.
///
/// # Examples
///
/// ```ignore
/// // Save with filtering
/// let mut persister = SafetensorsPersisterConfig::new()
///     .with_filter(&[r"^encoder\..*", r"^decoder\..*"])
///     .with_metadata("framework", "burn")
///     .build("model.safetensors")?;
/// module.collect_to(&mut persister)?;
///
/// // Load with remapping
/// let mut persister = SafetensorsPersisterConfig::new()
///     .with_remapping(&[("old_prefix", "new_prefix")])
///     .with_partial_loading(true)
///     .build("model.safetensors")?;
/// module.apply_from(&mut persister)?;
/// ```
#[derive(Debug, Clone)]
pub struct SafetensorsPersisterConfig {
    config: PersisterConfig,
    metadata: HashMap<String, String>,
}

impl SafetensorsPersisterConfig {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            config: PersisterConfig::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add regex patterns for filtering tensors (OR union)
    #[cfg(target_has_atomic = "ptr")]
    pub fn with_filter<S: AsRef<str>>(mut self, patterns: &[S]) -> Self {
        let patterns = patterns.iter().map(|s| s.as_ref().to_string()).collect();
        self.config.filter = TensorFilter::Patterns(patterns);
        self
    }

    /// Add specific tensor names to include
    pub fn with_tensor_names<S: Into<String>>(mut self, names: Vec<S>) -> Self {
        let names = names.into_iter().map(|s| s.into()).collect();
        self.config.filter = TensorFilter::Names(names);
        self
    }

    /// Add a predicate function for filtering
    pub fn with_predicate(mut self, predicate: fn(&str) -> bool) -> Self {
        self.config.filter = TensorFilter::Predicate(predicate);
        self
    }

    /// Add key remapping patterns
    #[cfg(target_has_atomic = "ptr")]
    pub fn with_remapping<S1: AsRef<str>, S2: AsRef<str>>(
        mut self,
        patterns: &[(S1, S2)],
    ) -> Self {
        let patterns = patterns
            .iter()
            .map(|(p, r)| (p.as_ref().to_string(), r.as_ref().to_string()))
            .collect();
        self.config.remapping = KeyRemapper::from_patterns(patterns);
        self
    }

    /// Set validation behavior
    pub fn with_validation(mut self, validate: bool) -> Self {
        self.config.validate = validate;
        self
    }

    /// Set partial loading behavior
    pub fn with_partial_loading(mut self, allow_partial: bool) -> Self {
        self.config.allow_partial = allow_partial;
        self
    }

    /// Add metadata to be saved with the model
    pub fn with_metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Build a persister for file operations
    pub fn build<P: AsRef<std::path::Path>>(
        self,
        path: P,
    ) -> Result<SafetensorsPersister, SafetensorsError> {
        SafetensorsPersister::new(path, self.config, self.metadata)
    }

    /// Build a persister for in-memory operations
    pub fn build_in_memory(self) -> SafetensorsMemoryPersister {
        SafetensorsMemoryPersister::new(self.config, self.metadata)
    }
}

impl Default for SafetensorsPersisterConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl PersisterBuilder<SafetensorsPersister> for SafetensorsPersisterConfig {
    type Error = SafetensorsError;

    fn build(self) -> Result<SafetensorsPersister, Self::Error> {
        Err(SafetensorsError::Unsupported(
            "Use build(path) for file-based persister".to_string(),
        ))
    }
}

/// File-based safetensors persister
///
/// This persister handles loading and saving safetensors files with the configured
/// filtering, remapping, and metadata options.
pub struct SafetensorsPersister {
    path: std::path::PathBuf,
    config: PersisterConfig,
    metadata: HashMap<String, String>,
}

impl SafetensorsPersister {
    /// Create a new file-based persister
    pub fn new<P: AsRef<std::path::Path>>(
        path: P,
        config: PersisterConfig,
        metadata: HashMap<String, String>,
    ) -> Result<Self, SafetensorsError> {
        Ok(Self {
            path: path.as_ref().to_path_buf(),
            config,
            metadata,
        })
    }

    /// Apply filtering to tensor views
    fn apply_filter(&self, views: HashMap<String, TensorView>) -> HashMap<String, TensorView> {
        match &self.config.filter {
            TensorFilter::All => views,
            
            #[cfg(target_has_atomic = "ptr")]
            TensorFilter::Patterns(patterns) => {
                if patterns.is_empty() {
                    return views;
                }

                views
                    .into_iter()
                    .filter(|(path, _)| {
                        patterns.iter().any(|pattern| {
                            if let Ok(regex) = regex::Regex::new(pattern) {
                                regex.is_match(path)
                            } else {
                                false
                            }
                        })
                    })
                    .collect()
            }
            
            TensorFilter::Predicate(predicate) => {
                views
                    .into_iter()
                    .filter(|(path, _)| predicate(path))
                    .collect()
            }
            
            TensorFilter::Names(names) => {
                views
                    .into_iter()
                    .filter(|(path, _)| names.contains(path))
                    .collect()
            }
        }
    }

    /// Apply key remapping to tensor views
    #[cfg(target_has_atomic = "ptr")]
    fn apply_remapping(
        &self,
        mut views: HashMap<String, TensorView>,
    ) -> Result<HashMap<String, TensorView>, SafetensorsError> {
        if self.config.remapping.is_empty() {
            return Ok(views);
        }

        let mut remapped = HashMap::new();

        for (name, view) in views.drain() {
            let mut new_name = name.clone();

            for (pattern, replacement) in &self.config.remapping.patterns {
                if let Ok(regex) = regex::Regex::new(pattern) {
                    if regex.is_match(&new_name) {
                        new_name = regex.replace_all(&new_name, replacement).to_string();
                    }
                } else {
                    return Err(SafetensorsError::SerdeError(format!(
                        "Invalid regex pattern: {}",
                        pattern
                    )));
                }
            }

            remapped.insert(new_name, view);
        }

        Ok(remapped)
    }

    #[cfg(not(target_has_atomic = "ptr"))]
    fn apply_remapping(
        &self,
        views: HashMap<String, TensorView>,
    ) -> Result<HashMap<String, TensorView>, SafetensorsError> {
        Ok(views)
    }
}

impl ModulePersister for SafetensorsPersister {
    type Error = SafetensorsError;

    fn collect_from<B: Backend, M: ModulePersist<B>>(
        &mut self,
        module: &M,
    ) -> Result<(), Self::Error> {
        // Collect tensors from module
        let views = module.collect();

        // Apply filtering
        let filtered_views = self.apply_filter(views);

        // Apply remapping
        let remapped_views = self.apply_remapping(filtered_views)?;

        // Write to file
        use std::fs::File;
        use std::io::BufWriter;

        let file = File::create(&self.path)
            .map_err(|e| SafetensorsError::Io(format!("Failed to create file: {}", e)))?;

        let mut writer = SafetensorsWriter::new(BufWriter::new(file));

        // Add metadata
        for (key, value) in &self.metadata {
            writer.add_metadata(key.clone(), value.clone());
        }

        writer.write_views(remapped_views)?;
        writer.finish()?;

        Ok(())
    }

    fn apply_to<B: Backend, M: ModulePersist<B>>(
        &mut self,
        module: &mut M,
    ) -> Result<ApplyResult, Self::Error> {
        // Read from file
        #[cfg(feature = "std")]
        {
            let reader = MemmapSafetensorsReader::from_file(&self.path)?;
            let views = reader.read_all_views()?;

            // For loading, we don't apply remapping or filtering since the data
            // is already transformed. We just apply directly to the module.
            let result = module.apply(views);
            Ok(result)
        }

        #[cfg(not(feature = "std"))]
        {
            Err(SafetensorsError::Unsupported(
                "File operations require std feature".to_string(),
            ))
        }
    }
}

/// In-memory safetensors persister
///
/// This persister handles loading and saving safetensors data to/from memory buffers.
pub struct SafetensorsMemoryPersister {
    config: PersisterConfig,
    metadata: HashMap<String, String>,
    data: Option<Vec<u8>>,
}

impl SafetensorsMemoryPersister {
    /// Create a new memory-based persister
    pub fn new(config: PersisterConfig, metadata: HashMap<String, String>) -> Self {
        Self {
            config,
            metadata,
            data: None,
        }
    }

    /// Get the serialized data buffer
    pub fn data(&self) -> Option<&[u8]> {
        self.data.as_deref()
    }

    /// Set data from a buffer (for loading operations)
    pub fn set_data(&mut self, data: Vec<u8>) {
        self.data = Some(data);
    }

    /// Apply filtering to tensor views
    fn apply_filter(&self, views: HashMap<String, TensorView>) -> HashMap<String, TensorView> {
        match &self.config.filter {
            TensorFilter::All => views,
            
            #[cfg(target_has_atomic = "ptr")]
            TensorFilter::Patterns(patterns) => {
                if patterns.is_empty() {
                    return views;
                }

                views
                    .into_iter()
                    .filter(|(path, _)| {
                        patterns.iter().any(|pattern| {
                            if let Ok(regex) = regex::Regex::new(pattern) {
                                regex.is_match(path)
                            } else {
                                false
                            }
                        })
                    })
                    .collect()
            }
            
            TensorFilter::Predicate(predicate) => {
                views
                    .into_iter()
                    .filter(|(path, _)| predicate(path))
                    .collect()
            }
            
            TensorFilter::Names(names) => {
                views
                    .into_iter()
                    .filter(|(path, _)| names.contains(path))
                    .collect()
            }
        }
    }

    /// Apply key remapping to tensor views
    #[cfg(target_has_atomic = "ptr")]
    fn apply_remapping(
        &self,
        mut views: HashMap<String, TensorView>,
    ) -> Result<HashMap<String, TensorView>, SafetensorsError> {
        if self.config.remapping.is_empty() {
            return Ok(views);
        }

        let mut remapped = HashMap::new();

        for (name, view) in views.drain() {
            let mut new_name = name.clone();

            for (pattern, replacement) in &self.config.remapping.patterns {
                if let Ok(regex) = regex::Regex::new(pattern) {
                    if regex.is_match(&new_name) {
                        new_name = regex.replace_all(&new_name, replacement).to_string();
                    }
                } else {
                    return Err(SafetensorsError::SerdeError(format!(
                        "Invalid regex pattern: {}",
                        pattern
                    )));
                }
            }

            remapped.insert(new_name, view);
        }

        Ok(remapped)
    }

    #[cfg(not(target_has_atomic = "ptr"))]
    fn apply_remapping(
        &self,
        views: HashMap<String, TensorView>,
    ) -> Result<HashMap<String, TensorView>, SafetensorsError> {
        Ok(views)
    }
}

impl ModulePersister for SafetensorsMemoryPersister {
    type Error = SafetensorsError;

    fn collect_from<B: Backend, M: ModulePersist<B>>(
        &mut self,
        module: &M,
    ) -> Result<(), Self::Error> {
        // Collect tensors from module
        let views = module.collect();

        // Apply filtering
        let filtered_views = self.apply_filter(views);

        // Apply remapping
        let remapped_views = self.apply_remapping(filtered_views)?;

        // Write to memory buffer
        use std::io::Cursor;
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = SafetensorsWriter::new(&mut buffer);

        // Add metadata
        for (key, value) in &self.metadata {
            writer.add_metadata(key.clone(), value.clone());
        }

        writer.write_views(remapped_views)?;
        writer.finish()?;

        self.data = Some(buffer.into_inner());
        Ok(())
    }

    fn apply_to<B: Backend, M: ModulePersist<B>>(
        &mut self,
        module: &mut M,
    ) -> Result<ApplyResult, Self::Error> {
        let data = self.data.as_ref().ok_or_else(|| {
            SafetensorsError::InvalidFormat("No data available for loading".to_string())
        })?;

        // Read from memory buffer
        use std::io::Cursor;
        let mut cursor = Cursor::new(data);
        let mut reader = SafetensorsReader::new(&mut cursor)?;
        let views = reader.read_all_views()?;

        // For loading, we don't apply remapping or filtering since the data
        // is already transformed. We just apply directly to the module.
        let result = module.apply(views);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate as burn;
    use crate::module::{Module, Param};
    use crate::tensor::backend::Backend;
    use crate::TestBackend;
    use burn_tensor::Tensor;

    #[derive(Module, Debug)]
    struct TestModule<B: Backend> {
        encoder: TestSubModule<B>,
        decoder: TestSubModule<B>,
    }

    #[derive(Module, Debug)]
    struct TestSubModule<B: Backend> {
        weight: Param<Tensor<B, 2>>,
        bias: Param<Tensor<B, 1>>,
    }

    impl<B: Backend> TestModule<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                encoder: TestSubModule::new(device),
                decoder: TestSubModule::new(device),
            }
        }
    }

    impl<B: Backend> TestSubModule<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                weight: Param::from_data([[1.0, 2.0], [3.0, 4.0]], device),
                bias: Param::from_data([5.0, 6.0], device),
            }
        }
    }

    #[test]
    fn test_config_builder() {
        #[cfg(target_has_atomic = "ptr")]
        {
            let config = SafetensorsPersisterConfig::new()
                .with_filter(&[r"^encoder\..*"])
                .with_remapping(&[("old", "new")])
                .with_metadata("framework", "burn")
                .with_validation(false);

            assert!(!config.config.validate);
            assert_eq!(config.metadata.get("framework"), Some(&"burn".to_string()));
        }
    }

    #[test]
    fn test_memory_persister_round_trip() {
        let device = Default::default();
        let module1 = TestModule::<TestBackend>::new(&device);
        let mut module2 = TestModule::<TestBackend>::new(&device);

        // Create persister for saving
        let mut save_persister = SafetensorsPersisterConfig::new()
            .with_metadata("test", "value")
            .build_in_memory();

        // Save module1
        save_persister.collect_from(&module1).unwrap();

        // Create persister for loading
        let mut load_persister = SafetensorsPersisterConfig::new().build_in_memory();
        load_persister.set_data(save_persister.data().unwrap().to_vec());

        // Load into module2
        let result = load_persister.apply_to(&mut module2).unwrap();
        assert!(result.is_success());
        assert_eq!(result.applied.len(), 4); // encoder.weight, encoder.bias, decoder.weight, decoder.bias
    }

    #[test]
    fn test_filtering() {
        #[cfg(target_has_atomic = "ptr")]
        {
            let device = Default::default();
            let module = TestModule::<TestBackend>::new(&device);

            // Test with encoder filter
            let mut persister = SafetensorsPersisterConfig::new()
                .with_filter(&[r"^encoder\..*"])
                .build_in_memory();

            persister.collect_from(&module).unwrap();

            // Load into new module and check only encoder tensors were saved
            let mut target_module = TestModule::<TestBackend>::new(&device);
            let result = persister.apply_to(&mut target_module).unwrap();
            
            assert_eq!(result.applied.len(), 2); // Only encoder.weight and encoder.bias
            assert!(result.applied.contains(&"encoder.weight".to_string()));
            assert!(result.applied.contains(&"encoder.bias".to_string()));
        }
    }

    #[test]
    fn test_tensor_names_filter() {
        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);

        let mut persister = SafetensorsPersisterConfig::new()
            .with_tensor_names(vec!["encoder.weight", "decoder.bias"])
            .build_in_memory();

        persister.collect_from(&module).unwrap();

        let mut target_module = TestModule::<TestBackend>::new(&device);
        let result = persister.apply_to(&mut target_module).unwrap();
        
        assert_eq!(result.applied.len(), 2);
        assert!(result.applied.contains(&"encoder.weight".to_string()));
        assert!(result.applied.contains(&"decoder.bias".to_string()));
    }

    #[test]
    fn test_predicate_filter() {
        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);

        fn weight_filter(path: &str) -> bool {
            path.ends_with(".weight")
        }

        let mut persister = SafetensorsPersisterConfig::new()
            .with_predicate(weight_filter)
            .build_in_memory();

        persister.collect_from(&module).unwrap();

        let mut target_module = TestModule::<TestBackend>::new(&device);
        let result = persister.apply_to(&mut target_module).unwrap();
        
        assert_eq!(result.applied.len(), 2); // encoder.weight and decoder.weight
        assert!(result.applied.contains(&"encoder.weight".to_string()));
        assert!(result.applied.contains(&"decoder.weight".to_string()));
    }

    #[test]
    fn test_remapping_save() {
        #[cfg(target_has_atomic = "ptr")]
        {
            let device = Default::default();
            let module = TestModule::<TestBackend>::new(&device);

            // Save with remapping to rename during save
            let mut save_persister = SafetensorsPersisterConfig::new()
                .with_remapping(&[("encoder", "enc"), ("decoder", "dec")])
                .build_in_memory();
            save_persister.collect_from(&module).unwrap();

            // Load without remapping - should find the renamed tensors
            let mut load_persister = SafetensorsPersisterConfig::new().build_in_memory();
            load_persister.set_data(save_persister.data().unwrap().to_vec());

            let mut target_module = TestModule::<TestBackend>::new(&device);
            let result = load_persister.apply_to(&mut target_module).unwrap();
            
            // Since we saved with remapping, the file should contain tensors with remapped names.
            // When we load them back, the module should get the renamed tensors applied
            // but since the module expects original names, they won't match.
            // This test demonstrates save-time remapping.
            assert_eq!(result.applied.len(), 0);
            assert_eq!(result.unused.len(), 4);
            // The unused tensors should have the remapped names
            assert!(result.unused.contains(&"enc.weight".to_string()));
            assert!(result.unused.contains(&"enc.bias".to_string()));
            assert!(result.unused.contains(&"dec.weight".to_string()));
            assert!(result.unused.contains(&"dec.bias".to_string()));
        }
    }
}