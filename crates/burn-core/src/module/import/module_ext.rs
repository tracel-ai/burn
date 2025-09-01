use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use hashbrown::HashMap;

#[cfg(target_has_atomic = "ptr")]
use regex::Regex;

use super::{ImportError, ImportResult, TensorApplier, TensorReader};
use crate::module::Module;
use crate::module::export::TensorView;
use crate::tensor::backend::Backend;

/// Extension trait for modules that provides tensor import functionality.
///
/// This trait provides convenient methods to import tensor views into any Burn module
/// using the same dot-notation paths as the export functionality. The import process
/// is lazy - tensor data is only materialized when actually applied to the module.
///
/// # Examples
///
/// ```ignore
/// use burn::module::{ModuleImport, ModuleExport};
///
/// // Direct round-trip from export to import
/// let exported = model1.export_tensor_views();
/// let result = model2.import_tensor_views(exported)?;
/// println!("Imported {} tensors", result.applied.len());
///
/// // Import from a reader (lazy loading)
/// let mut reader = SafeTensorsReader::new(file)?;
/// let result = model.import_from_reader(&mut reader)?;
///
/// // Import with filtering
/// let result = model.import_tensor_views_filtered(
///     views,
///     &[r"^encoder\..*"]  // Only import encoder tensors
/// )?;
///
/// // Import with custom predicate
/// let result = model.import_tensor_views_with_predicate(
///     views,
///     |path| !path.contains("frozen")  // Skip frozen layers
/// )?;
/// ```
pub trait ModuleImport<B: Backend>: Module<B> + Clone {
    /// Import tensor views directly into the module.
    ///
    /// This is the primary import method that applies tensor data from TensorViews
    /// to the corresponding tensors in the module. The views are typically obtained
    /// from `export_tensor_views()` or from a `TensorReader`.
    ///
    /// # Arguments
    ///
    /// * `views` - HashMap of tensor paths to TensorViews
    ///
    /// # Returns
    ///
    /// An `ImportResult` containing information about applied, skipped, missing,
    /// and unused tensors, as well as any errors encountered.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Direct export to import
    /// let exported = model1.export_tensor_views();
    /// let result = model2.import_tensor_views(exported)?;
    ///
    /// if result.is_success() {
    ///     println!("Successfully imported {} tensors", result.applied.len());
    /// } else {
    ///     println!("Import had errors: {:?}", result.errors);
    /// }
    /// ```
    fn import_tensor_views(&mut self, views: HashMap<String, TensorView>) -> ImportResult {
        let mut applier = TensorApplier::new(views);
        *self = self.clone().map(&mut applier);
        applier.into_result()
    }

    /// Import filtered tensor views matching any of the regex patterns.
    ///
    /// Multiple patterns work as an OR union - a tensor is imported if it matches ANY pattern.
    /// This allows selective loading of specific parts of a model, useful for fine-tuning
    /// or partial model updates.
    ///
    /// # Arguments
    ///
    /// * `views` - HashMap of tensor paths to TensorViews
    /// * `patterns` - An iterable of regex patterns
    ///
    /// # Returns
    ///
    /// * `Ok(ImportResult)` - Import results
    /// * `Err(ImportError)` - If any pattern is invalid regex
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Import only encoder tensors
    /// let result = model.import_tensor_views_filtered(
    ///     views,
    ///     &[r"^encoder\..*"]
    /// )?;
    ///
    /// // Import multiple specific parts
    /// let result = model.import_tensor_views_filtered(
    ///     views,
    ///     &[
    ///         r"^encoder\..*",     // All encoder tensors
    ///         r"^decoder\..*",     // All decoder tensors
    ///         r"^head\.weight$",   // Specific head weight
    ///     ]
    /// )?;
    ///
    /// // Import all weights and biases
    /// let result = model.import_tensor_views_filtered(
    ///     views,
    ///     &[r".*\.weight$", r".*\.bias$"]
    /// )?;
    /// ```
    #[cfg(target_has_atomic = "ptr")]
    fn import_tensor_views_filtered<I, S>(
        &mut self,
        views: HashMap<String, TensorView>,
        patterns: I,
    ) -> Result<ImportResult, ImportError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut applier = TensorApplier::with_filter(views, patterns)?;
        *self = self.clone().map(&mut applier);
        Ok(applier.into_result())
    }

    /// Import tensor views filtered by a custom predicate function.
    ///
    /// This method allows you to provide a custom function to filter which tensors
    /// are imported. The function receives the tensor path and should return `true`
    /// to import the tensor or `false` to skip it.
    ///
    /// # Arguments
    ///
    /// * `views` - HashMap of tensor paths to TensorViews
    /// * `device` - Device to create tensors on
    /// * `predicate` - A function that takes a path (&str) and returns bool
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Import only non-frozen layers
    /// let result = model.import_tensor_views_with_predicate(
    ///     views,
    ///     &device,
    ///     |path| !path.contains("frozen")
    /// );
    ///
    /// // Import specific tensors
    /// let result = model.import_tensor_views_with_predicate(
    ///     views,
    ///     &device,
    ///     |path| path == "encoder.weight" || path == "decoder.bias"
    /// );
    ///
    /// // Import based on complex logic
    /// let allowed_layers = vec![3, 4, 5];
    /// let result = model.import_tensor_views_with_predicate(
    ///     views,
    ///     &device,
    ///     move |path| {
    ///         if let Some(layer_num) = extract_layer_number(path) {
    ///             allowed_layers.contains(&layer_num)
    ///         } else {
    ///             false
    ///         }
    ///     }
    /// );
    /// ```
    fn import_tensor_views_with_predicate<F>(
        &mut self,
        views: HashMap<String, TensorView>,
        predicate: F,
    ) -> ImportResult
    where
        F: Fn(&str) -> bool + 'static,
    {
        let mut applier = TensorApplier::with_predicate(views, predicate);
        *self = self.clone().map(&mut applier);
        applier.into_result()
    }

    /// Import tensors from any TensorReader implementation.
    ///
    /// This method provides a convenient way to import from various sources
    /// (files, streams, databases) by using a TensorReader. The reader handles
    /// the lazy loading of tensor data.
    ///
    /// # Arguments
    ///
    /// * `reader` - A mutable reference to a TensorReader
    /// * `device` - Device to create tensors on
    ///
    /// # Returns
    ///
    /// * `Ok(ImportResult)` - Import results
    /// * `Err(ImportError)` - If reading fails
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Import from a SafeTensors file
    /// let file = File::open("model.safetensors")?;
    /// let mut reader = SafeTensorsReader::new(file)?;
    /// let result = model.import_from_reader(&mut reader)?;
    ///
    /// // Import from an NPZ file
    /// let mut reader = NpzReader::new("checkpoint.npz")?;
    /// let result = model.import_from_reader(&mut reader)?;
    ///
    /// // Import from a custom source
    /// struct DatabaseReader { /* ... */ }
    /// impl TensorReader for DatabaseReader { /* ... */ }
    /// let mut reader = DatabaseReader::new(connection);
    /// let result = model.import_from_reader(&mut reader)?;
    /// ```
    fn import_from_reader(
        &mut self,
        reader: &mut dyn TensorReader,
    ) -> Result<ImportResult, ImportError> {
        let views = reader
            .read_all_views()
            .map_err(|e| ImportError::Other(format!("Failed to read views: {}", e)))?;
        Ok(self.import_tensor_views(views))
    }

    /// Import tensors from a reader with filtering.
    ///
    /// Combines reader-based import with regex filtering to selectively load
    /// tensors from a source.
    ///
    /// # Arguments
    ///
    /// * `reader` - A mutable reference to a TensorReader
    /// * `patterns` - An iterable of regex patterns
    ///
    /// # Returns
    ///
    /// * `Ok(ImportResult)` - Import results
    /// * `Err(ImportError)` - If reading fails or regex is invalid
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Import only encoder tensors from a file
    /// let mut reader = SafeTensorsReader::new(file)?;
    /// let result = model.import_from_reader_filtered(
    ///     &mut reader,
    ///     &[r"^encoder\..*"]
    /// )?;
    ///
    /// // Import specific layers
    /// let result = model.import_from_reader_filtered(
    ///     &mut reader,
    ///     &[r"^model\.layer[0-2]\..*"]  // Only layers 0, 1, 2
    /// )?;
    /// ```
    #[cfg(target_has_atomic = "ptr")]
    fn import_from_reader_filtered<I, S>(
        &mut self,
        reader: &mut dyn TensorReader,
        patterns: I,
    ) -> Result<ImportResult, ImportError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let views = reader
            .read_all_views()
            .map_err(|e| ImportError::Other(format!("Failed to read views: {}", e)))?;
        self.import_tensor_views_filtered(views, patterns)
    }

    /// Import tensor views with key remapping using regex patterns.
    ///
    /// This method allows you to transform tensor paths during import, which is useful
    /// when loading models that have different naming conventions or when adapting
    /// models from other frameworks.
    ///
    /// # Arguments
    ///
    /// * `views` - HashMap of tensor paths to TensorViews
    /// * `key_remap` - Vector of (Regex, String) pairs for path transformation
    ///
    /// # Returns
    ///
    /// * `Ok((ImportResult, remapped_names))` - Import results and remapping information
    /// * `Err(ImportError)` - If regex compilation fails
    ///
    /// The returned `remapped_names` is a vector of tuples (new_path, original_path)
    /// showing how each path was transformed.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use regex::Regex;
    ///
    /// // Rename all "layer" to "block"
    /// let remaps = vec![
    ///     (Regex::new(r"layer")?, "block".to_string()),
    /// ];
    /// let (result, remapped) = model.import_tensor_views_remapped(views, remaps)?;
    ///
    /// // Add prefix to all tensors
    /// let remaps = vec![
    ///     (Regex::new(r"^")?, "model.".to_string()),
    /// ];
    /// let (result, remapped) = model.import_tensor_views_remapped(views, remaps)?;
    ///
    /// // Rename specific components
    /// let remaps = vec![
    ///     (Regex::new(r"encoder")?, "enc".to_string()),
    ///     (Regex::new(r"decoder")?, "dec".to_string()),
    ///     (Regex::new(r"weight")?, "w".to_string()),
    ///     (Regex::new(r"bias")?, "b".to_string()),
    /// ];
    /// let (result, remapped) = model.import_tensor_views_remapped(views, remaps)?;
    ///
    /// // Complex transformation - change layer numbering
    /// let remaps = vec![
    ///     (Regex::new(r"layers\.(\d+)")?, "blocks.$1".to_string()),
    /// ];
    /// let (result, remapped) = model.import_tensor_views_remapped(views, remaps)?;
    /// ```
    #[cfg(target_has_atomic = "ptr")]
    fn import_tensor_views_remapped(
        &mut self,
        views: HashMap<String, TensorView>,
        key_remap: Vec<(Regex, String)>,
    ) -> Result<(ImportResult, Vec<(String, String)>), ImportError> {
        let (remapped_views, remapped_names) = remap_tensor_paths(views, key_remap);
        let result = self.import_tensor_views(remapped_views);
        Ok((result, remapped_names))
    }

    /// Import tensor views with key remapping and filtering.
    ///
    /// Combines remapping and filtering - first remaps the paths, then applies filters
    /// to the remapped paths.
    ///
    /// # Arguments
    ///
    /// * `views` - HashMap of tensor paths to TensorViews
    /// * `key_remap` - Vector of (Regex, String) pairs for path transformation
    /// * `patterns` - Regex patterns to filter which tensors to import (applied after remapping)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use regex::Regex;
    ///
    /// // First rename layer to block, then only import block 0 and 1
    /// let remaps = vec![
    ///     (Regex::new(r"layer")?, "block".to_string()),
    /// ];
    /// let (result, remapped) = model.import_tensor_views_remapped_filtered(
    ///     views,
    ///     remaps,
    ///     &[r"^block\.[01]\..*"]
    /// )?;
    /// ```
    #[cfg(target_has_atomic = "ptr")]
    fn import_tensor_views_remapped_filtered<I, S>(
        &mut self,
        views: HashMap<String, TensorView>,
        key_remap: Vec<(Regex, String)>,
        patterns: I,
    ) -> Result<(ImportResult, Vec<(String, String)>), ImportError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let (remapped_views, remapped_names) = remap_tensor_paths(views, key_remap);
        let result = self.import_tensor_views_filtered(remapped_views, patterns)?;
        Ok((result, remapped_names))
    }
}

/// Remap tensor paths using regex patterns.
///
/// This function transforms the keys in a HashMap of tensor views according to
/// the provided regex patterns and replacement strings.
///
/// # Arguments
///
/// * `tensors` - HashMap of original tensor paths to TensorViews
/// * `key_remap` - Vector of (Regex, String) pairs for transformation
///
/// # Returns
///
/// A tuple containing:
/// * The remapped HashMap with transformed keys
/// * A vector of (new_path, original_path) showing the transformations
#[cfg(target_has_atomic = "ptr")]
fn remap_tensor_paths(
    mut tensors: HashMap<String, TensorView>,
    key_remap: Vec<(Regex, String)>,
) -> (HashMap<String, TensorView>, Vec<(String, String)>) {
    if key_remap.is_empty() {
        let remapped_names = tensors.keys().cloned().map(|s| (s.clone(), s)).collect();
        return (tensors, remapped_names);
    }

    let mut remapped = HashMap::new();
    let mut remapped_names = Vec::new();

    for (name, tensor) in tensors.drain() {
        let mut new_name = name.clone();
        for (pattern, replacement) in &key_remap {
            if pattern.is_match(&new_name) {
                new_name = pattern
                    .replace_all(&new_name, replacement.as_str())
                    .to_string();
            }
        }

        remapped_names.push((new_name.clone(), name));
        remapped.insert(new_name, tensor);
    }

    (remapped, remapped_names)
}

// Blanket implementation for all modules that implement Clone
impl<B: Backend, M: Module<B> + Clone> ModuleImport<B> for M {}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate as burn;
    use crate::{
        TestBackend,
        module::{Module, ModuleExport, Param},
        nn::{Linear, LinearConfig},
    };
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
                encoder: TestSubModule::new(device, 1.0),
                decoder: TestSubModule::new(device, 2.0),
            }
        }

        fn new_zeros(device: &B::Device) -> Self {
            Self {
                encoder: TestSubModule::new(device, 0.0),
                decoder: TestSubModule::new(device, 0.0),
            }
        }
    }

    impl<B: Backend> TestSubModule<B> {
        fn new(device: &B::Device, value: f32) -> Self {
            Self {
                weight: Param::from_data(
                    [[value, value * 2.0], [value * 3.0, value * 4.0]],
                    device,
                ),
                bias: Param::from_data([value * 5.0, value * 6.0], device),
            }
        }
    }

    #[test]
    fn test_import_export_round_trip() {
        let device = Default::default();
        let model1 = TestModule::<TestBackend>::new(&device);
        let mut model2 = TestModule::<TestBackend>::new_zeros(&device);

        // Export from model1
        let exported = model1.export_tensor_views();
        assert_eq!(exported.len(), 4);

        // Import into model2
        let result = model2.import_tensor_views(exported);

        assert!(result.is_success());
        assert_eq!(result.applied.len(), 4);
        assert_eq!(result.errors.len(), 0);

        // Verify the tensors were imported correctly
        let model2_exported = model2.export_tensor_views();
        let encoder_weight_data = model2_exported.get("encoder.weight").unwrap().to_data();
        assert_eq!(
            encoder_weight_data.to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn test_import_with_filter() {
        let device = Default::default();
        let model1 = TestModule::<TestBackend>::new(&device);
        let mut model2 = TestModule::<TestBackend>::new_zeros(&device);

        // Export all from model1
        let exported = model1.export_tensor_views();

        // Import only encoder tensors into model2
        let result = model2
            .import_tensor_views_filtered(exported, &[r"^encoder\..*"])
            .unwrap();

        assert!(result.is_success());
        assert_eq!(result.applied.len(), 2); // encoder.weight and encoder.bias
        assert_eq!(result.skipped.len(), 2); // decoder tensors were skipped

        // Verify only encoder was updated
        let model2_exported = model2.export_tensor_views();
        let encoder_weight_data = model2_exported.get("encoder.weight").unwrap().to_data();
        let decoder_weight_data = model2_exported.get("decoder.weight").unwrap().to_data();

        // Encoder should be updated
        assert_eq!(
            encoder_weight_data.to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        // Decoder should still be zeros
        assert_eq!(
            decoder_weight_data.to_vec::<f32>().unwrap(),
            vec![0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn test_import_with_predicate() {
        let device = Default::default();
        let model1 = TestModule::<TestBackend>::new(&device);
        let mut model2 = TestModule::<TestBackend>::new_zeros(&device);

        // Export all from model1
        let exported = model1.export_tensor_views();

        // Import only weight tensors
        let result = model2
            .import_tensor_views_with_predicate(exported, |path: &str| path.ends_with(".weight"));

        assert!(result.is_success());
        assert_eq!(result.applied.len(), 2); // encoder.weight and decoder.weight
        assert_eq!(result.skipped.len(), 2); // bias tensors were skipped

        // Verify only weights were updated
        let model2_exported = model2.export_tensor_views();
        let encoder_weight_data = model2_exported.get("encoder.weight").unwrap().to_data();
        let encoder_bias_data = model2_exported.get("encoder.bias").unwrap().to_data();

        // Weight should be updated
        assert_eq!(
            encoder_weight_data.to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        // Bias should still be zeros
        assert_eq!(encoder_bias_data.to_vec::<f32>().unwrap(), vec![0.0, 0.0]);
    }

    #[test]
    fn test_import_partial_views() {
        let device = Default::default();
        let model1 = TestModule::<TestBackend>::new(&device);
        let mut model2 = TestModule::<TestBackend>::new_zeros(&device);

        // Export only encoder from model1
        let exported = model1
            .export_tensor_views_filtered(&[r"^encoder\..*"])
            .unwrap();
        assert_eq!(exported.len(), 2);

        // Import into model2
        let result = model2.import_tensor_views(exported);

        assert!(result.is_success());
        assert_eq!(result.applied.len(), 2); // Only encoder tensors
        assert_eq!(result.missing.len(), 2); // decoder tensors are missing from import
        assert_eq!(result.unused.len(), 0); // All provided tensors were used

        // Verify partial import
        let model2_exported = model2.export_tensor_views();
        let encoder_weight_data = model2_exported.get("encoder.weight").unwrap().to_data();
        let decoder_weight_data = model2_exported.get("decoder.weight").unwrap().to_data();

        // Encoder should be updated
        assert_eq!(
            encoder_weight_data.to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        // Decoder should still be zeros
        assert_eq!(
            decoder_weight_data.to_vec::<f32>().unwrap(),
            vec![0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn test_import_with_linear_module() {
        let device = Default::default();
        let linear1 = LinearConfig::new(10, 20)
            .with_bias(true)
            .init::<TestBackend>(&device);
        let mut linear2 = LinearConfig::new(10, 20)
            .with_bias(true)
            .init::<TestBackend>(&device);

        // Export from linear1
        let exported = linear1.export_tensor_views();
        assert_eq!(exported.len(), 2); // weight and bias

        // Import into linear2
        let result = linear2.import_tensor_views(exported);

        assert!(result.is_success());
        assert_eq!(result.applied.len(), 2);
        assert_eq!(result.errors.len(), 0);

        // Verify shapes are correct
        let linear2_exported = linear2.export_tensor_views();
        let weight_data = linear2_exported.get("weight").unwrap().to_data();
        let bias_data = linear2_exported.get("bias").unwrap().to_data();

        assert_eq!(weight_data.shape, vec![10, 20]);
        assert_eq!(bias_data.shape, vec![20]);
    }

    // Test with mismatched shapes
    #[test]
    fn test_import_shape_mismatch() {
        let device = Default::default();

        // Create models with different shapes
        let linear1 = LinearConfig::new(10, 20)
            .with_bias(true)
            .init::<TestBackend>(&device);
        let mut linear2 = LinearConfig::new(5, 20) // Different input size
            .with_bias(true)
            .init::<TestBackend>(&device);

        // Export from linear1
        let exported = linear1.export_tensor_views();

        // Try to import into linear2 (should fail for weight, succeed for bias)
        let result = linear2.import_tensor_views(exported);

        assert!(!result.is_success());
        assert_eq!(result.applied.len(), 1); // Only bias should succeed
        assert_eq!(result.errors.len(), 1); // Weight should fail
        assert!(result.errors[0].contains("Shape mismatch"));
    }

    // Test with complex filtering patterns
    #[test]
    fn test_import_complex_filtering() {
        let device = Default::default();
        let model1 = TestModule::<TestBackend>::new(&device);
        let mut model2 = TestModule::<TestBackend>::new_zeros(&device);

        // Export all from model1
        let exported = model1.export_tensor_views();

        // Import with multiple patterns (OR union)
        let result = model2
            .import_tensor_views_filtered(
                exported,
                &[
                    r"^encoder\.weight$", // Specific encoder weight
                    r".*\.bias$",         // All biases
                ],
            )
            .unwrap();

        assert!(result.is_success());
        assert_eq!(result.applied.len(), 3); // encoder.weight, encoder.bias, decoder.bias
        assert_eq!(result.skipped.len(), 1); // decoder.weight

        // Verify the import
        let model2_exported = model2.export_tensor_views();
        let encoder_weight_data = model2_exported.get("encoder.weight").unwrap().to_data();
        let decoder_weight_data = model2_exported.get("decoder.weight").unwrap().to_data();
        let decoder_bias_data = model2_exported.get("decoder.bias").unwrap().to_data();

        // These should be updated
        assert_eq!(
            encoder_weight_data.to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(decoder_bias_data.to_vec::<f32>().unwrap(), vec![10.0, 12.0]);
        // This should still be zero
        assert_eq!(
            decoder_weight_data.to_vec::<f32>().unwrap(),
            vec![0.0, 0.0, 0.0, 0.0]
        );
    }

    // Test Vec of modules
    #[derive(Module, Debug)]
    struct VecModule<B: Backend> {
        layers: Vec<Linear<B>>,
    }

    impl<B: Backend> VecModule<B> {
        fn new(device: &B::Device, num_layers: usize) -> Self {
            Self {
                layers: (0..num_layers)
                    .map(|_| LinearConfig::new(4, 4).with_bias(true).init(device))
                    .collect(),
            }
        }

        fn new_zeros(device: &B::Device, num_layers: usize) -> Self {
            Self {
                layers: (0..num_layers)
                    .map(|_| {
                        let mut module = LinearConfig::new(4, 4).with_bias(true).init(device);
                        // Zero out the weights and biases
                        module.weight = Param::from_tensor(Tensor::zeros([4, 4], device));
                        module.bias = Some(Param::from_tensor(Tensor::zeros([4], device)));
                        module
                    })
                    .collect(),
            }
        }
    }

    #[test]
    fn test_import_vec_module() {
        let device = Default::default();
        let model1 = VecModule::<TestBackend>::new(&device, 3);
        let mut model2 = VecModule::<TestBackend>::new_zeros(&device, 3);

        // Export from model1
        let exported = model1.export_tensor_views();

        // Should have 6 tensors (3 layers × 2 tensors each)
        assert_eq!(exported.len(), 6);
        assert!(exported.contains_key("layers.0.weight"));
        assert!(exported.contains_key("layers.0.bias"));
        assert!(exported.contains_key("layers.1.weight"));
        assert!(exported.contains_key("layers.1.bias"));
        assert!(exported.contains_key("layers.2.weight"));
        assert!(exported.contains_key("layers.2.bias"));

        // Import into model2
        let result = model2.import_tensor_views(exported);

        assert!(result.is_success());
        assert_eq!(result.applied.len(), 6);
        assert_eq!(result.errors.len(), 0);

        // Verify the tensors were imported correctly
        let model2_exported = model2.export_tensor_views();

        // Check that all tensors are non-zero after import
        for i in 0..3 {
            let weight_path = format!("layers.{}.weight", i);
            let bias_path = format!("layers.{}.bias", i);

            let weight_data = model2_exported.get(&weight_path).unwrap().to_data();
            let bias_data = model2_exported.get(&bias_path).unwrap().to_data();

            // Shapes should be correct
            assert_eq!(weight_data.shape, vec![4, 4]);
            assert_eq!(bias_data.shape, vec![4]);
        }
    }

    #[test]
    fn test_import_vec_module_filtered() {
        let device = Default::default();
        let model1 = VecModule::<TestBackend>::new(&device, 3);
        let mut model2 = VecModule::<TestBackend>::new_zeros(&device, 3);

        // Export all from model1
        let exported = model1.export_tensor_views();

        // Import only layer 1 tensors
        let result = model2
            .import_tensor_views_filtered(exported, &[r"^layers\.1\..*"])
            .unwrap();

        assert!(result.is_success());
        assert_eq!(result.applied.len(), 2); // Only layer 1 tensors
        assert_eq!(result.skipped.len(), 4); // Other layers skipped
    }

    // Test array of modules
    #[derive(Module, Debug)]
    struct ArrayModule<B: Backend> {
        layers: [Linear<B>; 3],
    }

    impl<B: Backend> ArrayModule<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                layers: [
                    LinearConfig::new(4, 4).with_bias(true).init(device),
                    LinearConfig::new(4, 4).with_bias(true).init(device),
                    LinearConfig::new(4, 4).with_bias(true).init(device),
                ],
            }
        }

        fn new_zeros(device: &B::Device) -> Self {
            let create_zero_module = || {
                let mut module = LinearConfig::new(4, 4).with_bias(true).init(device);
                module.weight = Param::from_tensor(Tensor::zeros([4, 4], device));
                module.bias = Some(Param::from_tensor(Tensor::zeros([4], device)));
                module
            };

            Self {
                layers: [
                    create_zero_module(),
                    create_zero_module(),
                    create_zero_module(),
                ],
            }
        }
    }

    #[test]
    fn test_import_array_module() {
        let device = Default::default();
        let model1 = ArrayModule::<TestBackend>::new(&device);
        let mut model2 = ArrayModule::<TestBackend>::new_zeros(&device);

        // Export from model1
        let exported = model1.export_tensor_views();

        // Should have 6 tensors (3 layers × 2 tensors each)
        assert_eq!(exported.len(), 6);
        for i in 0..3 {
            assert!(exported.contains_key(&format!("layers.{}.weight", i)));
            assert!(exported.contains_key(&format!("layers.{}.bias", i)));
        }

        // Import into model2
        let result = model2.import_tensor_views(exported);

        assert!(result.is_success());
        assert_eq!(result.applied.len(), 6);
        assert_eq!(result.errors.len(), 0);

        // Verify shapes
        let model2_exported = model2.export_tensor_views();
        for i in 0..3 {
            let weight_data = model2_exported
                .get(&format!("layers.{}.weight", i))
                .unwrap()
                .to_data();
            let bias_data = model2_exported
                .get(&format!("layers.{}.bias", i))
                .unwrap()
                .to_data();

            assert_eq!(weight_data.shape, vec![4, 4]);
            assert_eq!(bias_data.shape, vec![4]);
        }
    }

    // Test enum modules
    #[derive(Module, Debug)]
    enum EnumModule<B: Backend> {
        Small(Linear<B>),
        Medium(Linear<B>),
        Large(Linear<B>),
    }

    impl<B: Backend> EnumModule<B> {
        fn new_small(device: &B::Device) -> Self {
            Self::Small(LinearConfig::new(2, 2).with_bias(true).init(device))
        }

        fn new_small_zeros(device: &B::Device) -> Self {
            let mut module = LinearConfig::new(2, 2).with_bias(true).init(device);
            module.weight = Param::from_tensor(Tensor::zeros([2, 2], device));
            module.bias = Some(Param::from_tensor(Tensor::zeros([2], device)));
            Self::Small(module)
        }
    }

    #[test]
    fn test_import_enum_module() {
        let device = Default::default();
        let model1 = EnumModule::<TestBackend>::new_small(&device);
        let mut model2 = EnumModule::<TestBackend>::new_small_zeros(&device);

        // Export from model1
        let exported = model1.export_tensor_views();

        // Should have variant name in the path
        assert_eq!(exported.len(), 2);
        assert!(exported.contains_key("Small.weight"));
        assert!(exported.contains_key("Small.bias"));

        // Import into model2
        let result = model2.import_tensor_views(exported);

        assert!(result.is_success());
        assert_eq!(result.applied.len(), 2);
        assert_eq!(result.errors.len(), 0);

        // Verify the import
        let model2_exported = model2.export_tensor_views();
        let weight_data = model2_exported.get("Small.weight").unwrap().to_data();
        let bias_data = model2_exported.get("Small.bias").unwrap().to_data();

        assert_eq!(weight_data.shape, vec![2, 2]);
        assert_eq!(bias_data.shape, vec![2]);
    }

    // Test deeply nested module with vecs
    #[derive(Module, Debug)]
    struct NestedWithVec<B: Backend> {
        encoder: VecModule<B>,
        decoder: VecModule<B>,
    }

    impl<B: Backend> NestedWithVec<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                encoder: VecModule::new(device, 2),
                decoder: VecModule::new(device, 2),
            }
        }

        fn new_zeros(device: &B::Device) -> Self {
            Self {
                encoder: VecModule::new_zeros(device, 2),
                decoder: VecModule::new_zeros(device, 2),
            }
        }
    }

    #[test]
    fn test_import_nested_with_vec() {
        let device = Default::default();
        let model1 = NestedWithVec::<TestBackend>::new(&device);
        let mut model2 = NestedWithVec::<TestBackend>::new_zeros(&device);

        // Export from model1
        let exported = model1.export_tensor_views();

        // Should have 8 tensors (2 modules × 2 layers × 2 tensors)
        assert_eq!(exported.len(), 8);
        assert!(exported.contains_key("encoder.layers.0.weight"));
        assert!(exported.contains_key("encoder.layers.0.bias"));
        assert!(exported.contains_key("encoder.layers.1.weight"));
        assert!(exported.contains_key("encoder.layers.1.bias"));
        assert!(exported.contains_key("decoder.layers.0.weight"));
        assert!(exported.contains_key("decoder.layers.0.bias"));
        assert!(exported.contains_key("decoder.layers.1.weight"));
        assert!(exported.contains_key("decoder.layers.1.bias"));

        // Import into model2
        let result = model2.import_tensor_views(exported);

        assert!(result.is_success());
        assert_eq!(result.applied.len(), 8);
        assert_eq!(result.errors.len(), 0);

        // Test selective import - only encoder.layers.0
        let model1 = NestedWithVec::<TestBackend>::new(&device);
        let mut model2 = NestedWithVec::<TestBackend>::new_zeros(&device);

        let exported = model1.export_tensor_views();
        let result = model2
            .import_tensor_views_filtered(exported, &[r"^encoder\.layers\.0\..*"])
            .unwrap();

        assert!(result.is_success());
        assert_eq!(result.applied.len(), 2); // Only encoder.layers.0 tensors
        assert_eq!(result.skipped.len(), 6); // Rest are skipped
    }

    // Test optional fields
    #[derive(Module, Debug)]
    struct OptionalModule<B: Backend> {
        required: Linear<B>,
        optional: Option<Linear<B>>,
    }

    impl<B: Backend> OptionalModule<B> {
        fn new_with_optional(device: &B::Device) -> Self {
            Self {
                required: LinearConfig::new(4, 4).with_bias(true).init(device),
                optional: Some(LinearConfig::new(4, 4).with_bias(true).init(device)),
            }
        }

        fn new_with_optional_zeros(device: &B::Device) -> Self {
            let create_zero_module = || {
                let mut module = LinearConfig::new(4, 4).with_bias(true).init(device);
                module.weight = Param::from_tensor(Tensor::zeros([4, 4], device));
                module.bias = Some(Param::from_tensor(Tensor::zeros([4], device)));
                module
            };

            Self {
                required: create_zero_module(),
                optional: Some(create_zero_module()),
            }
        }

        fn new_without_optional(device: &B::Device) -> Self {
            Self {
                required: LinearConfig::new(4, 4).with_bias(true).init(device),
                optional: None,
            }
        }
    }

    #[test]
    fn test_import_with_remapping() {
        let device = Default::default();
        let model1 = TestModule::<TestBackend>::new(&device);
        let mut model2 = TestModule::<TestBackend>::new_zeros(&device);

        // Export from model1 and simulate external naming (e.g., from another framework)
        let mut external_views = model1.export_tensor_views();

        // Simulate external naming convention: enc -> encoder, dec -> decoder
        let mut renamed_views = HashMap::new();
        for (key, value) in external_views.drain() {
            let new_key = key.replace("encoder", "enc").replace("decoder", "dec");
            renamed_views.insert(new_key, value);
        }

        // Now we have tensors with external names: enc.weight, enc.bias, dec.weight, dec.bias
        // We need to remap them back to match our module: encoder.*, decoder.*
        let remaps = vec![
            (Regex::new(r"^enc\.").unwrap(), "encoder.".to_string()),
            (Regex::new(r"^dec\.").unwrap(), "decoder.".to_string()),
        ];

        let (result, remapped_names) = model2
            .import_tensor_views_remapped(renamed_views, remaps)
            .unwrap();

        assert!(result.is_success());
        assert_eq!(result.applied.len(), 4); // All tensors should be applied

        // Check remapping happened correctly
        let found_encoder_weight = remapped_names
            .iter()
            .any(|(new, old)| new == "encoder.weight" && old == "enc.weight");
        let found_decoder_bias = remapped_names
            .iter()
            .any(|(new, old)| new == "decoder.bias" && old == "dec.bias");

        assert!(
            found_encoder_weight,
            "enc.weight should be remapped to encoder.weight"
        );
        assert!(
            found_decoder_bias,
            "dec.bias should be remapped to decoder.bias"
        );

        // Verify the data was actually imported
        let model2_exported = model2.export_tensor_views();
        assert!(model2_exported.contains_key("encoder.weight"));
        assert!(model2_exported.contains_key("decoder.bias"));
    }

    #[test]
    fn test_import_with_complex_remapping() {
        let device = Default::default();
        let model1 = VecModule::<TestBackend>::new(&device, 3);
        let mut model2 = VecModule::<TestBackend>::new_zeros(&device, 3);

        // Export from model1 and simulate external naming
        let mut external_views = model1.export_tensor_views();

        // Simulate external naming convention where "layers" is called "blocks"
        let mut renamed_views = HashMap::new();
        for (key, value) in external_views.drain() {
            let new_key = key.replace("layers", "blocks");
            renamed_views.insert(new_key, value);
        }

        // Now remap blocks.N back to layers.N to match our module
        let remaps = vec![(
            Regex::new(r"blocks\.(\d+)").unwrap(),
            "layers.$1".to_string(),
        )];

        let (result, remapped_names) = model2
            .import_tensor_views_remapped(renamed_views, remaps)
            .unwrap();

        assert!(result.is_success());
        assert_eq!(result.applied.len(), 6);

        // Check that blocks.0.weight was remapped to layers.0.weight
        let found_layer0 = remapped_names
            .iter()
            .any(|(new, old)| new == "layers.0.weight" && old == "blocks.0.weight");
        assert!(
            found_layer0,
            "blocks.0.weight should be remapped to layers.0.weight"
        );

        // Export and verify the data was actually imported
        let model2_exported = model2.export_tensor_views();
        assert!(model2_exported.contains_key("layers.0.weight"));
        assert!(model2_exported.contains_key("layers.2.bias"));
    }

    #[test]
    fn test_import_remapped_filtered() {
        let device = Default::default();
        let model1 = VecModule::<TestBackend>::new(&device, 3);
        let mut model2 = VecModule::<TestBackend>::new_zeros(&device, 3);

        // Export from model1 and simulate external naming
        let mut external_views = model1.export_tensor_views();

        // Simulate external naming where "layers" is called "blocks"
        let mut renamed_views = HashMap::new();
        for (key, value) in external_views.drain() {
            let new_key = key.replace("layers", "blocks");
            renamed_views.insert(new_key, value);
        }

        // Remap blocks back to layers, then filter to only import layers 0 and 1
        let remaps = vec![(Regex::new(r"blocks").unwrap(), "layers".to_string())];

        let (result, remapped_names) = model2
            .import_tensor_views_remapped_filtered(renamed_views, remaps, &[r"^layers\.[01]\..*"])
            .unwrap();

        assert!(result.is_success());
        assert_eq!(result.applied.len(), 4); // Only layers 0 and 1 (2 tensors each)
        assert_eq!(result.skipped.len(), 2); // layers.2 tensors skipped

        // Verify remapping
        let found_layer1_weight = remapped_names
            .iter()
            .any(|(new, old)| new == "layers.1.weight" && old == "blocks.1.weight");
        assert!(
            found_layer1_weight,
            "blocks.1.weight should be remapped to layers.1.weight"
        );
    }

    #[test]
    fn test_import_optional_module() {
        let device = Default::default();

        // Test with optional present
        let model1 = OptionalModule::<TestBackend>::new_with_optional(&device);
        let mut model2 = OptionalModule::<TestBackend>::new_with_optional_zeros(&device);

        let exported = model1.export_tensor_views();
        assert_eq!(exported.len(), 4); // 2 modules × 2 tensors

        let result = model2.import_tensor_views(exported);
        assert!(result.is_success());
        assert_eq!(result.applied.len(), 4);

        // Test with optional absent - export from model without optional
        let model1 = OptionalModule::<TestBackend>::new_without_optional(&device);
        let mut model2 = OptionalModule::<TestBackend>::new_with_optional_zeros(&device);

        let exported = model1.export_tensor_views();
        assert_eq!(exported.len(), 2); // Only required module

        let result = model2.import_tensor_views(exported);
        assert!(result.is_success());
        assert_eq!(result.applied.len(), 2); // Only required tensors applied
        assert_eq!(result.missing.len(), 2); // Optional tensors are missing
    }
}
