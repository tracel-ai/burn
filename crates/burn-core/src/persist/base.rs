use alloc::string::{String, ToString};
use alloc::vec::Vec;
use hashbrown::HashMap;

#[cfg(target_has_atomic = "ptr")]
use regex::{self, Regex};

use super::{
    TensorViewCollector,
    appliers::{ImportError, ImportResult, TensorApplier},
};
use crate::module::Module;
use crate::persist::TensorView;
use crate::tensor::backend::Backend;

/// Extension trait for modules that provides tensor persistence functionality.
///
/// This trait provides convenient methods to collect and apply tensor views from any Burn module.
/// Collection operations create lightweight tensor views without immediately copying data.
/// Apply operations apply tensor data from views to the corresponding tensors in the module.
///
/// # Examples
///
/// ```ignore
/// use burn::persist::ModulePersist;
///
/// // Collect all tensors
/// let all_views = model.collect();
/// for (path, view) in all_views.iter() {
///     println!("{}: {:?}", path, view.to_data().shape);
/// }
///
/// // Collect only encoder tensors
/// let encoder_views = model.collect_filtered(&[r"^encoder\..*"])?;
///
/// // Apply tensor views to another model
/// let result = model2.apply(collected)?;
/// println!("Applied {} tensors", result.applied.len());
///
/// // Apply with filtering
/// let result = model.apply_filtered(
///     views,
///     &[r"^encoder\..*"]  // Only apply encoder tensors
/// )?;
/// ```
pub trait ModulePersist<B: Backend>: Module<B> + Clone {
    /// Collect tensor views for inspection without copying data.
    ///
    /// Returns a HashMap where keys are the full module paths (e.g., "encoder.layer1.weight")
    /// and values are `TensorView` objects that can lazily materialize the tensor data.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let views = model.collect();
    /// println!("Total tensors: {}", views.len());
    ///
    /// // Materialize specific tensor data when needed
    /// if let Some(weight_view) = views.get("encoder.weight") {
    ///     let data = weight_view.to_data();
    ///     println!("Encoder weight shape: {:?}", data.shape);
    /// }
    /// ```
    fn collect(&self) -> HashMap<String, TensorView> {
        let mut collector = TensorViewCollector::new();
        self.visit(&mut collector);
        collector.tensors
    }

    /// Collect filtered tensor views matching any of the regex patterns.
    ///
    /// Multiple patterns work as an OR union - a tensor is collected if it matches ANY pattern.
    /// This allows flexible filtering strategies for complex module hierarchies.
    ///
    /// # Arguments
    ///
    /// * `patterns` - An iterable of regex patterns. Can be a slice, Vec, or any IntoIterator.
    ///
    /// # Returns
    ///
    /// * `Ok(HashMap)` - Map of matching tensor paths to their views
    /// * `Err(regex::Error)` - If any pattern is invalid regex
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Single pattern - collect only encoder tensors
    /// let encoder_tensors = model.collect_filtered(&[
    ///     r"^encoder\..*"
    /// ])?;
    ///
    /// // Multiple patterns (OR union) - collect encoder OR decoder tensors
    /// let tensors = model.collect_filtered(&[
    ///     r"^encoder\..*",
    ///     r"^decoder\..*",
    /// ])?;
    ///
    /// // Collect all weights and biases
    /// let params = model.collect_filtered(&[
    ///     r".*\.weight$",
    ///     r".*\.bias$",
    /// ])?;
    ///
    /// // Complex filtering - specific layers and tensor types
    /// let filtered = model.collect_filtered(&[
    ///     r"^model\.layer[0-2]\..*",        // layers 0, 1, 2
    ///     r"^attention\..*\.(query|key)$",  // attention Q and K
    ///     r"^head\..*",                     // all head tensors
    /// ])?;
    ///
    /// // Using with Vec for dynamic patterns
    /// let mut patterns = vec![r"^encoder\..*"];
    /// if include_decoder {
    ///     patterns.push(r"^decoder\..*");
    /// }
    /// let tensors = model.collect_filtered(patterns)?;
    /// ```
    #[cfg(target_has_atomic = "ptr")]
    fn collect_filtered<I, S>(
        &self,
        patterns: I,
    ) -> Result<HashMap<String, TensorView>, regex::Error>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut collector = TensorViewCollector::with_filter(patterns)?;
        self.visit(&mut collector);
        Ok(collector.tensors)
    }

    /// Collect tensor views filtered by a custom predicate function.
    ///
    /// This method allows you to provide a custom function to filter which tensors
    /// are collected. The function receives the tensor path (e.g., "encoder.layer1.weight")
    /// and should return `true` to include the tensor or `false` to exclude it.
    ///
    /// # Arguments
    ///
    /// * `predicate` - A function that takes a tensor path (&str) and returns bool
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Collect only tensors with specific names
    /// let tensors = model.collect_with_predicate(|path| {
    ///     path == "encoder.weight" || path == "decoder.bias"
    /// });
    ///
    /// // Collect tensors based on custom logic
    /// let large_tensors = model.collect_with_predicate(|path| {
    ///     // Only collect tensors from layers 3, 4, and 5
    ///     if let Some(captures) = regex::Regex::new(r"layer(\d+)")
    ///         .unwrap()
    ///         .captures(path)
    ///     {
    ///         if let Some(layer_num) = captures.get(1) {
    ///             if let Ok(num) = layer_num.as_str().parse::<u32>() {
    ///                 return num >= 3 && num <= 5;
    ///             }
    ///         }
    ///     }
    ///     false
    /// });
    ///
    /// // Collect tensors that don't contain certain keywords
    /// let filtered = model.collect_with_predicate(|path| {
    ///     !path.contains("dropout") && !path.contains("auxiliary")
    /// });
    ///
    /// // Combine multiple conditions
    /// let specific_tensors = model.collect_with_predicate(|path| {
    ///     let is_encoder = path.starts_with("encoder.");
    ///     let is_weight = path.ends_with(".weight");
    ///     let not_attention = !path.contains("attention");
    ///     
    ///     is_encoder && is_weight && not_attention
    /// });
    ///
    /// // Use with closure capturing external state
    /// let allowed_prefixes = vec!["encoder", "decoder", "head"];
    /// let tensors = model.collect_with_predicate(|path| {
    ///     allowed_prefixes.iter().any(|prefix| path.starts_with(prefix))
    /// });
    /// ```
    fn collect_with_predicate<F>(&self, predicate: F) -> HashMap<String, TensorView>
    where
        F: Fn(&str) -> bool + 'static,
    {
        let mut collector = TensorViewCollector::with_predicate(predicate);
        self.visit(&mut collector);
        collector.tensors
    }

    /// Apply tensor views directly to the module.
    ///
    /// This is the primary apply method that applies tensor data from TensorViews
    /// to the corresponding tensors in the module. The views are typically obtained
    /// from `collect()`
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
    /// // Direct collect to apply
    /// let collected = model1.collect();
    /// let result = model2.apply(collected)?;
    ///
    /// if result.is_success() {
    ///     println!("Successfully applied {} tensors", result.applied.len());
    /// } else {
    ///     println!("Apply had errors: {:?}", result.errors);
    /// }
    /// ```
    fn apply(&mut self, views: HashMap<String, TensorView>) -> ImportResult {
        let mut applier = TensorApplier::new(views);
        *self = self.clone().map(&mut applier);
        applier.into_result()
    }

    /// Apply filtered tensor views matching any of the regex patterns.
    ///
    /// Multiple patterns work as an OR union - a tensor is applied if it matches ANY pattern.
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
    /// // Apply only encoder tensors
    /// let result = model.apply_filtered(
    ///     views,
    ///     &[r"^encoder\..*"]
    /// )?;
    ///
    /// // Apply multiple specific parts
    /// let result = model.apply_filtered(
    ///     views,
    ///     &[
    ///         r"^encoder\..*",     // All encoder tensors
    ///         r"^decoder\..*",     // All decoder tensors
    ///         r"^head\.weight$",   // Specific head weight
    ///     ]
    /// )?;
    ///
    /// // Apply all weights and biases
    /// let result = model.apply_filtered(
    ///     views,
    ///     &[r".*\.weight$", r".*\.bias$"]
    /// )?;
    /// ```
    #[cfg(target_has_atomic = "ptr")]
    fn apply_filtered<I, S>(
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

    /// Apply tensor views filtered by a custom predicate function.
    ///
    /// This method allows you to provide a custom function to filter which tensors
    /// are applied. The function receives the tensor path and should return `true`
    /// to apply the tensor or `false` to skip it.
    ///
    /// # Arguments
    ///
    /// * `views` - HashMap of tensor paths to TensorViews
    /// * `predicate` - A function that takes a path (&str) and returns bool
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Apply only non-frozen layers
    /// let result = model.apply_with_predicate(
    ///     views,
    ///     |path| !path.contains("frozen")
    /// );
    ///
    /// // Apply specific tensors
    /// let result = model.apply_with_predicate(
    ///     views,
    ///     |path| path == "encoder.weight" || path == "decoder.bias"
    /// );
    ///
    /// // Apply based on complex logic
    /// let allowed_layers = vec![3, 4, 5];
    /// let result = model.apply_with_predicate(
    ///     views,
    ///     move |path| {
    ///         if let Some(layer_num) = extract_layer_number(path) {
    ///             allowed_layers.contains(&layer_num)
    ///         } else {
    ///             false
    ///         }
    ///     }
    /// );
    /// ```
    fn apply_with_predicate<F>(
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

    /// Apply tensor views with key remapping using regex patterns.
    ///
    /// This method allows you to transform tensor paths during apply, which is useful
    /// when loading models that have different naming conventions or when adapting
    /// models from other frameworks.
    ///
    /// # Arguments
    ///
    /// * `views` - HashMap of tensor paths to TensorViews
    /// * `key_remap` - Vector of (pattern, replacement) string pairs for path transformation
    ///
    /// # Returns
    ///
    /// * `Ok((ImportResult, remapped_names))` - Apply results and remapping information
    /// * `Err(ImportError)` - If regex compilation fails
    ///
    /// The returned `remapped_names` is a vector of tuples (new_path, original_path)
    /// showing how each path was transformed.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Rename all "layer" to "block"
    /// let remaps = vec![
    ///     (r"layer", "block"),
    /// ];
    /// let (result, remapped) = model.apply_remapped(views, remaps)?;
    ///
    /// // Add prefix to all tensors
    /// let remaps = vec![
    ///     (r"^", "model."),
    /// ];
    /// let (result, remapped) = model.apply_remapped(views, remaps)?;
    ///
    /// // Rename specific components
    /// let remaps = vec![
    ///     (r"encoder", "enc"),
    ///     (r"decoder", "dec"),
    ///     (r"weight", "w"),
    ///     (r"bias", "b"),
    /// ];
    /// let (result, remapped) = model.apply_remapped(views, remaps)?;
    ///
    /// // Complex transformation - change layer numbering
    /// let remaps = vec![
    ///     (r"layers\.(\d+)", "blocks.$1"),
    /// ];
    /// let (result, remapped) = model.apply_remapped(views, remaps)?;
    /// ```
    #[cfg(target_has_atomic = "ptr")]
    fn apply_remapped<S1, S2>(
        &mut self,
        views: HashMap<String, TensorView>,
        key_remap: Vec<(S1, S2)>,
    ) -> Result<(ImportResult, Vec<(String, String)>), ImportError>
    where
        S1: AsRef<str>,
        S2: AsRef<str>,
    {
        let compiled_remaps = compile_remap_patterns(key_remap)?;
        let (remapped_views, remapped_names) = remap_tensor_paths(views, compiled_remaps);
        let result = self.apply(remapped_views);
        Ok((result, remapped_names))
    }

    /// Apply tensor views with key remapping and filtering.
    ///
    /// Combines remapping and filtering - first remaps the paths, then applies filters
    /// to the remapped paths.
    ///
    /// # Arguments
    ///
    /// * `views` - HashMap of tensor paths to TensorViews
    /// * `key_remap` - Vector of (pattern, replacement) string pairs for path transformation
    /// * `patterns` - Regex patterns to filter which tensors to apply (applied after remapping)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // First rename layer to block, then only apply block 0 and 1
    /// let remaps = vec![
    ///     (r"layer", "block"),
    /// ];
    /// let (result, remapped) = model.apply_remapped_filtered(
    ///     views,
    ///     remaps,
    ///     &[r"^block\.[01]\..*"]
    /// )?;
    /// ```
    #[cfg(target_has_atomic = "ptr")]
    fn apply_remapped_filtered<S1, S2, I, S3>(
        &mut self,
        views: HashMap<String, TensorView>,
        key_remap: Vec<(S1, S2)>,
        patterns: I,
    ) -> Result<(ImportResult, Vec<(String, String)>), ImportError>
    where
        S1: AsRef<str>,
        S2: AsRef<str>,
        I: IntoIterator<Item = S3>,
        S3: AsRef<str>,
    {
        let compiled_remaps = compile_remap_patterns(key_remap)?;
        let (remapped_views, remapped_names) = remap_tensor_paths(views, compiled_remaps);
        let result = self.apply_filtered(remapped_views, patterns)?;
        Ok((result, remapped_names))
    }
}

/// Compile string patterns into regex patterns for remapping.
///
/// # Arguments
///
/// * `patterns` - Vector of (pattern, replacement) string pairs
///
/// # Returns
///
/// * `Ok(Vec<(Regex, String)>)` - Compiled regex patterns with replacements
/// * `Err(ImportError)` - If any pattern fails to compile
#[cfg(target_has_atomic = "ptr")]
fn compile_remap_patterns<S1, S2>(
    patterns: Vec<(S1, S2)>,
) -> Result<Vec<(Regex, String)>, ImportError>
where
    S1: AsRef<str>,
    S2: AsRef<str>,
{
    patterns
        .into_iter()
        .map(|(pattern, replacement)| {
            Regex::new(pattern.as_ref())
                .map(|regex| (regex, replacement.as_ref().to_string()))
                .map_err(ImportError::from)
        })
        .collect()
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
impl<B: Backend, M: Module<B> + Clone> ModulePersist<B> for M {}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate as burn;
    use crate::{
        TestBackend,
        module::{Module, Param},
        nn::{self, LinearConfig},
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
    fn test_collect() {
        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);

        let views = module.collect();

        assert_eq!(views.len(), 4);
        assert!(views.contains_key("encoder.weight"));
        assert!(views.contains_key("encoder.bias"));
        assert!(views.contains_key("decoder.weight"));
        assert!(views.contains_key("decoder.bias"));
    }

    #[test]
    fn test_collect_filtered() {
        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);

        let views = module.collect_filtered(&[r"^encoder\..*"]).unwrap();

        assert_eq!(views.len(), 2);
        assert!(views.contains_key("encoder.weight"));
        assert!(views.contains_key("encoder.bias"));
        assert!(!views.contains_key("decoder.weight"));
        assert!(!views.contains_key("decoder.bias"));
    }

    #[test]
    fn test_collect_filtered_multiple_patterns() {
        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);

        // Collect tensors matching either pattern (OR union)
        let views = module
            .collect_filtered(&[
                r"^encoder\.weight$", // Only encoder.weight
                r".*\.bias$",         // Any .bias tensor
            ])
            .unwrap();

        assert_eq!(views.len(), 3);
        assert!(views.contains_key("encoder.weight")); // matches first pattern
        assert!(views.contains_key("encoder.bias")); // matches second pattern
        assert!(views.contains_key("decoder.bias")); // matches second pattern
        assert!(!views.contains_key("decoder.weight")); // matches neither
    }

    #[test]
    fn test_with_linear_module() {
        let device = Default::default();
        let linear = LinearConfig::new(10, 20)
            .with_bias(true)
            .init::<TestBackend>(&device);

        let views = linear.collect();
        assert!(views.contains_key("weight"));
        assert!(views.contains_key("bias"));

        // Verify the views contain correct data
        let weight_view = views.get("weight").unwrap();
        let weight_data = weight_view.to_data();
        assert_eq!(weight_data.shape, vec![10, 20]);

        let bias_view = views.get("bias").unwrap();
        let bias_data = bias_view.to_data();
        assert_eq!(bias_data.shape, vec![20]);
    }

    // Deep nesting test (5 levels deep)
    #[derive(Module, Debug)]
    struct DeepNestedModule<B: Backend> {
        level1: Level1<B>,
    }

    #[derive(Module, Debug)]
    struct Level1<B: Backend> {
        level2_a: Level2<B>,
        level2_b: Level2<B>,
    }

    #[derive(Module, Debug)]
    struct Level2<B: Backend> {
        level3: Level3<B>,
    }

    #[derive(Module, Debug)]
    struct Level3<B: Backend> {
        level4_main: Level4<B>,
        level4_aux: Level4<B>,
    }

    #[derive(Module, Debug)]
    struct Level4<B: Backend> {
        conv: Param<Tensor<B, 4>>,
        norm: Param<Tensor<B, 1>>,
    }

    impl<B: Backend> DeepNestedModule<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                level1: Level1::new(device),
            }
        }
    }

    impl<B: Backend> Level1<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                level2_a: Level2::new(device),
                level2_b: Level2::new(device),
            }
        }
    }

    impl<B: Backend> Level2<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                level3: Level3::new(device),
            }
        }
    }

    impl<B: Backend> Level3<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                level4_main: Level4::new(device),
                level4_aux: Level4::new(device),
            }
        }
    }

    impl<B: Backend> Level4<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                conv: Param::from_data([[[[1.0]]]], device),
                norm: Param::from_data([0.5], device),
            }
        }
    }

    #[test]
    fn test_deep_nested_collect() {
        let device = Default::default();
        let model = DeepNestedModule::<TestBackend>::new(&device);

        // Test collecting all tensors from deeply nested structure
        let all_views = model.collect();

        // Should have 8 tensors total (2 tensors × 2 level4 modules × 2 level2 branches)
        assert_eq!(all_views.len(), 8);

        // Verify deep paths exist
        assert!(all_views.contains_key("level1.level2_a.level3.level4_main.conv"));
        assert!(all_views.contains_key("level1.level2_a.level3.level4_main.norm"));
        assert!(all_views.contains_key("level1.level2_a.level3.level4_aux.conv"));
        assert!(all_views.contains_key("level1.level2_a.level3.level4_aux.norm"));
        assert!(all_views.contains_key("level1.level2_b.level3.level4_main.conv"));
        assert!(all_views.contains_key("level1.level2_b.level3.level4_main.norm"));
        assert!(all_views.contains_key("level1.level2_b.level3.level4_aux.conv"));
        assert!(all_views.contains_key("level1.level2_b.level3.level4_aux.norm"));
    }

    #[test]
    fn test_deep_nested_filtered_collect() {
        let device = Default::default();
        let model = DeepNestedModule::<TestBackend>::new(&device);

        // Filter only level2_a branch
        let level2_a_views = model.collect_filtered(&[r"^level1\.level2_a\..*"]).unwrap();
        assert_eq!(level2_a_views.len(), 4);

        // Filter only main modules at any depth
        let main_views = model.collect_filtered(&[r".*\.level4_main\..*"]).unwrap();
        assert_eq!(main_views.len(), 4);

        // Filter only conv tensors
        let conv_views = model.collect_filtered(&[r".*\.conv$"]).unwrap();
        assert_eq!(conv_views.len(), 4);

        // Complex multi-pattern filter
        let complex_views = model
            .collect_filtered(&[
                r"^level1\.level2_a\..*\.norm$", // All norms in level2_a
                r"^level1\.level2_b\..*\.conv$", // All convs in level2_b
            ])
            .unwrap();
        assert_eq!(complex_views.len(), 4);
        assert!(complex_views.contains_key("level1.level2_a.level3.level4_main.norm"));
        assert!(complex_views.contains_key("level1.level2_a.level3.level4_aux.norm"));
        assert!(complex_views.contains_key("level1.level2_b.level3.level4_main.conv"));
        assert!(complex_views.contains_key("level1.level2_b.level3.level4_aux.conv"));
    }

    // Custom layer for testing various module configurations
    #[derive(Module, Debug)]
    struct CustomLayer<B: Backend> {
        weight: Param<Tensor<B, 2>>,
        scale: Param<Tensor<B, 1>>,
    }

    impl<B: Backend> CustomLayer<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                weight: Param::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device),
                scale: Param::from_data([0.1, 0.2, 0.3], device),
            }
        }
    }

    #[test]
    fn test_custom_layer_collect() {
        let device = Default::default();
        let custom = CustomLayer::<TestBackend>::new(&device);

        let views = custom.collect();

        assert_eq!(views.len(), 2);
        assert!(views.contains_key("weight"));
        assert!(views.contains_key("scale"));

        // Verify tensor shapes
        let weight_data = views.get("weight").unwrap().to_data();
        assert_eq!(weight_data.shape, vec![2, 3]);

        let scale_data = views.get("scale").unwrap().to_data();
        assert_eq!(scale_data.shape, vec![3]);
    }

    #[test]
    fn test_composite_module_collect() {
        let device = Default::default();

        // Create a composite module with multiple custom layers
        #[derive(Module, Debug)]
        struct CompositeModule<B: Backend> {
            layer1: CustomLayer<B>,
            layer2: CustomLayer<B>,
        }

        impl<B: Backend> CompositeModule<B> {
            fn new(device: &B::Device) -> Self {
                Self {
                    layer1: CustomLayer::new(device),
                    layer2: CustomLayer::new(device),
                }
            }
        }

        let composite = CompositeModule::<TestBackend>::new(&device);
        let views = composite.collect();

        assert_eq!(views.len(), 4);
        assert!(views.contains_key("layer1.weight"));
        assert!(views.contains_key("layer1.scale"));
        assert!(views.contains_key("layer2.weight"));
        assert!(views.contains_key("layer2.scale"));

        // Test filtering
        let weight_only = composite.collect_filtered(&[r".*\.weight$"]).unwrap();
        assert_eq!(weight_only.len(), 2);
        assert!(weight_only.contains_key("layer1.weight"));
        assert!(weight_only.contains_key("layer2.weight"));
    }

    // Test module with Option fields
    #[derive(Module, Debug)]
    struct OptionalFieldModule<B: Backend> {
        required: Param<Tensor<B, 2>>,
        optional: Option<Param<Tensor<B, 1>>>,
    }

    impl<B: Backend> OptionalFieldModule<B> {
        fn new_with_optional(device: &B::Device) -> Self {
            Self {
                required: Param::from_data([[1.0, 2.0], [3.0, 4.0]], device),
                optional: Some(Param::from_data([5.0, 6.0], device)),
            }
        }

        fn new_without_optional(device: &B::Device) -> Self {
            Self {
                required: Param::from_data([[1.0, 2.0], [3.0, 4.0]], device),
                optional: None,
            }
        }
    }

    #[test]
    fn test_optional_field_module_with_value() {
        let device = Default::default();
        let module = OptionalFieldModule::<TestBackend>::new_with_optional(&device);

        let views = module.collect();

        assert_eq!(views.len(), 2);
        assert!(views.contains_key("required"));
        assert!(views.contains_key("optional"));
    }

    #[test]
    fn test_optional_field_module_without_value() {
        let device = Default::default();
        let module = OptionalFieldModule::<TestBackend>::new_without_optional(&device);

        let views = module.collect();

        assert_eq!(views.len(), 1);
        assert!(views.contains_key("required"));
        assert!(!views.contains_key("optional"));
    }

    // Test Vec of modules
    #[derive(Module, Debug)]
    struct VecModule<B: Backend> {
        layers: Vec<CustomLayer<B>>,
    }

    impl<B: Backend> VecModule<B> {
        fn new(device: &B::Device, num_layers: usize) -> Self {
            Self {
                layers: (0..num_layers).map(|_| CustomLayer::new(device)).collect(),
            }
        }
    }

    #[test]
    fn test_vec_module_collect() {
        let device = Default::default();
        let module = VecModule::<TestBackend>::new(&device, 3);

        let views = module.collect();

        // With the fix, all Vec items should now be properly indexed and visited
        assert_eq!(views.len(), 6); // 3 layers × 2 tensors each = 6 tensors

        // Check that all indexed paths exist
        assert!(views.contains_key("layers.0.weight"));
        assert!(views.contains_key("layers.0.scale"));
        assert!(views.contains_key("layers.1.weight"));
        assert!(views.contains_key("layers.1.scale"));
        assert!(views.contains_key("layers.2.weight"));
        assert!(views.contains_key("layers.2.scale"));

        // Verify the data from all tensors
        for i in 0..3 {
            let weight_path = format!("layers.{}.weight", i);
            let scale_path = format!("layers.{}.scale", i);

            let weight_data = views.get(&weight_path).unwrap().to_data();
            assert_eq!(weight_data.shape, vec![2, 3]);

            let scale_data = views.get(&scale_path).unwrap().to_data();
            assert_eq!(scale_data.shape, vec![3]);
        }

        // Test filtering for specific layer
        let layer1_only = module.collect_filtered(&[r"^layers\.1\..*"]).unwrap();
        assert_eq!(layer1_only.len(), 2);
        assert!(layer1_only.contains_key("layers.1.weight"));
        assert!(layer1_only.contains_key("layers.1.scale"));

        // Test filtering for all weights
        let weights_only = module.collect_filtered(&[r".*\.weight$"]).unwrap();
        assert_eq!(weights_only.len(), 3);
        assert!(weights_only.contains_key("layers.0.weight"));
        assert!(weights_only.contains_key("layers.1.weight"));
        assert!(weights_only.contains_key("layers.2.weight"));
    }

    // Test array of modules
    #[derive(Module, Debug)]
    struct ArrayModule<B: Backend> {
        layers: [CustomLayer<B>; 3],
    }

    impl<B: Backend> ArrayModule<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                layers: [
                    CustomLayer::new(device),
                    CustomLayer::new(device),
                    CustomLayer::new(device),
                ],
            }
        }
    }

    #[test]
    fn test_array_module_collect() {
        let device = Default::default();
        let module = ArrayModule::<TestBackend>::new(&device);

        let views = module.collect();

        // All array items should be properly indexed
        assert_eq!(views.len(), 6); // 3 layers × 2 tensors each = 6 tensors

        // Check indexed paths
        for i in 0..3 {
            assert!(views.contains_key(&format!("layers.{}.weight", i)));
            assert!(views.contains_key(&format!("layers.{}.scale", i)));
        }

        // Test filtering for specific index
        let layer2_only = module.collect_filtered(&[r"^layers\.2\..*"]).unwrap();
        assert_eq!(layer2_only.len(), 2);
        assert!(layer2_only.contains_key("layers.2.weight"));
        assert!(layer2_only.contains_key("layers.2.scale"));
    }

    #[test]
    fn test_collect_with_predicate() {
        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);

        // Test with simple predicate - only encoder tensors
        let encoder_only = module.collect_with_predicate(|path| path.starts_with("encoder."));
        assert_eq!(encoder_only.len(), 2);
        assert!(encoder_only.contains_key("encoder.weight"));
        assert!(encoder_only.contains_key("encoder.bias"));

        // Test with specific names predicate
        let specific = module
            .collect_with_predicate(|path| path == "encoder.weight" || path == "decoder.bias");
        assert_eq!(specific.len(), 2);
        assert!(specific.contains_key("encoder.weight"));
        assert!(specific.contains_key("decoder.bias"));

        // Test with complex logic
        let complex = module.collect_with_predicate(|path| {
            let parts: Vec<&str> = path.split('.').collect();
            // Only collect if it's a weight OR if it's in the encoder
            (parts.len() == 2 && parts[1] == "weight") || parts[0] == "encoder"
        });
        assert_eq!(complex.len(), 3);
        assert!(complex.contains_key("encoder.weight"));
        assert!(complex.contains_key("encoder.bias"));
        assert!(complex.contains_key("decoder.weight"));
    }

    #[test]
    fn test_predicate_with_deep_module() {
        let device = Default::default();
        let model = DeepNestedModule::<TestBackend>::new(&device);

        // Filter using predicate - only level2_a branch
        let level2_a_only = model.collect_with_predicate(|path| path.contains("level2_a"));
        assert_eq!(level2_a_only.len(), 4);

        // Filter by depth - only tensors at exactly 5 levels deep
        let deep_only = model.collect_with_predicate(|path| path.split('.').count() == 5);
        assert_eq!(deep_only.len(), 8); // All conv and norm tensors are 5 levels deep

        // Complex predicate with multiple conditions
        let complex = model.collect_with_predicate(|path| {
            let is_main = path.contains("level4_main");
            let is_conv = path.ends_with(".conv");
            is_main && is_conv
        });
        assert_eq!(complex.len(), 2);
        assert!(complex.contains_key("level1.level2_a.level3.level4_main.conv"));
        assert!(complex.contains_key("level1.level2_b.level3.level4_main.conv"));
    }

    #[test]
    fn test_predicate_with_vec_module() {
        let device = Default::default();
        let module = VecModule::<TestBackend>::new(&device, 4);

        // Filter using predicate - only even indices
        let even_only = module.collect_with_predicate(|path| {
            if let Some(captures) = path.split('.').nth(1) {
                if let Ok(index) = captures.parse::<usize>() {
                    return index % 2 == 0;
                }
            }
            false
        });
        assert_eq!(even_only.len(), 4); // layers.0 and layers.2, each with 2 tensors

        // Filter for specific range of indices
        let range = module.collect_with_predicate(|path| {
            if let Some(captures) = path.split('.').nth(1) {
                if let Ok(index) = captures.parse::<usize>() {
                    return index >= 1 && index <= 2;
                }
            }
            false
        });
        assert_eq!(range.len(), 4); // layers.1 and layers.2, each with 2 tensors
    }

    #[test]
    fn test_predicate_with_closure_capturing_state() {
        let device = Default::default();
        let module = TestModule::<TestBackend>::new(&device);

        // Use closure that captures external state
        let allowed_modules = vec!["encoder"];
        let allowed_types = vec!["weight", "bias"];

        let filtered = module.collect_with_predicate(move |path| {
            let parts: Vec<&str> = path.split('.').collect();
            if parts.len() != 2 {
                return false;
            }
            allowed_modules.contains(&parts[0]) && allowed_types.contains(&parts[1])
        });

        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains_key("encoder.weight"));
        assert!(filtered.contains_key("encoder.bias"));
    }

    // Test enum modules - they ARE supported by Burn
    // Note: Burn requires all enum variants to have the same field type
    #[derive(Module, Debug)]
    enum EnumModule<B: Backend> {
        LayerA(CustomLayer<B>),
        LayerB(CustomLayer<B>),
        LayerC(CustomLayer<B>),
    }

    #[test]
    fn test_enum_module_collect() {
        let device = Default::default();

        // Test variant A
        let module_a = EnumModule::<TestBackend>::LayerA(CustomLayer::new(&device));
        let views_a = module_a.collect();

        // Should have the variant name in the path
        assert_eq!(views_a.len(), 2);
        assert!(views_a.contains_key("LayerA.weight"));
        assert!(views_a.contains_key("LayerA.scale"));

        // Test variant B
        let module_b = EnumModule::<TestBackend>::LayerB(CustomLayer::new(&device));
        let views_b = module_b.collect();

        assert_eq!(views_b.len(), 2);
        assert!(views_b.contains_key("LayerB.weight"));
        assert!(views_b.contains_key("LayerB.scale"));

        // Test variant C
        let module_c = EnumModule::<TestBackend>::LayerC(CustomLayer::new(&device));
        let views_c = module_c.collect();

        assert_eq!(views_c.len(), 2);
        assert!(views_c.contains_key("LayerC.weight"));
        assert!(views_c.contains_key("LayerC.scale"));

        // Verify tensor data
        let weight_data = views_a.get("LayerA.weight").unwrap().to_data();
        assert_eq!(weight_data.shape, vec![2, 3]);

        // Test filtering on enum module
        let scale_only = module_a.collect_filtered(&[r".*\.scale$"]).unwrap();
        assert_eq!(scale_only.len(), 1);
        assert!(scale_only.contains_key("LayerA.scale"));
    }

    // Test enum module with built-in module types
    #[derive(Module, Debug)]
    enum NetworkVariant<B: Backend> {
        Small(nn::Linear<B>),
        Medium(nn::Linear<B>),
        Large(nn::Linear<B>),
    }

    #[test]
    fn test_enum_with_linear_modules() {
        let device = Default::default();

        // Create different sized linear layers
        let small_linear = LinearConfig::new(10, 20)
            .with_bias(true)
            .init::<TestBackend>(&device);
        let medium_linear = LinearConfig::new(20, 50)
            .with_bias(true)
            .init::<TestBackend>(&device);
        let large_linear = LinearConfig::new(50, 100)
            .with_bias(true)
            .init::<TestBackend>(&device);

        // Test small variant
        let small_net = NetworkVariant::Small(small_linear);
        let small_views = small_net.collect();

        assert_eq!(small_views.len(), 2);
        assert!(small_views.contains_key("Small.weight"));
        assert!(small_views.contains_key("Small.bias"));

        // Verify shapes
        let weight_data = small_views.get("Small.weight").unwrap().to_data();
        assert_eq!(weight_data.shape, vec![10, 20]);

        // Test medium variant
        let medium_net = NetworkVariant::Medium(medium_linear);
        let medium_views = medium_net.collect();

        assert_eq!(medium_views.len(), 2);
        assert!(medium_views.contains_key("Medium.weight"));
        assert!(medium_views.contains_key("Medium.bias"));

        let weight_data = medium_views.get("Medium.weight").unwrap().to_data();
        assert_eq!(weight_data.shape, vec![20, 50]);

        // Test large variant
        let large_net = NetworkVariant::Large(large_linear);
        let large_views = large_net.collect();

        assert_eq!(large_views.len(), 2);
        assert!(large_views.contains_key("Large.weight"));
        assert!(large_views.contains_key("Large.bias"));

        let weight_data = large_views.get("Large.weight").unwrap().to_data();
        assert_eq!(weight_data.shape, vec![50, 100]);
    }
}
