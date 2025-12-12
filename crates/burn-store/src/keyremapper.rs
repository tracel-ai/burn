use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use regex::{self, Regex};

use crate::TensorSnapshot;

/// Key remapper for transforming tensor names.
///
/// This allows mapping tensor names from one naming convention to another,
/// which is useful for loading models from different frameworks or versions.
///
/// # Examples
///
/// ```rust
/// # use burn_store::KeyRemapper;
/// // Create a key remapper
/// let remapper = KeyRemapper::new()
///     .add_pattern(r"^pytorch\.(.*)", "burn.$1").expect("valid regex")  // pytorch.layer -> burn.layer
///     .add_pattern(r"\.gamma$", ".weight").expect("valid regex");       // layer.gamma -> layer.weight
///
/// // Use remapper with stores
/// // store.remap(remapper)
/// ```
#[derive(Debug, Clone, Default)]
pub struct KeyRemapper {
    /// Pattern-based remapping rules (regex pattern, replacement string)
    pub patterns: Vec<(Regex, String)>,
}

impl KeyRemapper {
    /// Create a new empty key remapper
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a remapping pattern (compiles regex)
    ///
    /// # Arguments
    ///
    /// * `from` - Source pattern (regex string)
    /// * `to` - Replacement string (can include capture groups like `$1`)
    ///
    /// # Returns
    ///
    /// * `Ok(Self)` - Updated remapping configuration
    /// * `Err(regex::Error)` - If regex compilation fails
    pub fn add_pattern<S1, S2>(mut self, from: S1, to: S2) -> Result<Self, regex::Error>
    where
        S1: AsRef<str>,
        S2: Into<String>,
    {
        let regex = Regex::new(from.as_ref())?;
        self.patterns.push((regex, to.into()));
        Ok(self)
    }

    /// Create from a list of compiled regex patterns
    pub fn from_compiled_patterns(patterns: Vec<(Regex, String)>) -> Self {
        Self { patterns }
    }

    /// Create from string patterns (will compile to regex)
    ///
    /// # Arguments
    ///
    /// * `patterns` - Vector of (pattern, replacement) tuples
    ///
    /// # Returns
    ///
    /// * `Ok(Self)` - New remapping configuration
    /// * `Err(regex::Error)` - If any regex compilation fails
    pub fn from_patterns<S1, S2>(patterns: Vec<(S1, S2)>) -> Result<Self, regex::Error>
    where
        S1: AsRef<str>,
        S2: Into<String>,
    {
        let mut compiled_patterns = Vec::new();
        for (pattern, replacement) in patterns {
            let regex = Regex::new(pattern.as_ref())?;
            compiled_patterns.push((regex, replacement.into()));
        }
        Ok(Self {
            patterns: compiled_patterns,
        })
    }

    /// Create from an iterator of patterns
    ///
    /// # Arguments
    ///
    /// * `iter` - Iterator yielding (pattern, replacement) tuples
    ///
    /// # Returns
    ///
    /// * `Ok(Self)` - New remapping configuration
    /// * `Err(regex::Error)` - If any regex compilation fails
    pub fn from_pattern_iter<I, S1, S2>(iter: I) -> Result<Self, regex::Error>
    where
        I: IntoIterator<Item = (S1, S2)>,
        S1: AsRef<str>,
        S2: Into<String>,
    {
        let patterns: Result<Vec<_>, _> = iter
            .into_iter()
            .map(|(from, to)| Ok((Regex::new(from.as_ref())?, to.into())))
            .collect();
        Ok(Self {
            patterns: patterns?,
        })
    }

    /// Check if the remapping is empty
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Convert to the format expected by remap_tensor_paths_with_patterns
    pub fn to_regex_pairs(&self) -> Vec<(Regex, String)> {
        self.patterns.clone()
    }

    /// Remap tensor paths using the configured patterns.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Vec of TensorSnapshots to remap
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * The remapped Vec of TensorSnapshots with updated paths
    /// * A vector of (new_path, original_path) showing the transformations
    pub fn remap(
        &self,
        mut tensors: Vec<TensorSnapshot>,
    ) -> (Vec<TensorSnapshot>, Vec<(String, String)>) {
        if self.patterns.is_empty() {
            let remapped_names = tensors
                .iter()
                .map(|v| {
                    let path = v.full_path();
                    (path.clone(), path)
                })
                .collect();
            return (tensors, remapped_names);
        }

        let mut remapped_snapshots = Vec::new();
        let mut remapped_names = Vec::new();

        for mut snapshot in tensors.drain(..) {
            let original_path = snapshot.full_path();
            let mut new_path = original_path.clone();

            // Apply all patterns to get the new path
            for (pattern, replacement) in &self.patterns {
                if pattern.is_match(&new_path) {
                    new_path = pattern
                        .replace_all(&new_path, replacement.as_str())
                        .to_string();
                }
            }

            // Update the snapshot's internal path_stack if the path changed
            if new_path != original_path
                && let Some(ref mut path_stack) = snapshot.path_stack
            {
                *path_stack = new_path.split('.').map(|s| s.to_string()).collect();
            }

            remapped_names.push((new_path.clone(), original_path));
            remapped_snapshots.push(snapshot);
        }

        (remapped_snapshots, remapped_names)
    }
}

/// Map tensor paths to have contiguous numeric indices.
///
/// This function detects numeric indices in tensor paths and renumbers them
/// to be contiguous (0, 1, 2, ...) while preserving their relative order.
/// It handles nested sequential structures by processing ALL numeric indices
/// in each path independently based on their position context.
///
/// This is useful when loading PyTorch models that have gaps in layer numbering,
/// such as when using `nn.Sequential` with mixed layer types (e.g., Conv2d + ReLU
/// where only Conv2d has parameters).
///
/// # Example
///
/// Simple case - input paths:
/// - `fc.0.weight`, `fc.0.bias`
/// - `fc.2.weight`, `fc.2.bias`
/// - `fc.4.weight`, `fc.4.bias`
///
/// Output paths:
/// - `fc.0.weight`, `fc.0.bias`
/// - `fc.1.weight`, `fc.1.bias`
/// - `fc.2.weight`, `fc.2.bias`
///
/// Nested case - input paths:
/// - `feature.layers.0.conv_block.0.weight`
/// - `feature.layers.0.conv_block.2.weight`
/// - `feature.layers.2.conv_block.0.weight`
/// - `feature.layers.2.conv_block.2.weight`
///
/// Output paths:
/// - `feature.layers.0.conv_block.0.weight`
/// - `feature.layers.0.conv_block.1.weight`
/// - `feature.layers.1.conv_block.0.weight`
/// - `feature.layers.1.conv_block.1.weight`
///
/// # Arguments
///
/// * `tensors` - Vec of TensorSnapshots to map
///
/// # Returns
///
/// A tuple containing:
/// * The mapped Vec of TensorSnapshots with updated paths
/// * A vector of (new_path, original_path) showing the transformations
pub fn map_indices_contiguous(
    mut tensors: Vec<TensorSnapshot>,
) -> (Vec<TensorSnapshot>, Vec<(String, String)>) {
    if tensors.is_empty() {
        return (tensors, Vec::new());
    }

    // Step 1: Collect all paths and find all index positions
    // For each index position (identified by prefix using ORIGINAL indices),
    // collect all indices seen at that position.
    //
    // Key: prefix using original path (e.g., "feature.layers." or "feature.layers.0.conv_block.")
    // Value: BTreeMap of original_index -> new_index
    let mut index_maps: BTreeMap<String, BTreeMap<usize, usize>> = BTreeMap::new();

    // First pass: collect all indices at each position using original prefixes
    for snapshot in &tensors {
        let path = snapshot.full_path();
        let parts: Vec<&str> = path.split('.').collect();

        // Check each part for numeric indices
        for (i, part) in parts.iter().enumerate() {
            if let Ok(index) = part.parse::<usize>() {
                // The prefix is everything before this index (using original path)
                let prefix = if i > 0 {
                    format!("{}.", parts[..i].join("."))
                } else {
                    String::new()
                };

                index_maps
                    .entry(prefix)
                    .or_default()
                    .entry(index)
                    .or_insert(usize::MAX); // Placeholder
            }
        }
    }

    // Second pass: assign contiguous indices for each position
    for indices in index_maps.values_mut() {
        let mut sorted_indices: Vec<usize> = indices.keys().cloned().collect();
        sorted_indices.sort();

        for (new_idx, old_idx) in sorted_indices.into_iter().enumerate() {
            indices.insert(old_idx, new_idx);
        }
    }

    // Third pass: apply the remapping to all tensors
    // We use original prefixes for lookup since that's how we collected indices
    let mut mapped_snapshots = Vec::new();
    let mut transformations = Vec::new();

    for mut snapshot in tensors.drain(..) {
        let original_path = snapshot.full_path();
        let new_path = remap_all_indices_with_original_prefix(&original_path, &index_maps);

        // Update the snapshot's internal path_stack if the path changed
        if new_path != original_path
            && let Some(ref mut path_stack) = snapshot.path_stack
        {
            *path_stack = new_path.split('.').map(|s| s.to_string()).collect();
        }

        transformations.push((new_path, original_path));
        mapped_snapshots.push(snapshot);
    }

    (mapped_snapshots, transformations)
}

/// Remap all numeric indices in a path using the provided index maps.
/// Uses original path prefixes for lookup.
fn remap_all_indices_with_original_prefix(
    path: &str,
    index_maps: &BTreeMap<String, BTreeMap<usize, usize>>,
) -> String {
    let parts: Vec<&str> = path.split('.').collect();
    let mut result_parts: Vec<String> = Vec::with_capacity(parts.len());

    for (i, part) in parts.iter().enumerate() {
        if let Ok(index) = part.parse::<usize>() {
            // Build the prefix from ORIGINAL parts (not remapped)
            let prefix = if i > 0 {
                format!("{}.", parts[..i].join("."))
            } else {
                String::new()
            };

            // Look up the new index using original prefix
            if let Some(index_map) = index_maps.get(&prefix)
                && let Some(&new_index) = index_map.get(&index)
            {
                result_parts.push(new_index.to_string());
                continue;
            }
        }
        // Not a numeric index or no mapping found, keep as-is
        result_parts.push((*part).to_string());
    }

    result_parts.join(".")
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use burn_core::module::ParamId;
    use burn_tensor::TensorData;

    fn create_test_tensor_snapshot(name: &str) -> TensorSnapshot {
        let data = TensorData {
            bytes: burn_tensor::Bytes::from_bytes_vec(vec![1, 2, 3, 4]),
            shape: vec![2, 2],
            dtype: burn_tensor::DType::F32,
        };
        let path_parts: Vec<String> = name.split('.').map(|s| s.to_string()).collect();
        TensorSnapshot::from_data(data, path_parts, vec!["Test".to_string()], ParamId::new())
    }

    #[test]
    fn test_key_remapper_basic() {
        let remapper = KeyRemapper::new()
            .add_pattern(r"^encoder\.", "transformer.encoder.")
            .expect("valid regex");

        let tensors = vec![
            create_test_tensor_snapshot("encoder.layer1.weight"),
            create_test_tensor_snapshot("decoder.layer1.weight"),
        ];

        let (remapped, transformations) = remapper.remap(tensors);

        // Check that remapped views exist with correct paths
        assert!(
            remapped
                .iter()
                .any(|v| v.full_path() == "transformer.encoder.layer1.weight")
        );
        assert!(
            remapped
                .iter()
                .any(|v| v.full_path() == "decoder.layer1.weight")
        );
        assert_eq!(remapped.len(), 2);

        // Check transformations
        let encoder_transform = transformations
            .iter()
            .find(|(_new, old)| old == "encoder.layer1.weight")
            .expect("should find encoder transformation");
        assert_eq!(encoder_transform.0, "transformer.encoder.layer1.weight");
    }

    #[test]
    fn test_key_remapper_multiple_patterns() {
        let remapper = KeyRemapper::new()
            .add_pattern(r"^encoder\.", "transformer.encoder.")
            .expect("valid regex")
            .add_pattern(r"\.gamma$", ".weight")
            .expect("valid regex");

        let tensors = vec![create_test_tensor_snapshot("encoder.layer1.gamma")];

        let (remapped, _) = remapper.remap(tensors);

        assert!(
            remapped
                .iter()
                .any(|v| v.full_path() == "transformer.encoder.layer1.weight")
        );
        assert_eq!(remapped.len(), 1);
    }

    #[test]
    fn test_key_remapper_from_patterns() {
        let patterns = vec![(r"^pytorch\.", "burn."), (r"\.bias$", ".bias_param")];
        let remapper = KeyRemapper::from_patterns(patterns).expect("valid patterns");

        let tensors = vec![create_test_tensor_snapshot("pytorch.linear.bias")];

        let (remapped, _) = remapper.remap(tensors);

        assert!(
            remapped
                .iter()
                .any(|v| v.full_path() == "burn.linear.bias_param")
        );
    }

    #[test]
    fn test_key_remapper_empty() {
        let remapper = KeyRemapper::new();
        assert!(remapper.is_empty());

        let tensors = vec![create_test_tensor_snapshot("test.weight")];

        let (remapped, transformations) = remapper.remap(tensors);

        assert!(remapped.iter().any(|v| v.full_path() == "test.weight"));
        assert_eq!(remapped.len(), 1);
        assert_eq!(transformations.len(), 1);
        assert_eq!(
            transformations[0],
            ("test.weight".to_string(), "test.weight".to_string())
        );
    }

    #[test]
    fn test_map_indices_contiguous_basic() {
        // Simulate PyTorch nn.Sequential with Conv2d (0, 2, 4) and ReLU (1, 3, 5)
        // Only Conv2d layers have parameters
        let tensors = vec![
            create_test_tensor_snapshot("fc.0.weight"),
            create_test_tensor_snapshot("fc.0.bias"),
            create_test_tensor_snapshot("fc.2.weight"),
            create_test_tensor_snapshot("fc.2.bias"),
            create_test_tensor_snapshot("fc.4.weight"),
            create_test_tensor_snapshot("fc.4.bias"),
        ];

        let (reindexed, transformations) = map_indices_contiguous(tensors);

        // Check that indices are now contiguous
        assert!(reindexed.iter().any(|v| v.full_path() == "fc.0.weight"));
        assert!(reindexed.iter().any(|v| v.full_path() == "fc.0.bias"));
        assert!(reindexed.iter().any(|v| v.full_path() == "fc.1.weight"));
        assert!(reindexed.iter().any(|v| v.full_path() == "fc.1.bias"));
        assert!(reindexed.iter().any(|v| v.full_path() == "fc.2.weight"));
        assert!(reindexed.iter().any(|v| v.full_path() == "fc.2.bias"));
        assert_eq!(reindexed.len(), 6);

        // Check transformations
        let transform_2_to_1 = transformations
            .iter()
            .find(|(_, old)| old == "fc.2.weight")
            .expect("should find fc.2.weight transformation");
        assert_eq!(transform_2_to_1.0, "fc.1.weight");

        let transform_4_to_2 = transformations
            .iter()
            .find(|(_, old)| old == "fc.4.weight")
            .expect("should find fc.4.weight transformation");
        assert_eq!(transform_4_to_2.0, "fc.2.weight");
    }

    #[test]
    fn test_map_indices_contiguous_already_contiguous() {
        // Already contiguous indices should remain unchanged
        let tensors = vec![
            create_test_tensor_snapshot("fc.0.weight"),
            create_test_tensor_snapshot("fc.1.weight"),
            create_test_tensor_snapshot("fc.2.weight"),
        ];

        let (reindexed, transformations) = map_indices_contiguous(tensors);

        assert!(reindexed.iter().any(|v| v.full_path() == "fc.0.weight"));
        assert!(reindexed.iter().any(|v| v.full_path() == "fc.1.weight"));
        assert!(reindexed.iter().any(|v| v.full_path() == "fc.2.weight"));
        assert_eq!(reindexed.len(), 3);

        // All transformations should have same old and new paths
        for (new, old) in &transformations {
            assert_eq!(new, old);
        }
    }

    #[test]
    fn test_map_indices_contiguous_multiple_prefixes() {
        // Different prefixes should be mapped independently
        let tensors = vec![
            create_test_tensor_snapshot("encoder.0.weight"),
            create_test_tensor_snapshot("encoder.2.weight"),
            create_test_tensor_snapshot("decoder.1.weight"),
            create_test_tensor_snapshot("decoder.5.weight"),
        ];

        let (reindexed, _) = map_indices_contiguous(tensors);

        // encoder: 0, 2 -> 0, 1
        assert!(
            reindexed
                .iter()
                .any(|v| v.full_path() == "encoder.0.weight")
        );
        assert!(
            reindexed
                .iter()
                .any(|v| v.full_path() == "encoder.1.weight")
        );

        // decoder: 1, 5 -> 0, 1
        assert!(
            reindexed
                .iter()
                .any(|v| v.full_path() == "decoder.0.weight")
        );
        assert!(
            reindexed
                .iter()
                .any(|v| v.full_path() == "decoder.1.weight")
        );
    }

    #[test]
    fn test_map_indices_contiguous_no_indices() {
        // Paths without indices should remain unchanged
        let tensors = vec![
            create_test_tensor_snapshot("encoder.weight"),
            create_test_tensor_snapshot("decoder.bias"),
        ];

        let (reindexed, transformations) = map_indices_contiguous(tensors);

        assert!(reindexed.iter().any(|v| v.full_path() == "encoder.weight"));
        assert!(reindexed.iter().any(|v| v.full_path() == "decoder.bias"));

        for (new, old) in &transformations {
            assert_eq!(new, old);
        }
    }

    #[test]
    fn test_map_indices_contiguous_empty() {
        let tensors: Vec<TensorSnapshot> = vec![];
        let (reindexed, transformations) = map_indices_contiguous(tensors);

        assert!(reindexed.is_empty());
        assert!(transformations.is_empty());
    }

    #[test]
    fn test_map_indices_contiguous_mixed_indexed_and_non_indexed() {
        // Mix of indexed and non-indexed paths
        let tensors = vec![
            create_test_tensor_snapshot("fc.0.weight"),
            create_test_tensor_snapshot("fc.2.weight"),
            create_test_tensor_snapshot("output.weight"), // no index
        ];

        let (reindexed, _) = map_indices_contiguous(tensors);

        assert!(reindexed.iter().any(|v| v.full_path() == "fc.0.weight"));
        assert!(reindexed.iter().any(|v| v.full_path() == "fc.1.weight")); // 2 -> 1
        assert!(reindexed.iter().any(|v| v.full_path() == "output.weight")); // unchanged
    }

    #[test]
    fn test_map_indices_contiguous_nested_sequential() {
        // Test nested sequential structures like:
        // feature = nn.Sequential(ConvBlock, ReLU, ConvBlock, ReLU, ConvBlock)
        // where ConvBlock = nn.Sequential(Conv2d, ReLU, Conv2d)
        //
        // This produces paths like:
        // feature.layers.0.conv_block.0.weight (layer 0, conv 0)
        // feature.layers.0.conv_block.2.weight (layer 0, conv 2 - skipping ReLU at 1)
        // feature.layers.2.conv_block.0.weight (layer 2 - skipping ReLU at 1, conv 0)
        // feature.layers.2.conv_block.2.weight (layer 2, conv 2)
        let tensors = vec![
            create_test_tensor_snapshot("feature.layers.0.conv_block.0.weight"),
            create_test_tensor_snapshot("feature.layers.0.conv_block.2.weight"),
            create_test_tensor_snapshot("feature.layers.2.conv_block.0.weight"),
            create_test_tensor_snapshot("feature.layers.2.conv_block.2.weight"),
        ];

        let (mapped, transformations) = map_indices_contiguous(tensors);

        // Expected mapping:
        // feature.layers: 0, 2 -> 0, 1
        // feature.layers.0.conv_block: 0, 2 -> 0, 1
        // feature.layers.2.conv_block: 0, 2 -> 0, 1
        //
        // Result:
        // feature.layers.0.conv_block.0.weight -> feature.layers.0.conv_block.0.weight
        // feature.layers.0.conv_block.2.weight -> feature.layers.0.conv_block.1.weight
        // feature.layers.2.conv_block.0.weight -> feature.layers.1.conv_block.0.weight
        // feature.layers.2.conv_block.2.weight -> feature.layers.1.conv_block.1.weight

        assert!(
            mapped
                .iter()
                .any(|v| v.full_path() == "feature.layers.0.conv_block.0.weight"),
            "0.0 should stay as 0.0"
        );
        assert!(
            mapped
                .iter()
                .any(|v| v.full_path() == "feature.layers.0.conv_block.1.weight"),
            "0.2 should become 0.1"
        );
        assert!(
            mapped
                .iter()
                .any(|v| v.full_path() == "feature.layers.1.conv_block.0.weight"),
            "2.0 should become 1.0"
        );
        assert!(
            mapped
                .iter()
                .any(|v| v.full_path() == "feature.layers.1.conv_block.1.weight"),
            "2.2 should become 1.1"
        );

        // Verify specific transformations
        let t1 = transformations
            .iter()
            .find(|(_, old)| old == "feature.layers.2.conv_block.2.weight");
        assert_eq!(
            t1.map(|(new, _)| new.as_str()),
            Some("feature.layers.1.conv_block.1.weight"),
            "2.2 should map to 1.1"
        );
    }

    #[test]
    fn test_map_indices_contiguous_deeply_nested() {
        // Test with three levels of nesting
        let tensors = vec![
            create_test_tensor_snapshot("a.0.b.0.c.0.weight"),
            create_test_tensor_snapshot("a.0.b.0.c.2.weight"),
            create_test_tensor_snapshot("a.0.b.2.c.0.weight"),
            create_test_tensor_snapshot("a.2.b.0.c.0.weight"),
        ];

        let (mapped, _) = map_indices_contiguous(tensors);

        // a: 0, 2 -> 0, 1
        // a.0.b: 0, 2 -> 0, 1
        // a.2.b: 0 -> 0
        // a.0.b.0.c: 0, 2 -> 0, 1
        // a.0.b.2.c: 0 -> 0
        // a.2.b.0.c: 0 -> 0

        assert!(mapped.iter().any(|v| v.full_path() == "a.0.b.0.c.0.weight"));
        assert!(
            mapped.iter().any(|v| v.full_path() == "a.0.b.0.c.1.weight"),
            "a.0.b.0.c.2 should become a.0.b.0.c.1"
        );
        assert!(
            mapped.iter().any(|v| v.full_path() == "a.0.b.1.c.0.weight"),
            "a.0.b.2.c.0 should become a.0.b.1.c.0"
        );
        assert!(
            mapped.iter().any(|v| v.full_path() == "a.1.b.0.c.0.weight"),
            "a.2.b.0.c.0 should become a.1.b.0.c.0"
        );
    }
}
