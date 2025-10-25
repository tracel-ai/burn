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
/// ```rust,no_run
/// # use burn_store::KeyRemapper;
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a key remapper
/// let remapper = KeyRemapper::new()
///     .add_pattern(r"^pytorch\.(.*)", "burn.$1")?  // pytorch.layer -> burn.layer
///     .add_pattern(r"\.gamma$", ".weight")?;       // layer.gamma -> layer.weight
///
/// // Use remapper with stores
/// // store.remap(remapper)
/// # Ok(())
/// # }
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
}
