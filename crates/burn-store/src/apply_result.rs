//! Result types and diagnostics for tensor application operations

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use burn_tensor::{DType, Shape};

/// Error types that can occur during tensor application
#[derive(Debug, Clone)]
pub enum ApplyError {
    /// Shape mismatch between expected and actual tensor
    ShapeMismatch {
        /// Path of the tensor
        path: String,
        /// Expected shape
        expected: Shape,
        /// Found shape
        found: Shape,
    },
    /// Data type mismatch between expected and actual tensor
    DTypeMismatch {
        /// Path of the tensor
        path: String,
        /// Expected data type
        expected: DType,
        /// Found data type
        found: DType,
    },
    /// Error from adapter transformation
    AdapterError {
        /// Path of the tensor
        path: String,
        /// Error message
        message: String,
    },
    /// Error loading tensor data
    LoadError {
        /// Path of the tensor
        path: String,
        /// Error message
        message: String,
    },
}

impl core::fmt::Display for ApplyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ShapeMismatch {
                path,
                expected,
                found,
            } => {
                write!(
                    f,
                    "Shape mismatch for '{}': expected {:?}, found {:?}",
                    path, expected, found
                )
            }
            Self::DTypeMismatch {
                path,
                expected,
                found,
            } => {
                write!(
                    f,
                    "DType mismatch for '{}': expected {:?}, found {:?}",
                    path, expected, found
                )
            }
            Self::AdapterError { path, message } => {
                write!(f, "Adapter error for '{}': {}", path, message)
            }
            Self::LoadError { path, message } => {
                write!(f, "Load error for '{}': {}", path, message)
            }
        }
    }
}

impl core::error::Error for ApplyError {}

/// Result of applying tensor snapshots to a module
#[derive(Clone)]
pub struct ApplyResult {
    /// Successfully applied tensor paths
    pub applied: Vec<String>,
    /// Skipped tensor paths (due to filter)
    pub skipped: Vec<String>,
    /// Missing tensor paths with their container stacks in dot notation (path, container_stack)
    /// Container stack shows the hierarchy: "Struct:Model.Struct:Linear" or "Struct:Model.Enum:ConvType.Struct:Linear"
    pub missing: Vec<(String, String)>,
    /// Unused tensor paths (in snapshots but not in module)
    pub unused: Vec<String>,
    /// Errors encountered during application
    pub errors: Vec<ApplyError>,
}

impl ApplyResult {
    /// Try to strip enum variant from a path
    /// e.g., "field.BaseConv.weight" -> "field.weight"
    fn strip_enum_variant(path: &str) -> Option<String> {
        let segments: Vec<&str> = path.split('.').collect();

        // Find segments that look like enum variants (CamelCase in middle of path)
        let variant_indices: Vec<usize> = segments
            .iter()
            .enumerate()
            .filter(|(i, segment)| {
                *i > 0 && *i < segments.len() - 1 // Not first or last
                    && !segment.is_empty()
                    && segment.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                    && segment.len() > 1
                    && segment.chars().skip(1).any(|c| c.is_lowercase())
            })
            .map(|(i, _)| i)
            .collect();

        if variant_indices.is_empty() {
            return None;
        }

        // Remove the first found variant and return the modified path
        let mut result_segments = segments.clone();
        result_segments.remove(variant_indices[0]);
        Some(result_segments.join("."))
    }

    /// Find similar paths for a given missing path (for "Did you mean?" suggestions)
    fn find_similar_paths(&self, missing_path: &str, max_suggestions: usize) -> Vec<String> {
        // First, try exact match with enum variant stripped
        if let Some(stripped) = Self::strip_enum_variant(missing_path)
            && self.unused.contains(&stripped)
        {
            return vec![stripped];
        }

        // Fall back to Jaro similarity (used by Elixir for "did you mean?" suggestions)
        // Jaro gives higher weight to matching prefixes, ideal for hierarchical tensor paths
        let mut similarities: Vec<(String, f64)> = self
            .unused
            .iter()
            .map(|available| {
                let similarity = textdistance::nstr::jaro(missing_path, available);
                (available.clone(), similarity)
            })
            .collect();

        // Sort by similarity (higher = more similar)
        similarities
            .sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));

        // Only suggest paths with >= 70% similarity
        const SIMILARITY_THRESHOLD: f64 = 0.7;
        similarities
            .into_iter()
            .filter(|(_, sim)| *sim >= SIMILARITY_THRESHOLD)
            .take(max_suggestions)
            .map(|(path, _)| path)
            .collect()
    }
}

impl ApplyResult {
    /// Check if the apply operation was successful (no errors)
    /// Note: Missing tensors are not considered errors when allow_partial is true
    pub fn is_success(&self) -> bool {
        self.errors.is_empty()
    }
}

impl core::fmt::Debug for ApplyResult {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Delegate to Display for comprehensive output
        core::fmt::Display::fmt(self, f)
    }
}

impl core::fmt::Display for ApplyResult {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "┌─ Tensor Loading Summary ─────────────────────────")?;
        writeln!(f, "│")?;
        writeln!(
            f,
            "│ ✓ Successfully applied: {} tensors",
            self.applied.len()
        )?;
        writeln!(f, "│ ⊘ Skipped (filtered):  {} tensors", self.skipped.len())?;
        writeln!(
            f,
            "│ ✗ Missing in source:    {} tensors",
            self.missing.len()
        )?;
        writeln!(f, "│ ? Unused in target:     {} tensors", self.unused.len())?;
        writeln!(f, "│ ! Errors:               {} errors", self.errors.len())?;

        if !self.missing.is_empty() {
            writeln!(f, "│")?;
            writeln!(
                f,
                "├─ Missing Tensors (requested by model but not found in source)"
            )?;
            writeln!(f, "│")?;

            // Use actual container stack data to detect enum variants
            // Count how many missing paths have "Enum:" in their container stack
            let enum_variant_missing: Vec<_> = self
                .missing
                .iter()
                .filter(|(_, stack)| stack.contains("Enum:"))
                .collect();

            if !enum_variant_missing.is_empty() {
                writeln!(
                    f,
                    "│  ⚠️  {} paths contain enum variants (detected from container stack)",
                    enum_variant_missing.len()
                )?;
                writeln!(
                    f,
                    "│      Burn includes enum variant names in paths, but PyTorch doesn't."
                )?;
                writeln!(
                    f,
                    "│      Example: Burn has 'field.BaseConv.weight', PyTorch has 'field.weight'"
                )?;
                writeln!(f, "│")?;
                writeln!(
                    f,
                    "│      💡 Solution 1: Enable skip_enum_variants flag (simplest):"
                )?;
                writeln!(f, "│")?;
                writeln!(
                    f,
                    "│         let mut store = PytorchStore::from_file(\"model.pth\")"
                )?;
                writeln!(f, "│             .skip_enum_variants(true);  // ← Add this")?;
                writeln!(f, "│")?;
                writeln!(
                    f,
                    "│      💡 Solution 2: Remap enum keys in source (most precise):"
                )?;
                writeln!(f, "│")?;
                writeln!(
                    f,
                    "│         let mut store = SafetensorsStore::from_file(\"model.safetensors\")"
                )?;
                writeln!(
                    f,
                    "│             .with_key_remapping(r\"field\\.(\\w+)\", \"field.BaseConv.$1\");"
                )?;
                writeln!(f, "│")?;
            }

            writeln!(f, "│  First 10 missing tensors:")?;
            for (path, _) in self.missing.iter().take(10) {
                writeln!(f, "│    • {}", path)?;

                // Show "Did you mean?" suggestions for this path
                let suggestions = self.find_similar_paths(path, 1);
                if !suggestions.is_empty() {
                    writeln!(f, "│        Did you mean: '{}'?", suggestions[0])?;
                }
            }
            if self.missing.len() > 10 {
                writeln!(f, "│    ... and {} more", self.missing.len() - 10)?;
            }
        }

        if !self.unused.is_empty() {
            writeln!(f, "│")?;
            writeln!(f, "├─ Unused Tensors (in source but not used by model)")?;
            writeln!(f, "│")?;
            writeln!(f, "│  First 10 unused tensors:")?;
            for path in self.unused.iter().take(10) {
                writeln!(f, "│    • {}", path)?;
            }
            if self.unused.len() > 10 {
                writeln!(f, "│    ... and {} more", self.unused.len() - 10)?;
            }
        }

        if !self.errors.is_empty() {
            writeln!(f, "│")?;
            writeln!(f, "├─ Errors")?;
            writeln!(f, "│")?;
            for error in self.errors.iter().take(10) {
                writeln!(f, "│  ⚠️  {}", error)?;
            }
            if self.errors.len() > 10 {
                writeln!(f, "│    ... and {} more", self.errors.len() - 10)?;
            }
        }

        writeln!(f, "│")?;
        write!(f, "└───────────────────────────────────────────────────")?;

        Ok(())
    }
}
