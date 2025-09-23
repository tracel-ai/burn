use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

#[cfg(feature = "std")]
use regex::Regex;

/// A sophisticated path filter that supports multiple matching strategies.
///
/// The filter uses an OR logic - a path is included if it matches ANY of the configured criteria.
/// This allows for flexible and powerful filtering configurations.
///
/// # Examples
///
/// ```rust,ignore
/// // Create a filter that matches encoder paths or any weight path
/// let filter = PathFilter::new()
///     .with_regex(r"^encoder\..*")
///     .with_regex(r".*\.weight$")
///     .with_full_path("special_tensor");
///
/// // Check if a path should be included
/// if filter.matches("encoder.layer1.weight") {
///     // This will match due to both regex patterns
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct PathFilter {
    /// Compiled regex patterns for matching paths
    #[cfg(feature = "std")]
    regex_patterns: Vec<Regex>,

    /// Exact full paths to match
    exact_paths: Vec<String>,

    /// Predicate functions for custom matching logic based on path and container path
    /// Note: These cannot be cloned, so we store them separately
    predicates: Vec<fn(&str, &str) -> bool>,

    /// If true, matches all paths (overrides other filters)
    match_all: bool,
}

impl PathFilter {
    /// Create a new empty filter (matches nothing by default)
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a filter that matches all paths
    pub fn all() -> Self {
        Self {
            match_all: true,
            ..Default::default()
        }
    }

    /// Create a filter that matches nothing
    pub fn none() -> Self {
        Self::default()
    }

    /// Add a regex pattern for matching paths
    #[cfg(feature = "std")]
    pub fn with_regex<S: AsRef<str>>(mut self, pattern: S) -> Self {
        if let Ok(regex) = Regex::new(pattern.as_ref()) {
            self.regex_patterns.push(regex);
        }
        // TODO: Consider returning Result to handle regex compilation errors
        self
    }

    /// Add multiple regex patterns
    #[cfg(feature = "std")]
    pub fn with_regexes<I, S>(mut self, patterns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for pattern in patterns {
            if let Ok(regex) = Regex::new(pattern.as_ref()) {
                self.regex_patterns.push(regex);
            }
        }
        self
    }

    /// Add an exact full path to match
    pub fn with_full_path<S: Into<String>>(mut self, path: S) -> Self {
        self.exact_paths.push(path.into());
        self
    }

    /// Add multiple exact full paths
    pub fn with_full_paths<I, S>(mut self, paths: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.exact_paths.extend(paths.into_iter().map(|p| p.into()));
        self
    }

    /// Add a predicate function for custom matching based on path and container path
    pub fn with_predicate(mut self, predicate: fn(&str, &str) -> bool) -> Self {
        self.predicates.push(predicate);
        self
    }

    /// Add multiple predicates
    pub fn with_predicates<I>(mut self, predicates: I) -> Self
    where
        I: IntoIterator<Item = fn(&str, &str) -> bool>,
    {
        self.predicates.extend(predicates);
        self
    }

    /// Set to match all paths
    pub fn match_all(mut self) -> Self {
        self.match_all = true;
        self
    }

    /// Check if a path matches this filter (assumes empty container path for backward compatibility)
    pub fn matches(&self, path: &str) -> bool {
        self.matches_with_container_path_str(path, "")
    }

    /// Check if a path and container type match this filter (for backward compatibility)
    pub fn matches_with_container(&self, path: &str, container_type: &str) -> bool {
        // For backward compatibility, treat single container type as the full path
        self.matches_with_container_path_str(path, container_type)
    }

    /// Check if a path and container path match this filter
    pub fn matches_with_container_path(&self, path: &[String], container_stack: &[String]) -> bool {
        let path_str = path.join(".");
        let container_path = container_stack.join(".");
        self.matches_with_container_path_str(&path_str, &container_path)
    }

    /// Check if a path and container path (dot-notated strings) match this filter
    pub fn matches_with_container_path_str(&self, path: &str, container_path: &str) -> bool {
        // If match_all is set, always return true
        if self.match_all {
            return true;
        }

        // If no filters are configured, match nothing
        if self.is_empty() {
            return false;
        }

        // Check exact path matches
        if self.exact_paths.iter().any(|p| p == path) {
            return true;
        }

        // Check regex patterns (on the path)
        #[cfg(feature = "std")]
        {
            for regex in &self.regex_patterns {
                if regex.is_match(path) {
                    return true;
                }
            }
        }

        // Check predicates with container path
        if self
            .predicates
            .iter()
            .any(|pred| pred(path, container_path))
        {
            return true;
        }

        false
    }

    /// Check if the filter is empty (matches nothing)
    pub fn is_empty(&self) -> bool {
        if self.match_all {
            return false;
        }

        #[cfg(feature = "std")]
        let regex_empty = self.regex_patterns.is_empty();
        #[cfg(not(feature = "std"))]
        let regex_empty = true;

        self.exact_paths.is_empty() && self.predicates.is_empty() && regex_empty
    }

    /// Get the number of filter criteria configured
    pub fn criteria_count(&self) -> usize {
        if self.match_all {
            return 1;
        }

        #[allow(unused_mut)]
        let mut count = self.exact_paths.len() + self.predicates.len();

        #[cfg(feature = "std")]
        {
            count += self.regex_patterns.len();
        }

        count
    }

    /// Clear all regex patterns
    #[cfg(feature = "std")]
    pub fn clear_regex(&mut self) -> &mut Self {
        self.regex_patterns.clear();
        self
    }

    /// Clear all exact paths
    pub fn clear_paths(&mut self) -> &mut Self {
        self.exact_paths.clear();
        self
    }

    /// Clear all predicates
    pub fn clear_predicates(&mut self) -> &mut Self {
        self.predicates.clear();
        self
    }

    /// Clear all filters
    pub fn clear(&mut self) -> &mut Self {
        #[cfg(feature = "std")]
        self.clear_regex();

        self.clear_paths().clear_predicates();
        self.match_all = false;
        self
    }

    /// Create a filter from regex patterns only
    #[cfg(feature = "std")]
    pub fn from_regex_patterns<I, S>(patterns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self::new().with_regexes(patterns)
    }

    /// Create a filter from exact paths only
    pub fn from_paths<I, S>(paths: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::new().with_full_paths(paths)
    }

    /// Create a filter from a single predicate
    pub fn from_predicate(predicate: fn(&str, &str) -> bool) -> Self {
        Self::new().with_predicate(predicate)
    }

    /// Combine with another filter using OR logic
    pub fn or(mut self, other: Self) -> Self {
        if self.match_all || other.match_all {
            return Self::all();
        }

        #[cfg(feature = "std")]
        {
            self.regex_patterns.extend(other.regex_patterns);
        }

        self.exact_paths.extend(other.exact_paths);
        self.predicates.extend(other.predicates);

        self
    }
}

impl fmt::Display for PathFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.match_all {
            return write!(f, "PathFilter::all()");
        }

        if self.is_empty() {
            return write!(f, "PathFilter::none()");
        }

        write!(f, "PathFilter[")?;

        let mut parts = Vec::new();

        #[cfg(feature = "std")]
        if !self.regex_patterns.is_empty() {
            parts.push(format!("regex: {:?}", self.regex_patterns));
        }

        if !self.exact_paths.is_empty() {
            parts.push(format!("paths: {:?}", self.exact_paths));
        }

        if !self.predicates.is_empty() {
            parts.push(format!("predicates: {}", self.predicates.len()));
        }

        write!(f, "{}]", parts.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_filter() {
        let filter = PathFilter::new();
        assert!(filter.is_empty());
        assert!(!filter.matches("encoder.weight"));
        assert!(!filter.matches("decoder.bias"));
    }

    #[test]
    fn match_all() {
        let filter = PathFilter::all();
        assert!(!filter.is_empty());
        assert!(filter.matches("encoder.weight"));
        assert!(filter.matches("decoder.bias"));
        assert!(filter.matches("anything"));
    }

    #[test]
    fn exact_paths() {
        let filter = PathFilter::new()
            .with_full_path("encoder.weight")
            .with_full_path("decoder.bias");

        assert!(filter.matches("encoder.weight"));
        assert!(filter.matches("decoder.bias"));
        assert!(!filter.matches("encoder.bias"));
        assert!(!filter.matches("decoder.weight"));
    }

    #[test]
    #[cfg(feature = "std")]
    fn regex_patterns() {
        let filter = PathFilter::new()
            .with_regex(r"^encoder\..*")
            .with_regex(r".*\.weight$");

        assert!(filter.matches("encoder.layer1.bias"));
        assert!(filter.matches("decoder.weight"));
        assert!(filter.matches("encoder.weight"));
        assert!(!filter.matches("decoder.bias"));
    }

    #[test]
    fn predicates() {
        fn contains_norm(path: &str, _container_path: &str) -> bool {
            path.contains("norm")
        }

        fn is_short(path: &str, _container_path: &str) -> bool {
            path.len() < 10
        }

        let filter = PathFilter::new()
            .with_predicate(contains_norm)
            .with_predicate(is_short);

        assert!(filter.matches("norm.weight"));
        assert!(filter.matches("layer.norm.bias"));
        assert!(filter.matches("bias"));
        assert!(!filter.matches("encoder.decoder.weight.long.name"));
    }

    #[test]
    fn combined_filters() {
        let filter = PathFilter::new()
            .with_full_path("special.tensor")
            .with_predicate(|path, _container_path| path.contains("attention"));

        #[cfg(feature = "std")]
        let filter = filter.with_regex(r"^encoder\..*");

        assert!(filter.matches("special.tensor"));
        assert!(filter.matches("self_attention.query"));

        #[cfg(feature = "std")]
        assert!(filter.matches("encoder.anything"));

        assert!(!filter.matches("decoder.weight"));
    }

    #[test]
    fn or_combination() {
        let encoder_filter = PathFilter::new().with_full_path("encoder.weight");
        let decoder_filter = PathFilter::new().with_full_path("decoder.bias");

        let combined = encoder_filter.or(decoder_filter);

        assert!(combined.matches("encoder.weight"));
        assert!(combined.matches("decoder.bias"));
        assert!(!combined.matches("model.head.weight"));
    }

    #[test]
    #[cfg(feature = "std")]
    fn common_patterns() {
        // Test encoder pattern
        let encoder = PathFilter::new().with_regex(r"^encoder\..*");
        assert!(encoder.matches("encoder.weight"));
        assert!(!encoder.matches("decoder.weight"));

        // Test weights-only pattern
        let weights = PathFilter::new().with_regex(r".*\.weight$");
        assert!(weights.matches("encoder.weight"));
        assert!(weights.matches("decoder.weight"));
        assert!(!weights.matches("encoder.bias"));

        // Test layer-specific patterns
        let layers = PathFilter::new()
            .with_regex(r"(^|.*\.)layers\.0\.")
            .with_regex(r"(^|.*\.)layers\.2\.")
            .with_regex(r"(^|.*\.)layers\.4\.");
        assert!(layers.matches("model.layers.0.weight"));
        assert!(layers.matches("layers.2.bias"));
        assert!(!layers.matches("layers.1.weight"));
    }

    #[test]
    fn criteria_count() {
        let filter = PathFilter::new()
            .with_full_path("path1")
            .with_full_path("path2")
            .with_predicate(|_, _| true);

        #[cfg(feature = "std")]
        let filter = filter.with_regex(".*");

        #[cfg(feature = "std")]
        assert_eq!(filter.criteria_count(), 4);

        #[cfg(not(feature = "std"))]
        assert_eq!(filter.criteria_count(), 3);
    }

    #[test]
    fn clear_operations() {
        let mut filter = PathFilter::new().with_full_path("test");

        filter.clear_paths();
        assert!(!filter.matches("test"));

        filter.clear();
        assert!(filter.is_empty());
    }

    #[test]
    fn container_predicates() {
        // Filter that matches only Linear module weights
        let linear_weights = PathFilter::new().with_predicate(|path, container_path| {
            container_path.split('.').next_back() == Some("Linear") && path.ends_with(".weight")
        });

        assert!(linear_weights.matches_with_container("layer1.weight", "Linear"));
        assert!(!linear_weights.matches_with_container("layer1.weight", "Conv2d"));
        assert!(!linear_weights.matches_with_container("layer1.bias", "Linear"));

        // Filter for specific container types
        let conv_only = PathFilter::new().with_predicate(|_path, container_path| {
            let last = container_path.split('.').next_back();
            last == Some("Conv2d") || last == Some("ConvTranspose2d")
        });

        assert!(conv_only.matches_with_container("encoder.weight", "Conv2d"));
        assert!(conv_only.matches_with_container("decoder.weight", "ConvTranspose2d"));
        assert!(!conv_only.matches_with_container("fc.weight", "Linear"));

        // Combine path and container predicates
        let combined = PathFilter::new()
            .with_predicate(|path, _container_path| path.starts_with("encoder."))
            .with_predicate(|_path, container_path| {
                container_path.split('.').next_back() == Some("BatchNorm2d")
            });

        // Should match either condition (OR logic)
        assert!(combined.matches_with_container("encoder.layer1", "Linear"));
        assert!(combined.matches_with_container("decoder.bn", "BatchNorm2d"));
        assert!(!combined.matches_with_container("decoder.layer", "Linear"));
    }

    #[test]
    fn container_predicate_with_regex() {
        // Combine regex patterns with container predicates
        #[cfg(feature = "std")]
        {
            let filter = PathFilter::new()
                .with_regex(r"^encoder\..*")
                .with_predicate(|path, container_path| {
                    container_path.split('.').next_back() == Some("Linear")
                        && path.contains(".bias")
                });

            // Matches due to regex
            assert!(filter.matches_with_container("encoder.layer1.weight", "Conv2d"));
            // Matches due to container predicate
            assert!(filter.matches_with_container("decoder.fc.bias", "Linear"));
            // Doesn't match either
            assert!(!filter.matches_with_container("decoder.conv.weight", "Conv2d"));
        }
    }

    #[test]
    fn container_stack_predicates() {
        // Filter using full container path - only tensors nested in a specific hierarchy
        let nested_filter = PathFilter::new().with_predicate(|_path, container_path| {
            // Check if tensor is nested within: Model -> TransformerBlock -> Linear
            let parts: Vec<&str> = container_path.split('.').collect();
            parts.len() >= 3
                && parts[0] == "Model"
                && parts[1] == "TransformerBlock"
                && parts[2] == "Linear"
        });

        assert!(nested_filter.matches_with_container_path_str(
            "encoder.weight",
            "Model.TransformerBlock.Linear.Param"
        ));
        assert!(
            !nested_filter
                .matches_with_container_path_str("decoder.weight", "Model.Decoder.Linear.Param")
        );
        assert!(!nested_filter.matches_with_container_path_str(
            "encoder.weight",
            "Model.TransformerBlock.Conv2d.Param"
        ));

        // Filter that checks for specific depth in hierarchy
        let depth_filter = PathFilter::new().with_predicate(|_path, container_path| {
            let parts: Vec<&str> = container_path.split('.').collect();
            parts.len() == 4 && parts.get(2) == Some(&"Linear")
        });

        assert!(depth_filter.matches_with_container_path_str(
            "model.layer.weight",
            "Model.TransformerBlock.Linear.Param"
        ));
        assert!(
            !depth_filter
                .matches_with_container_path_str("model.weight", "Model.TransformerBlock.Conv2d")
        ); // Too shallow

        // Filter that checks any Linear in the path (not just the last)
        let any_linear = PathFilter::new()
            .with_predicate(|_path, container_path| container_path.contains("Linear"));

        assert!(
            any_linear.matches_with_container_path_str(
                "some.path",
                "Model.TransformerBlock.Linear.Param"
            )
        );
        assert!(
            any_linear.matches_with_container_path_str("other.path", "Model.Decoder.Linear.Param")
        );
        assert!(
            !any_linear.matches_with_container_path_str(
                "conv.path",
                "Model.TransformerBlock.Conv2d.Param"
            )
        );
    }

    #[test]
    fn container_path_dot_notation() {
        // Filter using dot-notated container path
        let dot_filter = PathFilter::new().with_predicate(|_path, container_path| {
            container_path.starts_with("Model.TransformerBlock")
        });

        // Test with matches_with_container_path
        assert!(
            dot_filter.matches_with_container_path_str("weight", "Model.TransformerBlock.Linear")
        );
        assert!(!dot_filter.matches_with_container_path_str("weight", "Model.Decoder.Linear"));

        // Filter that checks for specific patterns in container path
        let pattern_filter = PathFilter::new().with_predicate(|_path, container_path| {
            // Match any path that has Linear after a block
            container_path.contains("Block.Linear") || container_path.contains("Block.Conv")
        });

        assert!(
            pattern_filter
                .matches_with_container_path_str("weight", "Model.TransformerBlock.Linear")
        );
        assert!(pattern_filter.matches_with_container_path_str("weight", "Model.ResBlock.Conv2d"));
        assert!(!pattern_filter.matches_with_container_path_str("weight", "Model.Linear.Param"));

        // Filter combining path and container path patterns
        let combined = PathFilter::new().with_predicate(|path, container_path| {
            // Only weights in Linear layers that are inside blocks
            path.ends_with(".weight")
                && container_path.contains("Block")
                && container_path.split('.').next_back() == Some("Linear")
        });

        assert!(
            combined
                .matches_with_container_path_str("layer.weight", "Model.TransformerBlock.Linear")
        );
        assert!(
            !combined
                .matches_with_container_path_str("layer.bias", "Model.TransformerBlock.Linear")
        );
        assert!(!combined.matches_with_container_path_str("layer.weight", "Model.Decoder.Linear"));
    }
}
