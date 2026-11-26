/// A borrowed field path relative to one parameter's optimizer state.
/// Field names follow the record schema, independent of Rust component type names.
/// Parents distinguish repeated instances of a component without allocating strings.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatePath<'a> {
    /// The parameter state itself.
    Root,
    /// An entry in a vector-valued record field.
    Index {
        /// Path of the vector.
        parent: &'a StatePath<'a>,
        /// Zero-based position within it.
        index: usize,
    },
    /// A named field within its parent state.
    Field {
        /// Enclosing state path.
        parent: &'a StatePath<'a>,
        /// Name used by the state record.
        name: &'static str,
    },
}

impl<'a> StatePath<'a> {
    /// Append a record field without allocating.
    pub fn field(&'a self, name: &'static str) -> Self {
        Self::Field { parent: self, name }
    }

    /// Append a vector index without allocating.
    pub fn index(&'a self, index: usize) -> Self {
        Self::Index {
            parent: self,
            index,
        }
    }

    /// Compare a path made entirely of named fields, supplied from root to leaf.
    /// Use equality with [`Self::index`] for paths through vectors.
    pub fn matches(&self, fields: &[&str]) -> bool {
        match (self, fields.split_last()) {
            (Self::Root, None) => true,
            (Self::Field { parent, name }, Some((last, rest))) => {
                name == last && parent.matches(rest)
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn repeated_components_have_distinct_record_paths() {
        let root = StatePath::Root;
        let left = root.field("left");
        let right = root.field("right");
        assert!(left.field("moment").matches(&["left", "moment"]));
        assert!(!right.field("moment").matches(&["left", "moment"]));
        assert!(!left.field("moment").matches(&["moment"]));
        assert!(root.matches(&[]));
    }
}
