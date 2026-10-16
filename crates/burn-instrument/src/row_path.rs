//! Where a row sits in a breakdown, named by the labels leading to it.

use std::fmt::{self, Display};

/// The labels from one row down to another, outermost first:
/// `attention › query`.
///
/// What tells two rows apart when a label alone does not: a mixed stack's
/// dense and sparse `down` are both `down`, and a row is merged on its label
/// only among its siblings. A kernel names the rows whose work it carried by
/// their path from the row it is recorded on, and a comparison keys a row of
/// one run against the other by its path from the root.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RowPath(Vec<String>);

impl RowPath {
    /// What separates the labels when a path is printed.
    const SEPARATOR: &str = " › ";

    /// The path through `labels`, outermost first.
    pub fn new<Label: Into<String>>(labels: impl IntoIterator<Item = Label>) -> Self {
        Self(labels.into_iter().map(Into::into).collect())
    }

    /// The path to the row `label` names, one level below this one.
    pub fn child(&self, label: &str) -> Self {
        let mut labels = self.0.clone();
        labels.push(label.to_string());
        Self(labels)
    }

    /// The labels, outermost first.
    pub fn labels(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(String::as_str)
    }
}

/// Through [`Formatter::pad`](fmt::Formatter::pad), so a path in a table
/// column takes the column's width.
impl Display for RowPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(&self.0.join(Self::SEPARATOR))
    }
}
