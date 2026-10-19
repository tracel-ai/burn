//! Where a component sits in a breakdown, named by the labels leading to it.

use std::fmt::{self, Display};

use serde::{Deserialize, Serialize};

/// The labels from one component down to another, outermost first:
/// `attention › query`.
///
/// What tells two components apart when a label alone does not: a mixed stack's
/// dense and sparse `down` are both `down`, and a component is merged on its label
/// only among its siblings. A kernel names the components whose work it carried by
/// their path from the component it is recorded on, and a comparison keys a component of
/// one run against the other by its path from the root.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ComponentPath(Vec<String>);

impl ComponentPath {
    /// What separates the labels when a path is printed.
    const SEPARATOR: &str = " › ";

    /// The path through `labels`, outermost first.
    pub fn new<Label: Into<String>>(labels: impl IntoIterator<Item = Label>) -> Self {
        Self(labels.into_iter().map(Into::into).collect())
    }

    /// The path to the component `label` names, one level below this one.
    pub fn child(&self, label: &str) -> Self {
        let mut labels = self.0.clone();
        labels.push(label.to_string());
        Self(labels)
    }

    /// Whether this path leads through the component at `outer`, or ends there.
    pub fn starts_with(&self, outer: &ComponentPath) -> bool {
        self.0.starts_with(&outer.0)
    }

    /// The labels, outermost first.
    pub fn labels(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(String::as_str)
    }
}

/// Through [`Formatter::pad`](fmt::Formatter::pad), so a path in a table
/// column takes the column's width.
impl Display for ComponentPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(&self.0.join(Self::SEPARATOR))
    }
}
