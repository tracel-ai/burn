//! The components of a breakdown whose work ran in a kernel recorded elsewhere.

use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use super::Component;

/// The components of one breakdown whose work all ran in a fused kernel recorded elsewhere:
/// what [`Breakdown::fused_elsewhere`](super::Breakdown::fused_elsewhere) finds.
#[derive(Clone, Debug, Default)]
pub struct FusedElsewhere<'a> {
    components: HashSet<ByAddress<'a>>,
}

impl FusedElsewhere<'_> {
    /// Whether all of `component`'s work ran in a fused kernel recorded elsewhere. A
    /// component of another breakdown never is.
    pub fn contains(&self, component: &Component) -> bool {
        self.components.contains(&ByAddress(component))
    }
}

impl<'a> FromIterator<&'a Component> for FusedElsewhere<'a> {
    fn from_iter<I: IntoIterator<Item = &'a Component>>(components: I) -> Self {
        Self {
            components: components.into_iter().map(ByAddress).collect(),
        }
    }
}

/// A component, equal only to itself.
///
/// By address rather than by path: every caller holds a component borrowed from the
/// breakdown, and none of them its path. A tree that kept its collector's component indices
/// would make both unnecessary.
#[derive(Clone, Copy, Debug)]
struct ByAddress<'a>(&'a Component);

impl PartialEq for ByAddress<'_> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.0, other.0)
    }
}

impl Eq for ByAddress<'_> {}

impl Hash for ByAddress<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::ptr::hash(self.0, state);
    }
}
