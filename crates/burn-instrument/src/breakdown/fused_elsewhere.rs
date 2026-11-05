//! The components of a breakdown whose work ran in a kernel recorded elsewhere.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use super::Component;

/// The components of one breakdown whose work ran in a fused kernel recorded elsewhere, each
/// with how much of it did: what [`Breakdown::fused_elsewhere`](super::Breakdown::fused_elsewhere)
/// finds.
#[derive(Clone, Debug, Default)]
pub struct FusedElsewhere<'a> {
    components: HashMap<ByAddress<'a>, FusedShare>,
}

/// How much of a component's work ran in a fused kernel recorded elsewhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusedShare {
    /// All of it: the component ran no kernel of its own, so its floor is the kernel's that
    /// carried its work.
    All,
    /// Part of it: the component ran kernels of its own besides — a layer of a stack whose
    /// neighbours fused and it did not, say. Its time is only the part it ran, and its floor
    /// prices the whole, so the two are not a score.
    Part,
}

impl FusedElsewhere<'_> {
    /// How much of `component`'s work ran in a fused kernel recorded elsewhere, if any did. A
    /// component of another breakdown has none.
    pub fn share(&self, component: &Component) -> Option<FusedShare> {
        self.components.get(&ByAddress(component)).copied()
    }
}

impl<'a> FromIterator<(&'a Component, FusedShare)> for FusedElsewhere<'a> {
    fn from_iter<I: IntoIterator<Item = (&'a Component, FusedShare)>>(components: I) -> Self {
        Self {
            components: components
                .into_iter()
                .map(|(component, share)| (ByAddress(component), share))
                .collect(),
        }
    }
}

/// A component, equal only to itself.
///
/// By address rather than by path: every caller holds a component borrowed from the
/// breakdown, and none of them its path. A tree that kept its instrument's component indices
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
