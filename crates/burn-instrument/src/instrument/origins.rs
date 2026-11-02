//! Which component each operation the fusion server registered came from, and
//! so whose work a block of them carries out.
//!
//! Sealing order says which component a segment ran *during*, not whose work
//! it was. A block runs when some later operation seals its segment, and it can
//! carry out work several components registered — so a norm fused into the
//! projection after it is neither the norm's kernel nor the projection's, and
//! their common parent is the one component it belongs to whole. An operation that
//! ran unfused is a block of one, and goes back to the component that
//! registered it however late it ran.

use std::collections::HashMap;

use burn_ir::OperationIr;

use super::component_tree::{ComponentId, ComponentTree};
use super::cursor::{Attribution, Placement};

/// An operation the fusion server registered, known by its first output: fresh
/// for every operation the server registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OperationId(u64);

impl OperationId {
    /// The id of `operation`, or `None` for a `Drop` and for the few
    /// operations that write no tensor.
    ///
    /// A `Drop` is left out: it is registered by whoever let the tensor go,
    /// which says nothing about the work, and fused into a block it would widen
    /// it to a parent that did none of it.
    pub fn new(operation: &OperationIr) -> Option<Self> {
        match operation {
            OperationIr::Drop(_) => None,
            operation => operation
                .outputs()
                .next()
                .map(|output| Self(output.id.value())),
        }
    }
}

/// The component each operation came from.
#[derive(Default)]
pub struct OperationOrigins(HashMap<OperationId, Attribution>);

impl OperationOrigins {
    /// `operation` was registered under `origin`.
    pub fn record(&mut self, operation: OperationId, origin: Attribution) {
        self.0.insert(operation, origin);
    }

    /// Forget every operation: once the run is over, none is left to map
    /// back.
    pub fn clear(&mut self) {
        self.0.clear();
    }
}

/// Whose work a block carries out, read off the operations it runs.
#[derive(Debug, PartialEq, Eq)]
pub enum BlockOrigin {
    /// None of its operations was seen registering: they were queued before
    /// the run.
    Unseen,
    /// Work these components registered, in first-seen order, and nothing
    /// else.
    Within(Vec<ComponentId>),
    /// Work from outside every component, beside these components' work.
    WithOutside(Vec<ComponentId>),
}

impl BlockOrigin {
    /// What the block running `operations` carries out.
    pub fn new(origins: &OperationOrigins, operations: impl Iterator<Item = OperationId>) -> Self {
        let mut components: Vec<ComponentId> = Vec::new();
        let mut outside = false;
        let mut seen = false;
        for operation in operations {
            match origins.0.get(&operation) {
                Some(Attribution::Component(component)) => {
                    seen = true;
                    if !components.contains(component) {
                        components.push(*component);
                    }
                }
                Some(Attribution::Untagged | Attribution::EndOfRun) => {
                    seen = true;
                    outside = true;
                }
                None => {}
            }
        }
        match (seen, outside) {
            (false, _) => BlockOrigin::Unseen,
            (true, true) => BlockOrigin::WithOutside(components),
            (true, false) => BlockOrigin::Within(components),
        }
    }

    /// Where the block's launches land in `tree`: the closest component
    /// containing all of its work, named for the components it came from
    /// where there are several. `None` keeps the sealing order's answer.
    pub fn placement(self, tree: &ComponentTree) -> Option<Placement> {
        match self {
            BlockOrigin::Unseen => None,
            // Still named for the components whose work it carried: recorded
            // outside them, it is the only place their time went.
            BlockOrigin::WithOutside(span) => Some(Placement {
                attribution: Attribution::Untagged,
                span,
            }),
            BlockOrigin::Within(components) => Some(Placement {
                attribution: Self::closest_component(&components, tree),
                span: Self::named(components),
            }),
        }
    }

    /// The closest component holding all of `components`' work. Components under
    /// different roots — a forward and a backward fused across the step that
    /// joins them — share no component, so the kernel is recorded outside every one
    /// of them.
    fn closest_component(components: &[ComponentId], tree: &ComponentTree) -> Attribution {
        match tree.common_parent(components) {
            Some(common) => Attribution::Component(common),
            None => Attribution::Untagged,
        }
    }

    /// The components a kernel is named for: none for one component's work,
    /// which is that component's by placement.
    fn named(components: Vec<ComponentId>) -> Vec<ComponentId> {
        match components.len() {
            1 => Vec::new(),
            _ => components,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Declaration;

    /// A block that carried out several components' work lands on their
    /// closest common parent, named for them — the one component that holds all of
    /// it, however late the block ran.
    #[test]
    fn a_fused_block_lands_on_the_closest_parent_of_its_components() {
        let mut components = ComponentTree::default();
        let block = components.enter(declared("block"));
        let norm = components.enter(declared("attn norm"));
        components.exit();
        components.enter(declared("attention"));
        let query = components.enter(declared("query"));
        let origins = origins([(1, norm), (2, query)]);

        assert_eq!(
            BlockOrigin::new(&origins, [1, 2].map(OperationId).into_iter()).placement(&components),
            Some(Placement {
                attribution: block,
                span: vec![component(norm), component(query)],
            })
        );
    }

    /// An operation that ran unfused is a block of one, and goes back to the
    /// component that registered it however late something sealed it — the
    /// case sealing order alone got wrong.
    #[test]
    fn an_unfused_operation_goes_back_to_its_component() {
        let mut components = ComponentTree::default();
        let norm = components.enter(declared("final norm"));
        components.exit();
        components.enter(declared("lm head"));
        let origins = origins([(7, norm)]);

        assert_eq!(
            BlockOrigin::new(&origins, [7].map(OperationId).into_iter()).placement(&components),
            Some(Placement::at(norm)),
            "one component's work, so it names none"
        );
    }

    /// A block with work from outside every component is recorded outside
    /// them, and still names the components it carried.
    #[test]
    fn a_block_with_outside_work_is_recorded_outside_and_named() {
        let mut components = ComponentTree::default();
        let inner = components.enter(declared("inner"));
        let origins = origins([(1, Attribution::Untagged), (2, inner)]);

        assert_eq!(
            BlockOrigin::new(&origins, [1, 2].map(OperationId).into_iter()).placement(&components),
            Some(Placement {
                attribution: Attribution::Untagged,
                span: vec![component(inner)],
            })
        );
    }

    /// Components under different roots share no component — a forward and a
    /// backward, fused across the step that joins them — so a kernel that did
    /// both is recorded outside every component, named for them. It runs on
    /// the fusion worker, where a panic would take the device down with it.
    #[test]
    fn a_block_across_two_roots_is_recorded_outside_them() {
        let mut components = ComponentTree::default();
        let forward = components.enter(declared("transformer"));
        components.exit();
        let backward = components.enter(declared("backward"));
        let origins = origins([(1, forward), (2, backward)]);

        assert_eq!(
            BlockOrigin::new(&origins, [1, 2].map(OperationId).into_iter()).placement(&components),
            Some(Placement {
                attribution: Attribution::Untagged,
                span: vec![component(forward), component(backward)],
            })
        );
    }

    /// A block whose operations were queued before the run says nothing
    /// about whose work it is, and keeps the sealing order's answer.
    #[test]
    fn a_block_of_operations_never_seen_registering_keeps_the_sealing_order() {
        let components = ComponentTree::default();
        let origins = OperationOrigins::default();

        assert_eq!(
            BlockOrigin::new(&origins, [3].map(OperationId).into_iter()),
            BlockOrigin::Unseen
        );
        assert_eq!(
            BlockOrigin::Unseen.placement(&components),
            None,
            "no placement of its own"
        );
    }

    fn origins<const N: usize>(registered: [(u64, Attribution); N]) -> OperationOrigins {
        let mut origins = OperationOrigins::default();
        for (operation, origin) in registered {
            origins.record(OperationId(operation), origin);
        }
        origins
    }

    fn component(entered: Attribution) -> ComponentId {
        match entered {
            Attribution::Component(component) => component,
            Attribution::Untagged | Attribution::EndOfRun => {
                unreachable!("entering a component lands on it")
            }
        }
    }

    fn declared(label: &str) -> Declaration<'_> {
        Declaration {
            label,
            variant: None,
            site: "test::forward",
            bounds: None,
        }
    }
}
