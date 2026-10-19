//! Which component each operation the fusion server registered came from, and
//! so whose work a block of them carries out.
//!
//! Sealing order says which component a segment ran *during*, not whose work
//! it was. A block runs when some later operation seals its segment, and it can
//! carry out work several components registered — so a norm fused into the
//! projection after it is neither the norm's kernel nor the projection's, and
//! their common parent is the one row it belongs to whole. An operation that
//! ran unfused is a block of one, and goes back to the component that
//! registered it however late it ran.

use std::collections::HashMap;

use super::cursor::{Attribution, Placement};
use super::row_tree::RowTree;

/// The component each operation came from, by its first output's id.
#[derive(Default)]
pub(crate) struct OperationOrigins(HashMap<u64, Attribution>);

impl OperationOrigins {
    /// `operation` was registered under `origin`.
    pub(crate) fn record(&mut self, operation: u64, origin: Attribution) {
        self.0.insert(operation, origin);
    }

    /// Forget every operation: once the pass has run, none is left to map
    /// back.
    pub(crate) fn clear(&mut self) {
        self.0.clear();
    }
}

/// Whose work a block carries out, read off the operations it runs.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum BlockWork {
    /// None of its operations was seen registering: they were queued before
    /// the pass.
    Unseen,
    /// Work these components registered, in first-seen order, and nothing
    /// else.
    Within(Vec<usize>),
    /// Work from outside every component, beside these components' work.
    WithOutside(Vec<usize>),
}

impl BlockWork {
    /// What the block running `operations` carries out.
    pub(crate) fn new(origins: &OperationOrigins, operations: impl Iterator<Item = u64>) -> Self {
        let mut components: Vec<usize> = Vec::new();
        let mut outside = false;
        let mut seen = false;
        for operation in operations {
            match origins.0.get(&operation) {
                Some(Attribution::Component(row)) => {
                    seen = true;
                    if !components.contains(row) {
                        components.push(*row);
                    }
                }
                Some(Attribution::Untagged | Attribution::EndOfPass) => {
                    seen = true;
                    outside = true;
                }
                None => {}
            }
        }
        match (seen, outside) {
            (false, _) => BlockWork::Unseen,
            (true, true) => BlockWork::WithOutside(components),
            (true, false) => BlockWork::Within(components),
        }
    }

    /// Where the block's launches land among `rows`: the closest component
    /// containing all of its work, named for the components it came from
    /// where there are several. `None` keeps the sealing order's answer.
    pub(crate) fn placement(self, rows: &RowTree) -> Option<Placement> {
        match self {
            BlockWork::Unseen => None,
            // Still named for the components whose work it carried: recorded
            // outside them, it is the only place their time went.
            BlockWork::WithOutside(span) => Some(Placement {
                attribution: Attribution::Untagged,
                span,
            }),
            BlockWork::Within(components) => Some(Self::within(components, rows)),
        }
    }

    fn within(components: Vec<usize>, rows: &RowTree) -> Placement {
        let common = rows.common_parent(&components);
        let span = match components.len() {
            1 => Vec::new(),
            _ => components,
        };
        match common {
            Some(common) => Placement {
                attribution: Attribution::Component(common),
                span,
            },
            // Components under different roots — a forward and a backward
            // fused across the step that joins them — share no row, so the
            // kernel is recorded outside every one of them, still named for
            // what it did.
            None => Placement {
                attribution: Attribution::Untagged,
                span,
            },
        }
    }
}
