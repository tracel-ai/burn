use super::GraphNode;
use std::collections::HashSet;

/// Whether executing the nodes in the given order keeps every read resource live.
///
/// The nodes are taken already in execution order — pass a (cloneable, cheap) iterator over the
/// candidate order rather than materializing it; the function walks it twice.
///
/// Simulates resource lifetimes: every resource a node reads must be live when the node
/// executes, a freed resource dies after the freeing node, and produced resources become live.
/// A resource read before any ordered node produced it is external (from an earlier segment)
/// and assumed live — unless some ordered node *does* produce it, in which case the order reads
/// it before its producer and is invalid.
pub fn is_valid_execution_order<N: GraphNode>(order: impl Iterator<Item = N> + Clone) -> bool {
    let mut produced = HashSet::new();
    for node in order.clone() {
        produced.extend(node.produced());
    }

    let mut alive = HashSet::new();
    let mut dead = HashSet::new();
    for node in order {
        for resource in node.read() {
            if dead.contains(&resource) {
                return false; // Read after the resource was freed: bad order.
            }
            if !alive.contains(&resource) {
                if produced.contains(&resource) {
                    return false; // Produced in this segment but not live yet: bad order.
                }
                alive.insert(resource); // External resource from a prior segment.
            }
        }
        for resource in node.freed() {
            alive.remove(&resource);
            dead.insert(resource);
        }
        for resource in node.produced() {
            // Producing redefines the resource, even if an id were ever reused after a free.
            dead.remove(&resource);
            alive.insert(resource);
        }
    }

    true
}
