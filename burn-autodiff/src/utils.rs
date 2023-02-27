use crate::ops::OpsNode;

/// Duplicate the given object for each node that requires gradients.
///
/// # Notes
///
/// This is usefull since you don't have to keep N cloned references alive event if just 1 node
/// will be updated.
///
/// If the object is a tensor and if one reference exist to a tensor at one time, it can be
/// updated inplace.
pub fn duplicate<T: Clone, const N: usize>(nodes: [&OpsNode<(), 0>; N], obj: T) -> [Option<T>; N] {
    nodes.map(|node| match node {
        OpsNode::Node(_, _) => Some(obj.clone()),
        OpsNode::Untrack => None,
    })
}
