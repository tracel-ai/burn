use crate::ops::OpsNode;

/// Duplicate the given object for each node that requires gradients.
///
/// # Notes
///
/// This is usefull since you don't have to keep N cloned references alive event if just 1 node
/// will be updated.
///
/// If the object is a tensor and if one reference exists, it can be updated inplace.
pub fn duplicate<T: Clone + std::fmt::Debug, const N: usize>(
    nodes: &[OpsNode<(), 0>; N],
    obj: Option<T>,
) -> [Option<T>; N] {
    nodes
        .iter()
        .map(|node| match node {
            OpsNode::Tracked(_, _) => obj.clone(),
            OpsNode::Untrack => None,
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}
