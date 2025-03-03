use alloc::vec::Vec;

use crate::graph::NodeRef;
/// Duplicate the given object for each node that requires gradients.
///
/// # Notes
///
/// This is useful since you don't have to keep N cloned references alive event if just 1 node
/// will be updated.
///
/// If the object is a tensor and if one reference exists, it can be updated inplace.
pub fn duplicate<T: Clone + core::fmt::Debug, const N: usize>(
    nodes: &[Option<NodeRef>; N],
    obj: Option<T>,
) -> [Option<T>; N] {
    nodes
        .iter()
        .map(|node| match node {
            Some(_) => obj.clone(),
            None => None,
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}
