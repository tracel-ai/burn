use crate::graph::NodeRef;

/// Duplicate the given object for each node that requires gradients.
///
/// # Notes
///
/// This is useful since you don't have to keep N cloned references alive event if just 1 node
/// will be updated.
///
/// If the object is a tensor and if one reference exists, it can be updated inplace.
pub fn duplicate<T: Clone + std::fmt::Debug, const N: usize>(
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

// /// Make three copies of the given object for each node that requires gradients.
// ///
// /// # Notes
// ///
// /// Exactly the same as `duplicate` but with two copies.
// /// Useful for the one function that requires it
// pub fn triplicate<T: Clone + std::fmt::Debug, const N: usize>(
//     nodes: &[Option<NodeRef>; N],
//     obj: Option<T>,
// ) -> [Option<T>; N] {
//     nodes
//         .iter()
//         .flat_map(|node| match node {
//             Some(_) => [obj.clone(), obj.clone()],
//             None => None,
//         })
//         .collect::<Vec<_>>()
//         .try_into()
//         .unwrap()
// }
