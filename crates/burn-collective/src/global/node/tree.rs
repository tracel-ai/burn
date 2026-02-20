use std::{collections::HashMap, sync::Arc};

use crate::{NodeId, global::shared::GlobalCollectiveError, node::sync::SyncService};
use burn_communication::{Address, Protocol, data_service::TensorDataService};
use burn_tensor::{TensorMetadata, backend::Backend};
use futures::{StreamExt, stream::FuturesUnordered};

struct TreeTopology {
    parents: HashMap<NodeId, NodeId>,
    children: HashMap<NodeId, Vec<NodeId>>,
}

/// Global all-reduce, using a b-tree strategy.
///
/// Returns the resulting tensor on the same device as the input tensor
pub(crate) async fn tree_all_reduce_sum<B, P>(
    node: NodeId,
    nodes: &HashMap<NodeId, Address>,
    data_service: Arc<TensorDataService<B, P>>,
    sync_service: Arc<SyncService<P>>,
    tensor: B::FloatTensorPrimitive,
    arity: u32,
    base_id: u64,
) -> Result<B::FloatTensorPrimitive, GlobalCollectiveError>
where
    B: Backend,
    P: Protocol,
{
    let shape = tensor.shape();
    let device = &B::float_device(&tensor);

    // Topology could be cached based on (nodes.keys().cloned(), arity)
    let strategy = get_tree_topology(nodes.keys().cloned().collect::<Vec<_>>(), arity);

    // Transfer 1: Download and sum tensors from children
    let mut result = tensor;

    if let Some(children) = strategy.children.get(&node) {
        let mut downloads = children
            .iter()
            .map(|child| {
                let child_addr = nodes.get(child).unwrap().clone();
                let data_service = data_service.clone();
                async move {
                    let data = data_service
                        .download_tensor(child_addr.clone(), base_id.into())
                        .await
                        .ok_or(GlobalCollectiveError::PeerLost(*child))?;
                    Ok::<B::FloatTensorPrimitive, GlobalCollectiveError>(B::float_from_data(
                        data, device,
                    ))
                }
            })
            .collect::<FuturesUnordered<_>>();

        for _ in children {
            let res = downloads.next().await.unwrap().unwrap();
            if res.shape() != shape {
                return Err(GlobalCollectiveError::PeerSentIncoherentTensor);
            }
            result = B::float_add(result, res);
        }
    }

    // Transfer 2: Expose result to parent and download final result if not root
    if let Some(parent) = strategy.parents.get(&node) {
        data_service.expose(result.clone(), 1, base_id.into()).await;

        let parent_addr = nodes.get(parent).unwrap().clone();

        let data = data_service
            .download_tensor(parent_addr.clone(), (base_id + 1).into())
            .await
            .ok_or(GlobalCollectiveError::PeerLost(*parent))?;

        let parent_tensor = B::float_from_data(data, device);
        if parent_tensor.shape() != shape {
            return Err(GlobalCollectiveError::PeerSentIncoherentTensor);
        }
        result = parent_tensor;
    }

    // Transfer 3: Expose final result to children (if any)
    if let Some(children) = strategy.children.get(&node)
        && !children.is_empty()
    {
        data_service
            .expose(result.clone(), children.len() as u32, (base_id + 1).into())
            .await;
    }

    // Final barrier
    sync_service.sync().await;

    Ok(result)
}

/// Get the tree topology.
///
/// * `nodes` - List of node ids. Order doesn't matter. Nodes must be unique.
fn get_tree_topology(mut nodes: Vec<NodeId>, arity: u32) -> TreeTopology {
    assert!(arity >= 1, "Arity must be ≥ 1");

    nodes.sort(); // Sort 

    let n = nodes.len();
    let k = arity as usize;

    let mut parents: HashMap<_, _> = HashMap::with_capacity(n);
    let mut children: HashMap<_, _> = HashMap::with_capacity(n);

    for (i, &parent_id) in nodes.iter().enumerate() {
        // compute the window [first_child, last_child)
        let first = i * k + 1;
        if first < n {
            let last = usize::min(first + k, n);
            let mut ch = Vec::with_capacity(last - first);
            for &child_id in &nodes[first..last] {
                parents.insert(child_id, parent_id);
                ch.push(child_id);
            }
            children.insert(parent_id, ch);
        } else {
            // leaf‐node: no children
            children.insert(parent_id, Vec::new());
        }
    }

    TreeTopology { parents, children }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test the tree topology algorithm with arity 2 and 7 nodes
    #[test]
    fn test_get_tree_topology_arity2_size7() {
        let mut nodes = vec![];
        for i in 0..7 {
            nodes.push(i.into());
        }

        let topology = get_tree_topology(nodes, 2);

        // Root is 0, so it should have no parent
        assert!(!topology.parents.contains_key(&0.into()));

        // Parents:
        //   Node 1 and 2 → parent 0
        //   Node 3 and 4 → parent 1
        //   Node 5 and 6 → parent 2
        let expected_parents = [
            (1.into(), 0.into()),
            (2.into(), 0.into()),
            (3.into(), 1.into()),
            (4.into(), 1.into()),
            (5.into(), 2.into()),
            (6.into(), 2.into()),
        ];
        for (child, parent) in &expected_parents {
            assert_eq!(
                topology.parents.get(child),
                Some(parent),
                "wrong parent for {child:?}"
            );
        }
        // There should be exactly 6 entries in parents
        assert_eq!(topology.parents.len(), expected_parents.len());

        // Children:
        //   0 → [1, 2]
        //   1 → [3, 4]
        //   2 → [5, 6]
        //   3,4,5,6 → []
        assert_eq!(
            topology.children.get(&0.into()),
            Some(&vec![1.into(), 2.into()])
        );
        assert_eq!(
            topology.children.get(&1.into()),
            Some(&vec![3.into(), 4.into()])
        );
        assert_eq!(
            topology.children.get(&2.into()),
            Some(&vec![5.into(), 6.into()])
        );
        // Leaves
        for leaf in 3..7 {
            assert_eq!(
                topology.children.get(&leaf.into()),
                Some(&Vec::new()),
                "leaf {leaf:?} should have no children"
            );
        }
        // Ensure we have exactly 7 entries in children
        assert_eq!(topology.children.len(), 7);
    }
}
