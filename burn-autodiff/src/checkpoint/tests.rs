use std::marker::PhantomData;

use burn_tensor::{backend::Backend, Data, Tensor};
use burn_wgpu::AutoGraphicsApi;

use crate::graph::NodeID;

use std::{collections::HashMap, sync::Arc};

use crate::{
    checkpoint::{
        base::{Checkpointer, NodeTree, RetroForwards},
        state::StateContent,
    },
    graph::{Node, NodeRef, Requirement},
};

use super::{
    base::RetroForward,
    state::{BackwardStates, State},
};

pub type TestBackend = burn_wgpu::Wgpu<AutoGraphicsApi, f32, i32>;

#[derive(new)]
/// For testing purpose, all operations are float divisions.
pub struct RetroDiv<B, const D: usize> {
    lhs_parent_id: NodeID,
    rhs_parent_id: NodeID,
    self_id: NodeID,
    _backend: PhantomData<B>,
}

impl<B: Backend, const D: usize> RetroForward for RetroDiv<B, D> {
    /// Typical content of a [RetroForward] function.
    fn forward(&self, states: &mut BackwardStates) {
        // Get the needed outputs downcasted to their expected types
        // This will decrement n_required for both parent states
        let lhs = states.get_state::<B::FloatTensorPrimitive<D>>(&self.lhs_parent_id);
        let rhs = states.get_state::<B::FloatTensorPrimitive<D>>(&self.rhs_parent_id);

        // Compute the output through a call to the inner backend operation
        let out = B::float_div(lhs, rhs);

        // Replace the state for this node id by the new computed output
        // without changing n_required
        states.save(self.self_id.clone(), out);
    }
}

#[test]
fn div_lazy_tree_has_expected_leaves() {
    let device = Default::default();
    let ids = make_ids();
    let mut checkpointer = div_recompute_tree::<TestBackend>(&device, ids.clone(), HashMap::new());

    expect_tensor::<TestBackend>(
        &mut checkpointer,
        ids[1].clone(),
        Tensor::<TestBackend, 2>::from_data([[3.0, 6.0], [8.0, -9.0]], &device).into_primitive(),
    );

    expect_tensor::<TestBackend>(
        &mut checkpointer,
        ids[2].clone(),
        Tensor::<TestBackend, 2>::from_data([[-6.0, 1.0], [4.0, 2.0]], &device).into_primitive(),
    );
}

#[test]
fn div_lazy_tree_accepts_several_independant_node_gets() {
    let device = Default::default();
    let ids = make_ids();
    let mut checkpointer = div_recompute_tree::<TestBackend>(&device, ids.clone(), HashMap::new());

    expect_tensor::<TestBackend>(
        &mut checkpointer,
        ids[4].clone(),
        Tensor::<TestBackend, 2>::from_data([[0.6666, -0.1666], [0.625, -0.2222]], &device)
            .into_primitive(),
    );

    expect_tensor::<TestBackend>(
        &mut checkpointer,
        ids[5].clone(),
        Tensor::<TestBackend, 2>::from_data([[-1.5, 0.125], [-0.8, 0.4]], &device).into_primitive(),
    );
}

#[test]
#[should_panic]
fn div_lazy_tree_rejects_more_gets_than_required() {
    let device = Default::default();
    let ids = make_ids();
    let mut checkpointer = div_recompute_tree::<TestBackend>(&device, ids.clone(), HashMap::new());

    expect_tensor::<TestBackend>(
        &mut checkpointer,
        ids[5].clone(),
        Tensor::<TestBackend, 2>::from_data([[-1.5, 0.125], [-0.8, 0.4]], &device).into_primitive(),
    );

    expect_tensor::<TestBackend>(
        &mut checkpointer,
        ids[6].clone(),
        Tensor::<TestBackend, 2>::from_data([[-0.4444, -1.3328], [-0.78125, -0.5555]], &device)
            .into_primitive(),
    );
}

#[test]
fn div_lazy_tree_called_twice_uses_cached_values() {
    let device = Default::default();
    let ids = make_ids();
    let mut n_required_changed = HashMap::new();
    n_required_changed.insert(ids[6].clone(), 2);
    let mut checkpointer =
        div_recompute_tree::<TestBackend>(&device, ids.clone(), n_required_changed);

    // First call
    checkpointer
        .retrieve_output::<<TestBackend as Backend>::FloatTensorPrimitive<2>>(ids[6].clone());

    // Artificially changes parent answer and should not impact already computed child
    checkpointer.checkpoint(
        ids[4].clone(),
        Tensor::<TestBackend, 2>::from_data([[99., 99.], [99., 99.]], &device),
        1,
    );

    // Second call
    expect_tensor::<TestBackend>(
        &mut checkpointer,
        ids[6].clone(),
        Tensor::<TestBackend, 2>::from_data([[-0.4444, -1.3328], [-0.78125, -0.5555]], &device)
            .into_primitive(),
    );
}

#[test]
fn div_computed_tree_has_expected_directly_computed_node() {
    let device = Default::default();
    let ids = make_ids();
    let mut checkpointer = div_precomputed_tree::<TestBackend>(&device, ids.clone());

    expect_tensor::<TestBackend>(
        &mut checkpointer,
        ids[4].clone(),
        Tensor::<TestBackend, 2>::from_data([[0.6666, -0.1666], [0.625, -0.2222]], &device)
            .into_primitive(),
    );

    expect_tensor::<TestBackend>(
        &mut checkpointer,
        ids[5].clone(),
        Tensor::<TestBackend, 2>::from_data([[10.0, 10.0], [10.0, 10.0]], &device).into_primitive(),
    );
}

#[test]
fn div_computed_tree_has_expected_lazily_computed_node() {
    let device = Default::default();
    let ids = make_ids();
    let mut checkpointer = div_precomputed_tree::<TestBackend>(&device, ids.clone());

    expect_tensor::<TestBackend>(
        &mut checkpointer,
        ids[6].clone(),
        Tensor::<TestBackend, 2>::from_data([[0.0666, -0.0166], [0.0625, -0.0222]], &device)
            .into_primitive(),
    );
}

#[test]
fn div_lazy_graph_with_duplicate_works() {
    let device = Default::default();
    let ids = [NodeID::new(), NodeID::new(), NodeID::new(), NodeID::new()];
    let mut checkpointer = div_lazy_graph_with_duplicate::<TestBackend>(&device, ids.clone());

    expect_tensor::<TestBackend>(
        &mut checkpointer,
        ids[3].clone(),
        Tensor::<TestBackend, 2>::from_data([[0.2222, -0.0277], [0.0781, 0.0247]], &device)
            .into_primitive(),
    );
}

/// Inserts a state that needs recompute.
/// Number of times required is defined here for tests; normally it should be incremented
/// during the forward pass while building the autodiff graph
fn insert_recompute(id: NodeID, inner_states: &mut BackwardStates, n_required: Option<usize>) {
    inner_states.insert_state(
        id.clone(),
        State::Recompute {
            n_required: n_required.unwrap_or(1),
        },
    );
}

/// Inserts a pre-computed state
fn insert_precomputed(
    id: NodeID,
    inner_states: &mut BackwardStates,
    state_content: StateContent,
    n_required: Option<usize>,
) {
    inner_states.insert_state(
        id.clone(),
        State::Computed {
            state_content,
            n_required: n_required.unwrap_or(1),
        },
    );
}

/// Asserts the tensor obtained through checkpointing is the right one
fn expect_tensor<B: Backend>(
    checkpointer: &mut Checkpointer,
    id: NodeID,
    expected: B::FloatTensorPrimitive<2>,
) {
    let obtained: B::FloatTensorPrimitive<2> = checkpointer.retrieve_output(id.clone());
    let x: Data<f32, 2> = Tensor::<B, 2>::from_primitive(expected).to_data().convert();
    let y: Data<f32, 2> = Tensor::<B, 2>::from_primitive(obtained).to_data().convert();
    x.assert_approx_eq(&y, 3);
}

/// Ids for the div tree of 7 nodes
fn make_ids() -> [NodeID; 7] {
    [
        NodeID::new(),
        NodeID::new(),
        NodeID::new(),
        NodeID::new(),
        NodeID::new(),
        NodeID::new(),
        NodeID::new(),
    ]
}

/// Make the leaves for a div tree
fn make_leaves<B: Backend>(device: &B::Device, ids: [NodeID; 4]) -> (BackwardStates, NodeTree) {
    let mut node_tree = NodeTree::default();
    let mut inner_states = BackwardStates::default();

    // Leaves are just tensors, so they are always precomputed
    let mut make_leaf = |id: NodeID, t: B::FloatTensorPrimitive<2>| {
        let n: NodeRef = Arc::new(Node::new(Vec::new(), 0, id.clone(), Requirement::Grad));
        node_tree.insert_node(id.clone(), n);
        insert_precomputed(id, &mut inner_states, Box::new(t), None);
    };

    // Leaf 0
    make_leaf(
        ids[0].clone(),
        Tensor::<B, 2>::from_data(
            Data::<f32, 2>::from([[2.0, -1.0], [5.0, 2.0]]).convert(),
            device,
        )
        .into_primitive(),
    );

    // Leaf 1
    make_leaf(
        ids[1].clone(),
        Tensor::<B, 2>::from_data(
            Data::<f32, 2>::from([[3.0, 6.0], [8.0, -9.0]]).convert(),
            device,
        )
        .into_primitive(),
    );

    // Leaf 2
    make_leaf(
        ids[2].clone(),
        Tensor::<B, 2>::from_data(
            Data::<f32, 2>::from([[-6.0, 1.0], [4.0, 2.0]]).convert(),
            device,
        )
        .into_primitive(),
    );

    // Leaf 3
    make_leaf(
        ids[3].clone(),
        Tensor::<B, 2>::from_data(
            Data::<f32, 2>::from([[4.0, 8.0], [-5.0, 5.0]]).convert(),
            device,
        )
        .into_primitive(),
    );

    (inner_states, node_tree)
}

/// Makes a tree where every node except leaves are in a Recompute state
/// Ids at indices 0, 1, 2, 3 correspond leaves
/// Then ids 5, 6, 7 correspond to division nodes like in the folowing
/// 4: 0 / 1         5: 2 / 3
///        6: t4 / t5
fn div_recompute_tree<B: Backend>(
    device: &B::Device,
    ids: [NodeID; 7],
    n_required_changed: HashMap<NodeID, usize>,
) -> Checkpointer {
    let id_0 = ids[0].clone();
    let id_1 = ids[1].clone();
    let id_2 = ids[2].clone();
    let id_3 = ids[3].clone();

    let leaves = make_leaves::<B>(
        device,
        [id_0.clone(), id_1.clone(), id_2.clone(), id_3.clone()],
    );
    let mut inner_states = leaves.0;
    let mut nodes = leaves.1;
    let mut retro_forwards = RetroForwards::default();

    let mut make_div_node = |id: NodeID, parents: &[NodeID; 2]| {
        let n: NodeRef = Arc::new(Node::new(parents.into(), 0, id.clone(), Requirement::Grad));
        let retro_div = RetroDiv::<B, 2>::new(parents[0].clone(), parents[1].clone(), id.clone());
        retro_forwards.insert_retro_forward(id.clone(), Box::new(retro_div));
        nodes.insert_node(id.clone(), n);
    };

    // Node 4: t0/t1
    make_div_node(ids[4].clone(), &[id_0, id_1]);

    // Node 5: t2/t3
    make_div_node(ids[5].clone(), &[id_2, id_3]);

    // Node 6: t4/t5
    make_div_node(ids[6].clone(), &[ids[4].clone(), ids[5].clone()]);

    insert_recompute(
        ids[4].clone(),
        &mut inner_states,
        n_required_changed.get(&ids[4]).copied(),
    );
    insert_recompute(
        ids[5].clone(),
        &mut inner_states,
        n_required_changed.get(&ids[5]).copied(),
    );
    insert_recompute(
        ids[6].clone(),
        &mut inner_states,
        n_required_changed.get(&ids[6]).copied(),
    );

    Checkpointer::new(inner_states, retro_forwards, nodes)
}

/// Makes a tree like div_recompute_tree but where node id 5 is precomputed
/// Ids at indices 0, 1, 2, 3 correspond leaves
/// Then ids 5, 6, 7 correspond to division nodes like in the folowing
/// 4: 0 / 1         5: 2 / 3
///        6: 4 / 5
///
/// Note: precomputed result for 5 is wrong on purpose so it's possible to
/// differentiate between precomputed and recompute states.
fn div_precomputed_tree<B: Backend>(device: &B::Device, ids: [NodeID; 7]) -> Checkpointer {
    let id_0 = ids[0].clone();
    let id_1 = ids[1].clone();
    let id_2 = ids[2].clone();
    let id_3 = ids[3].clone();

    let leaves = make_leaves::<B>(
        device,
        [id_0.clone(), id_1.clone(), id_2.clone(), id_3.clone()],
    );
    let mut inner_states = leaves.0;
    let mut nodes = leaves.1;
    let mut retro_forwards = RetroForwards::default();

    let mut make_div_node = |id: NodeID, parents: &[NodeID; 2]| {
        let n: NodeRef = Arc::new(Node::new(parents.into(), 0, id.clone(), Requirement::Grad));
        let retro_div = RetroDiv::<B, 2>::new(parents[0].clone(), parents[1].clone(), id.clone());
        retro_forwards.insert_retro_forward(id.clone(), Box::new(retro_div));
        nodes.insert_node(id.clone(), n);
    };

    // Node 4: 0/t1
    make_div_node(ids[4].clone(), &[id_0, id_1]);

    // Node 5: t2/t3
    make_div_node(ids[5].clone(), &[id_2, id_3]);

    // Node 6: t4/t5
    make_div_node(ids[6].clone(), &[ids[4].clone(), ids[5].clone()]);

    insert_recompute(ids[4].clone(), &mut inner_states, None);
    insert_precomputed(
        ids[5].clone(),
        &mut inner_states,
        Box::new(
            Tensor::<B, 2>::from_data(
                Data::<f32, 2>::from([[10.0, 10.0], [10.0, 10.0]]).convert(),
                device,
            )
            .into_primitive(),
        ),
        None,
    );
    insert_recompute(ids[6].clone(), &mut inner_states, None);

    Checkpointer::new(inner_states, retro_forwards, nodes)
}

/// Makes this graph, where id 1 is used twice
/// (0 / 1) / 1
fn div_lazy_graph_with_duplicate<B: Backend>(device: &B::Device, ids: [NodeID; 4]) -> Checkpointer {
    let id_0 = ids[0].clone();
    let id_1 = ids[1].clone();

    let mut node_tree = NodeTree::default();
    let mut inner_states = BackwardStates::default();
    let mut retro_forwards = RetroForwards::default();

    // Leaves are just tensors, so they are always precomputed
    let mut make_leaf = |id: NodeID, t: B::FloatTensorPrimitive<2>, n_required: Option<usize>| {
        let n: NodeRef = Arc::new(Node::new(Vec::new(), 0, id.clone(), Requirement::Grad));
        node_tree.insert_node(id.clone(), n);
        insert_precomputed(id, &mut inner_states, Box::new(t), n_required);
    };

    // Leaf 0
    make_leaf(
        ids[0].clone(),
        Tensor::<B, 2>::from_data(
            Data::<f32, 2>::from([[2.0, -1.0], [5.0, 2.0]]).convert(),
            device,
        )
        .into_primitive(),
        None,
    );

    // Leaf 1
    make_leaf(
        ids[1].clone(),
        Tensor::<B, 2>::from_data(
            Data::<f32, 2>::from([[3.0, 6.0], [8.0, -9.0]]).convert(),
            device,
        )
        .into_primitive(),
        Some(2),
    );

    let mut make_div_node = |id: NodeID, parents: &[NodeID; 2]| {
        let n: NodeRef = Arc::new(Node::new(parents.into(), 0, id.clone(), Requirement::Grad));
        let retro_div = RetroDiv::<B, 2>::new(parents[0].clone(), parents[1].clone(), id.clone());
        retro_forwards.insert_retro_forward(id.clone(), Box::new(retro_div));
        node_tree.insert_node(id.clone(), n);
    };

    make_div_node(ids[2].clone(), &[id_0, id_1.clone()]);
    make_div_node(ids[3].clone(), &[ids[2].clone(), id_1]);

    insert_recompute(ids[2].clone(), &mut inner_states, None);
    insert_recompute(ids[3].clone(), &mut inner_states, None);

    Checkpointer::new(inner_states, retro_forwards, node_tree)
}
