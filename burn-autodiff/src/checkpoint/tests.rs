use std::marker::PhantomData;

use burn_tensor::{backend::Backend, Data, Tensor};
use burn_wgpu::AutoGraphicsApi;

use crate::graph::NodeID;

use super::{base::InnerStates, base::RetroForward, state::State};
use std::{collections::HashMap, sync::Arc};

// use burn_tensor::{backend::Backend, Data, Tensor};

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
    fn forward(&self, states: &mut InnerStates) {
        // Get the needed outputs downcasted to their expected types
        // This will decrement n_required for both parent states
        let lhs: B::FloatTensorPrimitive<D> = states
            .get_owned_and_downcasted::<Tensor<B, D>>(&self.lhs_parent_id)
            .into_primitive();
        let rhs: B::FloatTensorPrimitive<D> = states
            .get_owned_and_downcasted::<Tensor<B, D>>(&self.rhs_parent_id)
            .into_primitive();

        // Compute the output through a call to the inner backend operation
        let out: Tensor<B, D> = Tensor::<B, D>::from_primitive(B::float_div(lhs, rhs));

        // Replace the state for this node id by the new computed output
        // without changing n_required
        states.insert(
            self.self_id.clone(),
            State::Computed {
                state_content: Box::new(out),
                n_required: states.get_ref(&self.self_id).unwrap().n_required(),
            },
        );
    }
}

use crate::{
    checkpoint::{
        base::{Checkpoint, NodeTree, RetroForwards},
        state::StateContent,
    },
    graph::{Node, NodeRef, Requirement},
};

pub type TestBackend = burn_wgpu::Wgpu<AutoGraphicsApi, f32, i32>;

#[test]
fn div_lazy_tree_has_expected_leaves() {
    let device = Default::default();
    let ids = make_ids();
    let mut checkpoint = div_recompute_tree::<TestBackend>(&device, ids.clone(), HashMap::new());

    expect_tensor(
        &mut checkpoint,
        ids[1].clone(),
        Tensor::<TestBackend, 2>::from_data([[3.0, 6.0], [8.0, -9.0]], &device),
    );

    expect_tensor(
        &mut checkpoint,
        ids[2].clone(),
        Tensor::<TestBackend, 2>::from_data([[-6.0, 1.0], [4.0, 2.0]], &device),
    );
}

#[test]
fn div_lazy_tree_accepts_several_independant_node_gets() {
    let device = Default::default();
    let ids = make_ids();
    let mut checkpoint = div_recompute_tree::<TestBackend>(&device, ids.clone(), HashMap::new());

    expect_tensor(
        &mut checkpoint,
        ids[4].clone(),
        Tensor::<TestBackend, 2>::from_data([[0.6666, -0.1666], [0.625, -0.2222]], &device),
    );

    expect_tensor(
        &mut checkpoint,
        ids[5].clone(),
        Tensor::<TestBackend, 2>::from_data([[-1.5, 0.125], [-0.8, 0.4]], &device),
    );
}

#[test]
#[should_panic]
fn div_lazy_tree_rejects_more_gets_than_required() {
    let device = Default::default();
    let ids = make_ids();
    let mut checkpoint = div_recompute_tree::<TestBackend>(&device, ids.clone(), HashMap::new());

    expect_tensor(
        &mut checkpoint,
        ids[5].clone(),
        Tensor::<TestBackend, 2>::from_data([[-1.5, 0.125], [-0.8, 0.4]], &device),
    );

    expect_tensor(
        &mut checkpoint,
        ids[6].clone(),
        Tensor::<TestBackend, 2>::from_data([[-0.4444, -1.3328], [-0.78125, -0.5555]], &device),
    );
}

#[test]
fn div_lazy_tree_called_twice_uses_cached_values() {
    let device = Default::default();
    let ids = make_ids();
    let mut n_required_changed = HashMap::new();
    n_required_changed.insert(ids[6].clone(), 2);
    let mut checkpoint =
        div_recompute_tree::<TestBackend>(&device, ids.clone(), n_required_changed);

    // First call
    checkpoint.get::<Tensor<TestBackend, 2>>(ids[6].clone());

    // Artificially changes parent answer and should not impact already computed child
    checkpoint.insert_pre_computed(
        ids[4].clone(),
        State::Computed {
            state_content: Box::new(Tensor::<TestBackend, 2>::from_data(
                [[99., 99.], [99., 99.]],
                &device,
            )),
            n_required: 1,
        },
    );

    // Second call
    expect_tensor(
        &mut checkpoint,
        ids[6].clone(),
        Tensor::<TestBackend, 2>::from_data([[-0.4444, -1.3328], [-0.78125, -0.5555]], &device),
    );
}

#[test]
fn div_computed_tree_has_expected_directly_computed_node() {
    let device = Default::default();
    let ids = make_ids();
    let mut checkpoint = div_precomputed_tree::<TestBackend>(&device, ids.clone());

    expect_tensor(
        &mut checkpoint,
        ids[4].clone(),
        Tensor::<TestBackend, 2>::from_data([[0.6666, -0.1666], [0.625, -0.2222]], &device),
    );

    expect_tensor(
        &mut checkpoint,
        ids[5].clone(),
        Tensor::<TestBackend, 2>::from_data([[10.0, 10.0], [10.0, 10.0]], &device),
    );
}

#[test]
fn div_computed_tree_has_expected_lazily_computed_node() {
    let device = Default::default();
    let ids = make_ids();
    let mut checkpoint = div_precomputed_tree::<TestBackend>(&device, ids.clone());

    expect_tensor(
        &mut checkpoint,
        ids[6].clone(),
        Tensor::<TestBackend, 2>::from_data([[0.0666, -0.0166], [0.0625, -0.0222]], &device),
    );
}

#[test]
fn div_lazy_graph_with_cycle_works() {
    let device = Default::default();
    let ids = [NodeID::new(), NodeID::new(), NodeID::new(), NodeID::new()];
    let mut checkpoint = div_lazy_graph_with_cycle::<TestBackend>(&device, ids.clone());

    expect_tensor(
        &mut checkpoint,
        ids[3].clone(),
        Tensor::<TestBackend, 2>::from_data([[0.2222, -0.0277], [0.0781, 0.0247]], &device),
    );
}

/// Inserts a state that needs recompute.
/// Number of times required is defined here for tests; normally it should be incremented
/// during the forward pass while building the autodiff graph
fn insert_recompute(id: NodeID, inner_states: &mut InnerStates, n_required: Option<usize>) {
    inner_states.insert(
        id.clone(),
        State::Recompute {
            n_required: n_required.unwrap_or(1),
        },
    );
}

/// Inserts a pre-computed state
fn insert_precomputed(
    id: NodeID,
    inner_states: &mut InnerStates,
    state_content: StateContent,
    n_required: Option<usize>,
) {
    inner_states.insert(
        id.clone(),
        State::Computed {
            state_content,
            n_required: n_required.unwrap_or(1),
        },
    );
}

/// Asserts the tensor obtained through checkpointing is the right one
fn expect_tensor<B: Backend>(states: &mut Checkpoint, id: NodeID, expected: Tensor<B, 2>) {
    let obtained: Tensor<B, 2> = states.get(id.clone());
    let x: Data<f32, 2> = expected.to_data().convert();
    let y: Data<f32, 2> = obtained.to_data().convert();
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
fn make_leaves<B: Backend>(device: &B::Device, ids: [NodeID; 4]) -> (InnerStates, NodeTree) {
    let mut node_tree = NodeTree::default();
    let mut inner_states = InnerStates::default();

    // Leaves are just tensors, so they are always precomputed
    let mut make_leaf = |id: NodeID, t: Tensor<B, 2>| {
        let n: NodeRef = Arc::new(Node::new(Vec::new(), 0, id.clone(), Requirement::Grad));
        node_tree.insert(id.clone(), n);
        insert_precomputed(id, &mut inner_states, Box::new(t), None);
    };

    // Leaf 0
    make_leaf(
        ids[0].clone(),
        Tensor::<B, 2>::from_data(
            Data::<f32, 2>::from([[2.0, -1.0], [5.0, 2.0]]).convert(),
            device,
        )
        .require_grad(),
    );

    // Leaf 1
    make_leaf(
        ids[1].clone(),
        Tensor::<B, 2>::from_data(
            Data::<f32, 2>::from([[3.0, 6.0], [8.0, -9.0]]).convert(),
            device,
        )
        .require_grad(),
    );

    // Leaf 2
    make_leaf(
        ids[2].clone(),
        Tensor::<B, 2>::from_data(
            Data::<f32, 2>::from([[-6.0, 1.0], [4.0, 2.0]]).convert(),
            device,
        )
        .require_grad(),
    );

    // Leaf 3
    make_leaf(
        ids[3].clone(),
        Tensor::<B, 2>::from_data(
            Data::<f32, 2>::from([[4.0, 8.0], [-5.0, 5.0]]).convert(),
            device,
        )
        .require_grad(),
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
) -> Checkpoint {
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
        retro_forwards.insert(id.clone(), Box::new(retro_div));
        nodes.insert(id.clone(), n);
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
        n_required_changed.get(&ids[4]).map(|x| *x),
    );
    insert_recompute(
        ids[5].clone(),
        &mut inner_states,
        n_required_changed.get(&ids[5]).map(|x| *x),
    );
    insert_recompute(
        ids[6].clone(),
        &mut inner_states,
        n_required_changed.get(&ids[6]).map(|x| *x),
    );

    Checkpoint::new(inner_states, retro_forwards, nodes)
}

/// Makes a tree like div_recompute_tree but where node id 5 is precomputed
/// Ids at indices 0, 1, 2, 3 correspond leaves
/// Then ids 5, 6, 7 correspond to division nodes like in the folowing
/// 4: 0 / 1         5: 2 / 3
///        6: 4 / 5
///
/// Note: precomputed result for 5 is wrong on purpose so it's possible to
/// differentiate between precomputed and recompute states.
fn div_precomputed_tree<B: Backend>(device: &B::Device, ids: [NodeID; 7]) -> Checkpoint {
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
        retro_forwards.insert(id.clone(), Box::new(retro_div));
        nodes.insert(id.clone(), n);
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
        Box::new(Tensor::<B, 2>::from_data(
            Data::<f32, 2>::from([[10.0, 10.0], [10.0, 10.0]]).convert(),
            device,
        )),
        None,
    );
    insert_recompute(ids[6].clone(), &mut inner_states, None);

    Checkpoint::new(inner_states, retro_forwards, nodes)
}

/// Makes this graph, where id 2 is used twice
/// (0 / 1) / 1
fn div_lazy_graph_with_cycle<B: Backend>(device: &B::Device, ids: [NodeID; 4]) -> Checkpoint {
    let id_0 = ids[0].clone();
    let id_1 = ids[1].clone();

    let mut node_tree = NodeTree::default();
    let mut inner_states = InnerStates::default();
    let mut retro_forwards = RetroForwards::default();

    // Leaves are just tensors, so they are always precomputed
    let mut make_leaf = |id: NodeID, t: Tensor<B, 2>, n_required: Option<usize>| {
        let n: NodeRef = Arc::new(Node::new(Vec::new(), 0, id.clone(), Requirement::Grad));
        node_tree.insert(id.clone(), n);
        insert_precomputed(id, &mut inner_states, Box::new(t), n_required);
    };

    // Leaf 0
    make_leaf(
        ids[0].clone(),
        Tensor::<B, 2>::from_data(
            Data::<f32, 2>::from([[2.0, -1.0], [5.0, 2.0]]).convert(),
            device,
        )
        .require_grad(),
        None,
    );

    // Leaf 1
    make_leaf(
        ids[1].clone(),
        Tensor::<B, 2>::from_data(
            Data::<f32, 2>::from([[3.0, 6.0], [8.0, -9.0]]).convert(),
            device,
        )
        .require_grad(),
        Some(2),
    );

    let mut make_div_node = |id: NodeID, parents: &[NodeID; 2]| {
        let n: NodeRef = Arc::new(Node::new(parents.into(), 0, id.clone(), Requirement::Grad));
        let retro_div = RetroDiv::<B, 2>::new(parents[0].clone(), parents[1].clone(), id.clone());
        retro_forwards.insert(id.clone(), Box::new(retro_div));
        node_tree.insert(id.clone(), n);
    };

    make_div_node(ids[2].clone(), &[id_0, id_1.clone()]);
    make_div_node(ids[3].clone(), &[ids[2].clone(), id_1]);

    insert_recompute(ids[2].clone(), &mut inner_states, None);
    insert_recompute(ids[3].clone(), &mut inner_states, None);

    Checkpoint::new(inner_states, retro_forwards, node_tree)
}
