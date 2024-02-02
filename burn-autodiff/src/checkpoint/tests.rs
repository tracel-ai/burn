use std::marker::PhantomData;

use burn_tensor::{backend::Backend, Tensor};

use crate::graph::NodeID;

use super::{base::InnerStates, retro::RetroForward, state::State};

#[derive(new)]
pub struct RetroDiv<B, const D: usize> {
    lhs: NodeID,
    rhs: NodeID,
    out: NodeID,
    _backend: PhantomData<B>,
}

impl<B: Backend, const D: usize> RetroForward for RetroDiv<B, D> {
    fn forward(&self, states: &mut InnerStates) {
        let lhs: B::FloatTensorPrimitive<D> = states.get_own::<B, D>(&self.lhs).into_primitive();
        let rhs: B::FloatTensorPrimitive<D> = states.get_own::<B, D>(&self.rhs).into_primitive();

        let out: Tensor<B, D> = Tensor::<B, D>::from_primitive(B::float_div(lhs, rhs));

        states.insert(
            self.out.clone(),
            State::Computed {
                state_content: Box::new(out),
                n_required: states.get_ref(&self.out).unwrap().n_required(),
            },
        );
    }
}

#[cfg(test)]
mod tests {

    use std::{collections::HashMap, sync::Arc};

    use burn_tensor::{backend::Backend, Data, Tensor};

    use crate::{
        checkpoint::{
            base::{Checkpoint, InnerStates, NodeTree, RetroForwards},
            retro::RetroLeaf,
            state::{State, StateContent},
        },
        graph::{Node, NodeID, NodeRef, Requirement},
    };

    use super::*;

    #[cfg(not(target_os = "macos"))]
    pub type TestBackend = burn_wgpu::Wgpu<burn_wgpu::Vulkan, f32, i32>;

    #[cfg(target_os = "macos")]
    pub type TestBackend = burn_wgpu::Wgpu<burn_wgpu::Metal, f32, i32>;

    #[cfg(feature = "std")]
    pub type TestAutodiffBackend = burn_autodiff::Autodiff<TestBackend>;

    fn insert_lazy(id: NodeID, inner_states: &mut InnerStates, n_required: Option<usize>) {
        inner_states.insert(
            id.clone(),
            State::Lazy {
                node_id: id.clone(),
                n_required: match n_required {
                    Some(n) => n,
                    None => 1,
                },
            },
        );
    }

    fn insert_computed(id: NodeID, inner_states: &mut InnerStates, state_content: StateContent) {
        inner_states.insert(
            id.clone(),
            State::Computed {
                state_content,
                n_required: 1,
            },
        );
    }

    fn expect_tensor<B: Backend>(states: &mut Checkpoint, id: NodeID, expected: Tensor<B, 2>) {
        let obtained: Tensor<B, 2> = states.get(id.clone());
        let x: Data<f32, 2> = expected.to_data().convert();
        let y: Data<f32, 2> = obtained.to_data().convert();
        x.assert_approx_eq(&y, 3);
    }

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

    fn make_leaves<B: Backend>(
        device: &B::Device,
        ids: [NodeID; 4],
    ) -> (InnerStates, RetroForwards, NodeTree) {
        let mut retro_forwards = RetroForwards::default();
        let mut node_tree = NodeTree::default();
        let mut inner_states = InnerStates::default();

        let mut make_leaf = |id: NodeID, t: Tensor<B, 2>| {
            let n: NodeRef = Arc::new(Node::new(Vec::new(), 0, id.clone(), Requirement::Grad));
            let retro_leaf = RetroLeaf::new(id.clone(), t);
            retro_forwards.insert(id.clone(), Box::new(retro_leaf));
            node_tree.insert(id.clone(), n);
            insert_lazy(id, &mut inner_states, None);
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

        (inner_states, retro_forwards, node_tree)
    }

    fn div_lazy_tree<B: Backend>(
        device: &B::Device,
        ids: [NodeID; 7],
        n_required_changed: HashMap<NodeID, usize>,
    ) -> Checkpoint {
        // 4: 0 / 1         5: 2 / 3 -> 5
        //       --> 6: t4 / t5 <--

        let id_0 = ids[0].clone();
        let id_1 = ids[1].clone();
        let id_2 = ids[2].clone();
        let id_3 = ids[3].clone();

        let leaves = make_leaves::<B>(
            device,
            [id_0.clone(), id_1.clone(), id_2.clone(), id_3.clone()],
        );
        let mut inner_states = leaves.0;
        let mut retro_forwards = leaves.1;
        let mut nodes = leaves.2;

        let mut make_div_node = |id: NodeID, parents: &[NodeID; 2]| {
            let n: NodeRef = Arc::new(Node::new(parents.into(), 0, id.clone(), Requirement::Grad));
            let retro_div =
                RetroDiv::<B, 2>::new(parents[0].clone(), parents[1].clone(), id.clone());
            retro_forwards.insert(id.clone(), Box::new(retro_div));
            nodes.insert(id.clone(), n);
        };

        // Node 4: t0/t1
        make_div_node(ids[4].clone(), &[id_0, id_1]);

        // Node 5: t2/t3
        make_div_node(ids[5].clone(), &[id_2, id_3]);

        // Node 6: t4/t5
        make_div_node(ids[6].clone(), &[ids[4].clone(), ids[5].clone()]);

        insert_lazy(
            ids[4].clone(),
            &mut inner_states,
            n_required_changed.get(&ids[4]).map(|x| *x),
        );
        insert_lazy(
            ids[5].clone(),
            &mut inner_states,
            n_required_changed.get(&ids[5]).map(|x| *x),
        );
        insert_lazy(
            ids[6].clone(),
            &mut inner_states,
            n_required_changed.get(&ids[6]).map(|x| *x),
        );

        Checkpoint::new(inner_states, retro_forwards, nodes)
    }

    fn div_computed_tree<B: Backend>(device: &B::Device, ids: [NodeID; 7]) -> Checkpoint {
        // Here we hardcode a result for 5 (wrong for ensuring it doesn't just do the lazy computation)
        // We can then test that 5 gives that result, and that 6 gives the implied result
        // 4: 0 / 1         5: 2 / 3 -> 5 is Computed
        //       --> 6: t4 / t5 <--

        let id_0 = ids[0].clone();
        let id_1 = ids[1].clone();
        let id_2 = ids[2].clone();
        let id_3 = ids[3].clone();

        let leaves = make_leaves::<B>(
            device,
            [id_0.clone(), id_1.clone(), id_2.clone(), id_3.clone()],
        );
        let mut inner_states = leaves.0;
        let mut retro_forwards = leaves.1;
        let mut nodes = leaves.2;

        let mut make_div_node = |id: NodeID, parents: &[NodeID; 2]| {
            let n: NodeRef = Arc::new(Node::new(parents.into(), 0, id.clone(), Requirement::Grad));
            let retro_div =
                RetroDiv::<B, 2>::new(parents[0].clone(), parents[1].clone(), id.clone());
            retro_forwards.insert(id.clone(), Box::new(retro_div));
            nodes.insert(id.clone(), n);
        };

        // Node 4: 0/t1
        make_div_node(ids[4].clone(), &[id_0, id_1]);

        // Node 5: t2/t3
        make_div_node(ids[5].clone(), &[id_2, id_3]);

        // Node 6: t4/t5
        make_div_node(ids[6].clone(), &[ids[4].clone(), ids[5].clone()]);

        insert_lazy(ids[4].clone(), &mut inner_states, None);
        insert_computed(
            ids[5].clone(),
            &mut inner_states,
            Box::new(Tensor::<B, 2>::from_data(
                Data::<f32, 2>::from([[10.0, 10.0], [10.0, 10.0]]).convert(),
                device,
            )),
        );
        insert_lazy(ids[6].clone(), &mut inner_states, None);

        Checkpoint::new(inner_states, retro_forwards, nodes)
    }
    #[test]
    fn div_lazy_tree_has_expected_leaves() {
        let device = Default::default();
        let ids = make_ids();
        let mut states = div_lazy_tree::<TestBackend>(&device, ids.clone(), HashMap::new());

        expect_tensor(
            &mut states,
            ids[1].clone(),
            Tensor::<TestBackend, 2>::from_data([[3.0, 6.0], [8.0, -9.0]], &device),
        );

        expect_tensor(
            &mut states,
            ids[2].clone(),
            Tensor::<TestBackend, 2>::from_data([[-6.0, 1.0], [4.0, 2.0]], &device),
        );
    }

    #[test]
    fn div_lazy_tree_accepts_several_independant_node_gets() {
        let device = Default::default();
        let ids = make_ids();
        let mut states = div_lazy_tree::<TestBackend>(&device, ids.clone(), HashMap::new());

        expect_tensor(
            &mut states,
            ids[4].clone(),
            Tensor::<TestBackend, 2>::from_data([[0.6666, -0.1666], [0.625, -0.2222]], &device),
        );

        expect_tensor(
            &mut states,
            ids[5].clone(),
            Tensor::<TestBackend, 2>::from_data([[-1.5, 0.125], [-0.8, 0.4]], &device),
        );
    }

    #[test]
    #[should_panic]
    fn div_lazy_tree_rejects_more_gets_than_required() {
        let device = Default::default();
        let ids = make_ids();
        let mut states = div_lazy_tree::<TestBackend>(&device, ids.clone(), HashMap::new());

        expect_tensor(
            &mut states,
            ids[5].clone(),
            Tensor::<TestBackend, 2>::from_data([[-1.5, 0.125], [-0.8, 0.4]], &device),
        );

        expect_tensor(
            &mut states,
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
        let mut checkpoint = div_lazy_tree::<TestBackend>(&device, ids.clone(), n_required_changed);

        // First call
        checkpoint.get::<TestBackend, 2>(ids[6].clone());

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
        let mut states = div_computed_tree::<TestBackend>(&device, ids.clone());

        expect_tensor(
            &mut states,
            ids[4].clone(),
            Tensor::<TestBackend, 2>::from_data([[0.6666, -0.1666], [0.625, -0.2222]], &device),
        );

        expect_tensor(
            &mut states,
            ids[5].clone(),
            Tensor::<TestBackend, 2>::from_data([[10.0, 10.0], [10.0, 10.0]], &device),
        );
    }

    #[test]
    fn div_computed_tree_has_expected_lazily_computed_node() {
        let device = Default::default();
        let ids = make_ids();
        let mut states = div_computed_tree::<TestBackend>(&device, ids.clone());

        expect_tensor(
            &mut states,
            ids[6].clone(),
            Tensor::<TestBackend, 2>::from_data([[0.0666, -0.0166], [0.0625, -0.0222]], &device),
        );
    }
}
