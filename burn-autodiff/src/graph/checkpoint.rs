use std::{any::Any, collections::HashMap, marker::PhantomData};

use burn_tensor::{backend::Backend, Tensor};

use super::{NodeID, NodeRef};

trait RetroForward {
    fn forward(&self, states: &mut InnerStates);
}

#[derive(new)]
pub struct RetroLeaf<B: Backend, const D: usize> {
    out: NodeID,
    tensor: Tensor<B, D>, // maybe remove that state and just have retroleaves as always computed
}

impl<B: Backend, const D: usize> RetroForward for RetroLeaf<B, D> {
    fn forward(&self, states: &mut InnerStates) {
        states.insert(
            self.out.clone(),
            State::Computed {
                state_content: Box::new(self.tensor.clone()), // must not clone tensor
                n_required: 1,                                // TODO arbitrary for now
            },
        );
    }
}

#[derive(new)]
pub struct RetroDiv<B, const D: usize> {
    lhs: NodeID,
    rhs: NodeID,
    out: NodeID,
    _backend: PhantomData<B>,
}

impl<B: Backend, const D: usize> RetroForward for RetroDiv<B, D> {
    fn forward(&self, states: &mut InnerStates) {
        // We assume hashmap filled with parents
        let lhs: B::FloatTensorPrimitive<D> = states
            .get(&self.lhs)
            .get_state_content()
            .downcast_ref::<Tensor<B, D>>()
            .unwrap()
            .clone()
            .into_primitive();

        let rhs: B::FloatTensorPrimitive<D> = states
            .get(&self.rhs) // TODO get_mut because change num_required -=1
            .get_state_content()
            .downcast_ref::<Tensor<B, D>>()
            .unwrap()
            .clone()
            .into_primitive();

        let out: Tensor<B, D> = Tensor::<B, D>::from_primitive(B::float_div(lhs, rhs));

        states.insert(
            self.out.clone(),
            State::Computed {
                state_content: Box::new(out),
                n_required: 1, // TODO lazy's
            },
        );
    }
}

type StateContent = Box<dyn Any + Send + Sync>;

#[derive(Debug)]
enum State {
    // Weird nomenclature. Isn't it more lazy to not re-compute?
    Lazy {
        node_id: NodeID, // whose forward is required to compute state (is it needed, as States has it as the key)
        n_required: usize, // how many times it's used (has counter += and -=)
    },
    Computed {
        state_content: StateContent,
        n_required: usize,
    },
}

impl State {
    fn get_state_content(&self) -> &StateContent {
        match self {
            State::Lazy {
                node_id: _,
                n_required: _,
            } => unreachable!("A child has been called before its parents"),
            State::Computed {
                state_content,
                n_required: _,
            } => state_content,
        }
    }
}

#[derive(new, Default)]
struct RetroForwards {
    map: HashMap<NodeID, Box<dyn RetroForward>>,
}

impl RetroForwards {
    pub fn forward(&self, node_id: &NodeID, inner_states: &mut InnerStates) {
        if let State::Lazy {
            node_id,
            n_required,
        } = inner_states.get(node_id)
        {
            self.map.get(node_id).unwrap().forward(inner_states);
        }
    }

    pub fn insert(&mut self, node_id: NodeID, retro_forward: Box<dyn RetroForward>) {
        self.map.insert(node_id, retro_forward);
    }
}

// wrapper to keep track of n_requiered. if zero remove and give ownership. in the REAL forward +=1 n_required
#[derive(new, Default)]
struct InnerStates {
    map: HashMap<NodeID, State>,
}

impl InnerStates {
    pub fn get(&self, node_id: &NodeID) -> &State {
        self.map.get(node_id).unwrap()
    }

    pub fn insert(&mut self, node_id: NodeID, state: State) {
        self.map.insert(node_id, state);
    }
}

#[derive(new)]
struct States {
    inner_states: InnerStates,
    retro_forwards: RetroForwards,
    nodes: HashMap<NodeID, NodeRef>,
}

impl States {
    pub fn get(&mut self, node_id: NodeID) -> &StateContent {
        self.topological_sort(node_id.clone())
            .iter()
            .for_each(|node| self.retro_forwards.forward(&node, &mut self.inner_states));

        self.inner_states.get(&node_id).get_state_content()
    }

    fn topological_sort(&self, node_id: NodeID) -> Vec<NodeID> {
        match self.inner_states.get(&node_id) {
            State::Lazy {
                node_id: _,
                n_required: _,
            } => {
                let mut sorted = Vec::new();
                for parent_node in self.parents(node_id.clone()) {
                    sorted.extend(self.topological_sort(parent_node));
                }
                sorted.push(node_id);
                sorted
            }
            State::Computed {
                state_content: _,
                n_required: _,
            } => vec![node_id],
        }
    }

    fn parents(&self, node_id: NodeID) -> Vec<NodeID> {
        self.nodes.get(&node_id).unwrap().parents.clone()
    }
}

// progressive transformation:
// all operations saved a custom state for their backward (if necessary)
// this does not mean they are able to perform their forward
// we must leave them the ability of computing their backward from their state
// but not assume it means they have what they need for the retro forward
// conclusion: we do not withdraw the original state, but it's useless for checkpointing, therefore by default all operations are lazy (memory bound)
// some ops will do their own backward with their state, some will ask their ancestors to retroforward, but there's no link.
// have performance tests to decide if we keep both or uniformize

// during a backward,
// an operation may need its original inputs
// it will ask the states struct's get. twice if it has two inputs
// OR just once, asking for its own node id, should work just as well

#[cfg(test)]
mod tests {

    use std::sync::Arc;

    use burn_tensor::Data;

    use crate::graph::{Node, Requirement};

    use super::*;

    #[cfg(not(target_os = "macos"))]
    pub type TestBackend = burn_wgpu::Wgpu<burn_wgpu::Vulkan, f32, i32>;

    #[cfg(target_os = "macos")]
    pub type TestBackend = burn_wgpu::Wgpu<burn_wgpu::Metal, f32, i32>;

    #[cfg(feature = "std")]
    pub type TestAutodiffBackend = burn_autodiff::Autodiff<TestBackend>;

    fn insert_lazy(id: NodeID, inner_states: &mut InnerStates) {
        inner_states.insert(
            id.clone(),
            State::Lazy {
                node_id: id.clone(),
                n_required: 1,
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

    fn make_leaves<B: Backend>(
        device: &B::Device,
        ids: [NodeID; 4],
    ) -> (InnerStates, RetroForwards, HashMap<NodeID, NodeRef>) {
        let mut retro_forwards = RetroForwards::default();
        let mut nodes = HashMap::new();
        let mut inner_states = InnerStates::default();

        let mut make_leaf = |id: NodeID, t: Tensor<B, 2>| {
            let n: NodeRef = Arc::new(Node::new(Vec::new(), 0, id.clone(), Requirement::Grad));
            let retro_leaf = RetroLeaf::new(id.clone(), t);
            retro_forwards.insert(id.clone(), Box::new(retro_leaf));
            nodes.insert(id.clone(), n);
            insert_lazy(id, &mut inner_states);
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

        (inner_states, retro_forwards, nodes)
    }

    fn div_lazy_tree<B: Backend>(device: &B::Device, ids: [NodeID; 7]) -> States {
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

        insert_lazy(ids[4].clone(), &mut inner_states);
        insert_lazy(ids[5].clone(), &mut inner_states);
        insert_lazy(ids[6].clone(), &mut inner_states);

        States::new(inner_states, retro_forwards, nodes)
    }

    fn div_computed_tree<B: Backend>(device: &B::Device, ids: [NodeID; 7]) -> States {
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

        insert_lazy(ids[4].clone(), &mut inner_states);
        insert_computed(
            ids[5].clone(),
            &mut inner_states,
            Box::new(Tensor::<B, 2>::from_data(
                Data::<f32, 2>::from([[10.0, 10.0], [10.0, 10.0]]).convert(),
                device,
            )),
        );
        insert_lazy(ids[6].clone(), &mut inner_states);

        States::new(inner_states, retro_forwards, nodes)
    }

    fn expect_tensor<B: Backend>(states: &mut States, id: NodeID, expected: Tensor<B, 2>) {
        let state_content = states.get(id.clone());
        let obtained: Tensor<B, 2> = state_content
            .downcast_ref::<Tensor<B, 2>>()
            .unwrap()
            .clone();
        let x: Data<f32, 2> = expected.to_data().convert();
        let y: Data<f32, 2> = obtained.to_data().convert();
        x.assert_approx_eq(&y, 3);
    }

    #[test]
    fn div_lazy_tree_has_expected_leaves() {
        let device = Default::default();
        let ids = [
            NodeID::new(),
            NodeID::new(),
            NodeID::new(),
            NodeID::new(),
            NodeID::new(),
            NodeID::new(),
            NodeID::new(),
        ];
        let mut states = div_lazy_tree::<TestBackend>(&device, ids.clone());

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
    fn div_lazy_tree_has_expected_nodes() {
        let device = Default::default();
        let ids = [
            NodeID::new(),
            NodeID::new(),
            NodeID::new(),
            NodeID::new(),
            NodeID::new(),
            NodeID::new(),
            NodeID::new(),
        ];
        let mut states = div_lazy_tree::<TestBackend>(&device, ids.clone());

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

        expect_tensor(
            &mut states,
            ids[6].clone(),
            Tensor::<TestBackend, 2>::from_data([[-0.4444, -1.3328], [-0.78125, -0.5555]], &device),
        );
    }

    #[test]
    fn div_computed_tree_has_expected_directly_computed_node() {
        let device = Default::default();
        let ids = [
            NodeID::new(),
            NodeID::new(),
            NodeID::new(),
            NodeID::new(),
            NodeID::new(),
            NodeID::new(),
            NodeID::new(),
        ];
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
        let ids = [
            NodeID::new(),
            NodeID::new(),
            NodeID::new(),
            NodeID::new(),
            NodeID::new(),
            NodeID::new(),
            NodeID::new(),
        ];
        let mut states = div_computed_tree::<TestBackend>(&device, ids.clone());

        expect_tensor(
            &mut states,
            ids[6].clone(),
            Tensor::<TestBackend, 2>::from_data([[0.0666, -0.0166], [0.0625, -0.0222]], &device),
        );
    }
}
