use std::{any::Any, collections::HashMap, marker::PhantomData};

use burn_tensor::{backend::Backend, Tensor};

use super::{NodeID, NodeRef};

trait RetroForward {
    fn forward(&self, states: &mut HashMap<NodeID, State>);
}

#[derive(new)]
pub struct RetroLeaf<B: Backend, const D: usize> {
    out: NodeID,
    tensor: Tensor<B, D>,
}

impl<B: Backend, const D: usize> RetroForward for RetroLeaf<B, D> {
    fn forward(&self, states: &mut HashMap<NodeID, State>) {
        println!("here with {:?}", self.out.clone());
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
    fn forward(&self, states: &mut HashMap<NodeID, State>) {
        // We assume hashmap filled with parents
        let lhs: B::FloatTensorPrimitive<D> = states
            .get(&self.lhs)
            .unwrap()
            .get_state_content()
            .downcast_ref::<Tensor<B, D>>()
            .unwrap()
            .clone()
            .into_primitive();

        let rhs: B::FloatTensorPrimitive<D> = states
            .get(&self.rhs) // TODO get_mut because change num_required -=1
            .unwrap()
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

#[derive(new)]
struct States {
    inner_states: HashMap<NodeID, State>, // TODO wrapper to keep track of n_requiered. if zero remove and give ownership. in the REAL forward +=1 n_required
    retro_forwards: HashMap<NodeID, Box<dyn RetroForward>>,
    nodes: HashMap<NodeID, NodeRef>,
}

impl States {
    pub fn get(&mut self, node_id: NodeID) -> &StateContent {
        // get is called by backward, knowing exactly what it wants, so knows B and D
        // but must not be called recursively as D may change (and it's just a bad idea in general with muts)

        // we should build a topological sort as we only need to make sure parents are called before their children
        // then it's just a matter of adding to the hashmap in an order that won't panic, not caring about types :)
        let node_order: Vec<NodeID> = self.topological_sort(node_id.clone());

        for node in node_order {
            let retro_forward = self.retro_forwards.get(&node).unwrap();
            retro_forward.forward(&mut self.inner_states);
        }

        println!("{:?}", self.inner_states);
        self.inner_states.get(&node_id).unwrap().get_state_content()
    }

    fn topological_sort(&self, node_id: NodeID) -> Vec<NodeID> {
        match self.inner_states.get(&node_id).unwrap() {
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

    fn div_tree<B: Backend>(device: &B::Device, ids: [NodeID; 7]) -> States {
        // A: t1 / t2         B: t3 / t4
        //       --> C: t5 / t6 <--
        //                 t7

        let id_1 = ids[0].clone();
        let n_1: NodeRef = Arc::new(Node::new(Vec::new(), 0, id_1.clone(), Requirement::Grad));
        let t1 = Tensor::<B, 2>::from_data(
            Data::<f32, 2>::from([[2.0, -1.0], [5.0, 2.0]]).convert(),
            device,
        )
        .require_grad();
        let retro_leaf_1 = RetroLeaf::new(id_1.clone(), t1);

        let id_2 = ids[1].clone();
        let n_2: NodeRef = Arc::new(Node::new(Vec::new(), 0, id_2.clone(), Requirement::Grad));
        let t2 = Tensor::<B, 2>::from_data(
            Data::<f32, 2>::from([[3.0, 6.0], [8.0, -9.0]]).convert(),
            device,
        )
        .require_grad();
        let retro_leaf_2 = RetroLeaf::new(id_2.clone(), t2);

        let id_3 = ids[2].clone();
        let n_3: NodeRef = Arc::new(Node::new(Vec::new(), 0, id_3.clone(), Requirement::Grad));
        let t3 = Tensor::<B, 2>::from_data(
            Data::<f32, 2>::from([[-6.0, 1.0], [4.0, 2.0]]).convert(),
            device,
        )
        .require_grad();
        let retro_leaf_3 = RetroLeaf::new(id_3.clone(), t3);

        let id_4 = ids[3].clone();
        let n_4: NodeRef = Arc::new(Node::new(Vec::new(), 0, id_4.clone(), Requirement::Grad));
        let t4 = Tensor::<B, 2>::from_data(
            Data::<f32, 2>::from([[4.0, 8.0], [-5.0, 5.0]]).convert(),
            device,
        )
        .require_grad();
        let retro_leaf_4 = RetroLeaf::new(id_4.clone(), t4);

        let mut retro_forwards: HashMap<NodeID, Box<dyn RetroForward>> = HashMap::new();
        retro_forwards.insert(id_1.clone(), Box::new(retro_leaf_1));
        retro_forwards.insert(id_2.clone(), Box::new(retro_leaf_2));
        retro_forwards.insert(id_3.clone(), Box::new(retro_leaf_3));
        retro_forwards.insert(id_4.clone(), Box::new(retro_leaf_4));

        let id_a = ids[4].clone();
        let n_a: NodeRef = Arc::new(Node::new(
            vec![id_1.clone(), id_2.clone()],
            0,
            id_a.clone(),
            Requirement::Grad,
        ));

        let id_b = ids[5].clone();
        let n_b: NodeRef = Arc::new(Node::new(
            vec![id_3.clone(), id_4.clone()],
            0,
            id_b.clone(),
            Requirement::Grad,
        ));

        let id_c = ids[6].clone();
        let n_c: NodeRef = Arc::new(Node::new(
            vec![id_a.clone(), id_b.clone()],
            0,
            id_c.clone(),
            Requirement::Grad,
        ));

        let retro_div_a = RetroDiv::<TestBackend, 2>::new(id_1.clone(), id_2.clone(), id_a.clone());
        let retro_div_b = RetroDiv::<TestBackend, 2>::new(id_3.clone(), id_4.clone(), id_b.clone());
        let retro_div_c = RetroDiv::<TestBackend, 2>::new(id_a.clone(), id_b.clone(), id_c.clone());

        retro_forwards.insert(id_a.clone(), Box::new(retro_div_a));
        retro_forwards.insert(id_b.clone(), Box::new(retro_div_b));
        retro_forwards.insert(id_c.clone(), Box::new(retro_div_c));

        let mut nodes = HashMap::new();

        nodes.insert(id_1.clone(), n_1);
        nodes.insert(id_2.clone(), n_2);
        nodes.insert(id_3.clone(), n_3);
        nodes.insert(id_4.clone(), n_4);
        nodes.insert(id_a.clone(), n_a);
        nodes.insert(id_b.clone(), n_b);
        nodes.insert(id_c.clone(), n_c);

        // register all states as lazy
        let mut inner_states = HashMap::new();
        let mut insert_lazy = |id: NodeID| {
            inner_states.insert(
                id.clone(),
                State::Lazy {
                    node_id: id.clone(),
                    n_required: 1,
                },
            )
        };
        insert_lazy(id_1);
        insert_lazy(id_2.clone());
        insert_lazy(id_3);
        insert_lazy(id_4);
        insert_lazy(id_a);
        insert_lazy(id_b);
        insert_lazy(id_c);

        States::new(inner_states, retro_forwards, nodes)
    }

    #[test]
    fn div_tree_has_expected_leaves() {
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
        let mut states = div_tree::<TestBackend>(&device, ids.clone());

        let expected_t2: Tensor<TestBackend, 2> =
            Tensor::<TestBackend, 2>::from_data([[3.0, 6.0], [8.0, -9.0]], &device);
        let obtained_t2: Tensor<TestBackend, 2> =
            downcast_tensor::<TestBackend, 2>(states.get(ids[1].clone()));
        expected_t2
            .to_data()
            .assert_approx_eq(&obtained_t2.to_data(), 3);

        let expected_t3: Tensor<TestBackend, 2> =
            Tensor::<TestBackend, 2>::from_data([[-6.0, 1.0], [4.0, 2.0]], &device);
        let obtained_t3: Tensor<TestBackend, 2> =
            downcast_tensor::<TestBackend, 2>(states.get(ids[2].clone()));
        expected_t3
            .to_data()
            .assert_approx_eq(&obtained_t3.to_data(), 3);
    }

    #[test]
    fn div_tree_has_expected_middle_nodes() {
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
        let mut states = div_tree::<TestBackend>(&device, ids.clone());

        let expected_ta: Tensor<TestBackend, 2> =
            Tensor::<TestBackend, 2>::from_data([[0.6666, -0.1666], [0.625, -0.2222]], &device);
        let obtained_ta: Tensor<TestBackend, 2> =
            downcast_tensor::<TestBackend, 2>(states.get(ids[4].clone()));
        expected_ta
            .to_data()
            .assert_approx_eq(&obtained_ta.to_data(), 3);

        let expected_tb: Tensor<TestBackend, 2> =
            Tensor::<TestBackend, 2>::from_data([[-1.5, 0.125], [-0.8, 0.4]], &device);
        let obtained_tb: Tensor<TestBackend, 2> =
            downcast_tensor::<TestBackend, 2>(states.get(ids[5].clone()));
        expected_tb
            .to_data()
            .assert_approx_eq(&obtained_tb.to_data(), 3);
    }

    #[test]
    fn div_tree_has_expected_root() {
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
        let mut states = div_tree::<TestBackend>(&device, ids.clone());

        let expected_ta: Tensor<TestBackend, 2> =
            Tensor::<TestBackend, 2>::from_data([[-0.4444, -1.3328], [-0.78125, -0.5555]], &device);
        let obtained_ta: Tensor<TestBackend, 2> =
            downcast_tensor::<TestBackend, 2>(states.get(ids[6].clone()));
        expected_ta
            .to_data()
            .assert_approx_eq(&obtained_ta.to_data(), 3);
    }

    fn downcast_tensor<B: Backend, const D: usize>(state_content: &StateContent) -> Tensor<B, D> {
        state_content
            .downcast_ref::<Tensor<B, D>>()
            .unwrap()
            .clone()
    }
}
