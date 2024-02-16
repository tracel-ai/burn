use burn_tensor::backend::Backend;

use crate::{
    checkpoint::base::Checkpointer,
    grads::Gradients,
    graph::{
        CheckpointingActions, ComputingProperty, Graph, Node, NodeID, NodeRef, Requirement, Step,
    },
};

#[derive(Debug, Clone)]
pub struct AutodiffTensor<B: Backend, const D: usize> {
    pub primitive: B::FloatTensorPrimitive<D>,
    pub node: NodeRef,
    pub graph: Graph,
}

#[derive(new, Debug)]
struct RootStep {
    node: NodeRef,
}

impl Step for RootStep {
    fn step(self: Box<Self>, _grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        // Nothing to do
    }

    fn node(&self) -> NodeRef {
        self.node.clone()
    }
}

impl<B: Backend, const D: usize> AutodiffTensor<B, D> {
    /// Create a new leaf tensor.
    pub fn new(primitive: B::FloatTensorPrimitive<D>) -> Self {
        let id = NodeID::new();
        let node: NodeRef = Node::new(
            vec![],
            0,
            id,
            Requirement::None,
            ComputingProperty::Ambiguous,
        )
        .into();

        Self {
            primitive,
            node,
            graph: Graph::new(),
        }
    }

    pub fn is_tracked(&self) -> bool {
        !self.node.requirement.is_none()
    }

    /// Mark the tensor as requirering gradients.
    ///
    /// # Panics
    ///
    /// It panics if the tensor is non a leaf.
    pub fn require_grad(mut self) -> Self {
        match self.node.requirement {
            Requirement::Grad => self,
            Requirement::GradInBackward => {
                panic!("Can't convert a non leaf tensor into a tracked tensor")
            }
            Requirement::None => {
                self.node = Node::new(
                    vec![],
                    0,
                    self.node.id.clone(),
                    Requirement::Grad,
                    self.node.properties.clone(),
                )
                .into();
                let ops = RootStep::new(self.node.clone());

                self.register_step(ops)
            }
        }
    }

    /// Create a tensor from parent infos.
    pub fn from_parents<I: Iterator<Item = Graph>>(
        primitive: B::FloatTensorPrimitive<D>,
        parent_nodes: &[NodeRef],
        parent_graphs: I,
        requirement: Requirement,
        computing_properties: ComputingProperty,
        checkpointing_actions: CheckpointingActions,
    ) -> Self {
        let graph = parent_graphs
            .reduce(|acc, graph| acc.merge(graph))
            .unwrap_or_else(Graph::new);

        graph.extend_checkpointing_actions(checkpointing_actions);

        let order = parent_nodes
            .iter()
            .map(|node| node.order)
            .reduce(usize::max)
            .unwrap_or(0)
            + 1;

        let node: NodeRef = Node::new(
            parent_nodes.iter().map(|node| node.id.clone()).collect(),
            order,
            NodeID::new(),
            requirement,
            computing_properties,
        )
        .into();

        Self {
            primitive,
            node,
            graph,
        }
    }

    /// Register a step into a graph for that tensor.
    pub fn register_step<O: Step + 'static>(mut self, ops: O) -> Self {
        self.graph = self.graph.register(&self.node.id, Box::new(ops));
        self
    }
}
