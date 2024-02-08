use burn_tensor::backend::Backend;

use crate::{
    checkpoint::base::Checkpointer,
    grads::Gradients,
    graph::{ComputingProperties, Graph, Node, NodeID, NodeRef, Requirement, Step},
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
            ComputingProperties::Ambiguous,
        )
        .into();

        let graph = Graph::new();

        Self {
            primitive,
            node,
            graph,
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
        output: B::FloatTensorPrimitive<D>,
        parent_nodes: &[NodeRef],
        parent_graphs: I,
        requirement: Requirement,
        compute_properties: ComputingProperties,
        checkpointer: Option<Checkpointer>,
    ) -> Self {
        let graph = parent_graphs
            .reduce(|acc, graph| acc.merge(graph))
            .unwrap_or_else(Graph::new);

        let graph = if let Some(checkpointer) = checkpointer {
            graph.merge_checkpointer(checkpointer)
        } else {
            graph
        };

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
            compute_properties,
        )
        .into();

        // TODO rm
        graph.print_checkpoint();

        Self {
            primitive: output,
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
