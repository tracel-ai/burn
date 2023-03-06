use burn_tensor::backend::Backend;

use crate::{
    grads::Gradients,
    graph::{
        Node, NodeID, NodeRef, Requirement, {Graph, Step},
    },
    ADBackendDecorator,
};

#[derive(Debug, Clone)]
pub struct ADTensor<B: Backend, const D: usize> {
    pub primitive: B::TensorPrimitive<D>,
    pub node: NodeRef,
    pub graph: Graph,
}

pub type FloatElem<B> = <ADBackendDecorator<B> as Backend>::FloatElem;
pub type BoolTensor<B, const D: usize> = <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D>;
pub type IntTensor<B, const D: usize> = <ADBackendDecorator<B> as Backend>::IntTensorPrimitive<D>;

#[derive(new, Debug)]
struct RootStep {
    node: NodeRef,
}

impl Step for RootStep {
    fn step(self: Box<Self>, _grads: &mut Gradients) {
        // Nothing to do
    }

    fn node(&self) -> NodeRef {
        self.node.clone()
    }
}

impl<B: Backend, const D: usize> ADTensor<B, D> {
    /// Create a new leaf tensor.
    pub fn new(primitive: B::TensorPrimitive<D>) -> Self {
        let id = NodeID::new();
        let node = Node::new(vec![], 0, id, Requirement::None);

        Self {
            primitive,
            node: node.into(),
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
                self.node = Node::new(vec![], 0, self.node.id.clone(), Requirement::Grad).into();
                let ops = RootStep::new(self.node.clone());

                self.register_step(ops)
            }
        }
    }

    /// Create a tensor from parent infos.
    pub fn from_parents<I: Iterator<Item = Graph>>(
        output: B::TensorPrimitive<D>,
        parent_nodes: &[NodeRef],
        parent_graphs: I,
        requirement: Requirement,
    ) -> Self {
        let graph = parent_graphs
            .reduce(|acc, graph| acc.merge(graph))
            .unwrap_or_else(Graph::new);

        let order = parent_nodes
            .iter()
            .map(|node| node.order)
            .reduce(usize::max)
            .unwrap_or(0)
            + 1;

        let node = Node::new(
            parent_nodes.iter().map(|node| node.id.clone()).collect(),
            order,
            NodeID::new(),
            requirement,
        );

        Self {
            primitive: output,
            node: node.into(),
            graph,
        }
    }

    /// Register a step into a graph for that tensor.
    pub fn register_step<O: Step + 'static>(mut self, ops: O) -> Self {
        self.graph = self.graph.register(&self.node.id, Box::new(ops));
        self
    }
}
