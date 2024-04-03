use std::sync::Arc;

use crate::{
    checkpoint::{base::Checkpointer, builder::CheckpointerBuilder},
    grads::Gradients,
    graph::{ComputingProperty, Node, NodeID, NodeRef, Requirement, Step},
    runtime::{AutodiffClient, MutexClient},
};
use burn_tensor::backend::Backend;

#[derive(Debug, Clone)]
pub struct AutodiffTensor<B: Backend, const D: usize> {
    pub primitive: B::FloatTensorPrimitive<D>,
    pub node: NodeRef,
    pub rc: NodeRefCount,
}

pub type NodeRefCount = Arc<NodeID>;

#[derive(new, Debug)]
struct RootStep {
    node: NodeRef,
}

impl Step for RootStep {
    fn step(self: Box<Self>, _grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        // Nothing to do
    }

    fn node(&self) -> NodeID {
        self.node.id.clone()
    }

    fn parents(&self) -> Vec<NodeID> {
        self.node.parents.clone()
    }

    fn order(&self) -> usize {
        self.node.order
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
            MutexClient,
        )
        .into();

        Self {
            rc: Arc::new(node.id),
            primitive,
            node,
        }
    }

    pub fn is_tracked(&self) -> bool {
        !self.node.requirement.is_none()
    }

    /// Mark the tensor as requiring gradients.
    ///
    /// # Panics
    ///
    /// It panics if the tensor is not a leaf.
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
                    self.node.client.clone(),
                )
                .into();
                let ops = RootStep::new(self.node.clone());

                self.register_step(ops, CheckpointerBuilder::default())
            }
        }
    }

    /// Create a tensor from parent infos.
    pub fn from_parents(
        primitive: B::FloatTensorPrimitive<D>,
        parent_nodes: &[NodeRef],
        requirement: Requirement,
        computing_properties: ComputingProperty,
    ) -> Self {
        let order = parent_nodes
            .iter()
            .map(|node| node.order)
            .reduce(usize::max)
            .unwrap_or(0)
            + 1;

        let client = parent_nodes
            .first()
            .map(|node| node.client.clone())
            .unwrap_or_else(|| MutexClient);

        let node: NodeRef = Node::new(
            parent_nodes.iter().map(|node| node.id.clone()).collect(),
            order,
            NodeID::new(),
            requirement,
            computing_properties,
            client,
        )
        .into();

        Self {
            rc: Arc::new(node.id),
            primitive,
            node,
        }
    }

    /// Register a step into a graph for that tensor.
    pub fn register_step<O: Step + 'static>(self, ops: O, actions: CheckpointerBuilder) -> Self {
        self.node
            .client
            .register(self.rc.clone(), Box::new(ops), actions);
        self
    }

    pub fn into_primitive(self) -> B::FloatTensorPrimitive<D> {
        self.primitive
    }
}
