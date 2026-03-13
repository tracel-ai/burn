use crate::{
    checkpoint::{base::Checkpointer, builder::CheckpointerBuilder},
    grads::Gradients,
    graph::{ComputingProperty, Node, NodeId, NodeRef, Parent, Requirement, Step},
    runtime::{AutodiffClient, AutodiffClientImpl},
};
use alloc::{boxed::Box, sync::Arc, vec};
use burn_backend::{Backend, TensorMetadata};

#[derive(Debug, Clone)]
pub struct AutodiffTensor<B: Backend> {
    pub primitive: B::FloatTensorPrimitive,
    pub node: NodeRef,
    pub rc: NodeRefCount,
}

impl<B: Backend> TensorMetadata for AutodiffTensor<B> {
    fn dtype(&self) -> burn_std::DType {
        self.primitive.dtype()
    }

    fn shape(&self) -> burn_std::Shape {
        self.primitive.shape()
    }

    fn rank(&self) -> usize {
        self.primitive.rank()
    }
}

pub type NodeRefCount = Arc<NodeId>;

#[derive(new, Debug)]
pub(crate) struct RootStep {
    node: NodeRef,
}

impl Step for RootStep {
    fn step(self: Box<Self>, _grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        // Nothing to do
    }

    fn node(&self) -> NodeId {
        self.node.id
    }

    fn parents(&self) -> &[Parent] {
        &self.node.parents
    }

    fn depth(&self) -> usize {
        self.node.order
    }
}

impl<B: Backend> AutodiffTensor<B> {
    /// Create a new leaf tensor.
    pub fn new(primitive: B::FloatTensorPrimitive) -> Self {
        let id = NodeId::new();
        let node: NodeRef = Node::new(
            vec![],
            0,
            id,
            Requirement::None,
            ComputingProperty::Ambiguous,
            AutodiffClientImpl::new(),
        )
        .into();

        Self {
            rc: Arc::new(node.id),
            primitive,
            node: node.clone(),
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
                    self.node.id,
                    Requirement::Grad,
                    self.node.properties.clone(),
                    self.node.client.clone(),
                )
                .into();
                let step = RootStep::new(self.node.clone());

                self.register_step(step, CheckpointerBuilder::default())
            }
        }
    }

    /// Create a tensor from parent infos.
    pub fn from_parents(
        primitive: B::FloatTensorPrimitive,
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
            .unwrap_or_else(AutodiffClientImpl::new);

        let node: NodeRef = Node::new(
            parent_nodes
                .iter()
                .filter_map(|node| node.clone_if_require_grad())
                .map(|node| Parent::new(node.id))
                .collect(),
            order,
            NodeId::new(),
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
    ///
    /// # Warning
    ///
    /// This should be called only once per tensor.
    pub fn register_step<S: Step + 'static>(
        self,
        step_that_created_the_tensor: S,
        actions: CheckpointerBuilder,
    ) -> Self {
        self.node.client.register(
            self.rc.clone(),
            Box::new(step_that_created_the_tensor),
            actions,
        );
        self
    }

    pub fn into_primitive(self) -> B::FloatTensorPrimitive {
        self.primitive
    }

    pub fn backward(self) -> Gradients {
        let client = self.node.client.clone();

        AutodiffClient::backward::<B>(&client, self)
    }

    pub fn grad(&self, grads: &Gradients) -> Option<B::FloatTensorPrimitive> {
        grads.get::<B>(self)
    }

    pub fn grad_remove(&self, grads: &mut Gradients) -> Option<B::FloatTensorPrimitive> {
        grads.remove::<B>(self)
    }

    pub fn grad_replace(&self, grads: &mut Gradients, grad: B::FloatTensorPrimitive) {
        grads.remove::<B>(self);
        grads.register::<B>(self.node.id, grad);
    }
}
