use crate::{
    checkpoint::builder::CheckpointerBuilder,
    graph::{ComputingProperty, Node, NodeId, NodeRef, Parent, Requirement, Step},
    ops::NodeGuard,
    runtime::{AutodiffClient, AutodiffClientImpl},
    tensor::NodeRefCount,
};
use alloc::{boxed::Box, vec};
use burn_backend::{BackendTypes, TensorMetadata};
use burn_std::sync::Arc;

/// A quantized backend tensor with autodiff ownership: the packed values, and the graph node
/// gradients reach them through.
///
/// A gradient is always float. Quantizing and decoding pass the gradient through unchanged — a
/// straight-through estimator — so the rounding is never differentiated, and the operations
/// between them differentiate as their float form does.
#[derive(Debug, Clone)]
pub struct AutodiffQTensor<B: BackendTypes> {
    pub(crate) primitive: B::QuantizedTensorPrimitive,
    pub(crate) node: NodeRef,
    pub(crate) rc: NodeRefCount,
}

impl<B: BackendTypes> AutodiffQTensor<B> {
    /// A leaf: packed values no gradient reaches.
    pub fn new(primitive: B::QuantizedTensorPrimitive) -> Self {
        let node: NodeRef = Node::new(
            vec![],
            0,
            NodeId::new(),
            Requirement::None,
            ComputingProperty::Ambiguous,
            AutodiffClientImpl::new(),
            None,
        )
        .into();

        Self {
            rc: Arc::new(node.id),
            primitive,
            node,
        }
    }

    /// Returns a guard that keeps this node available while preparing a dependent operation.
    pub fn node(&self) -> NodeGuard {
        NodeGuard::new(self.node.clone(), self.rc.clone())
    }

    /// Whether gradients reach this tensor.
    pub fn is_tracked(&self) -> bool {
        !self.node.requirement.is_none()
    }

    /// Consumes the tensor, releasing its autodiff reference.
    pub fn into_primitive(self) -> B::QuantizedTensorPrimitive {
        self.primitive
    }

    /// The output of an operation over `parent_nodes`.
    pub(crate) fn from_parents(
        primitive: B::QuantizedTensorPrimitive,
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
                .map(|node| Parent::new(node.id, node.order == 0))
                .collect(),
            order,
            NodeId::new(),
            requirement,
            computing_properties,
            client,
            None,
        )
        .into();

        Self {
            rc: Arc::new(node.id),
            primitive,
            node,
        }
    }

    /// Register the step that created this tensor; once per tensor, its input guards alive
    /// until this returns.
    pub(crate) fn register_step<S: Step + 'static>(
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
}

impl<B: BackendTypes> TensorMetadata for AutodiffQTensor<B> {
    type Device = B::Device;

    fn dtype(&self) -> burn_std::DType {
        self.primitive.dtype()
    }

    fn shape(&self) -> burn_std::Shape {
        self.primitive.shape()
    }

    fn rank(&self) -> usize {
        self.primitive.rank()
    }

    fn device(&self) -> B::Device {
        self.primitive.device()
    }

    fn can_mut(&self) -> bool {
        self.primitive.can_mut()
    }
}
