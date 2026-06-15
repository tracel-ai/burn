use alloc::boxed::Box;
use burn_backend::{Backend, TensorMetadata, TensorPrimitive, tensor::FloatTensor};
use burn_std::tensor::container::TensorContainer;

use crate::{
    NodeId,
    graph::{NodeRef, Requirement},
    tensor::AutodiffTensor,
};

#[cfg(feature = "std")]
use crate::collections::HashMap;
#[cfg(feature = "std")]
use burn_backend::distributed::DistributedParams;

/// Gradient identifier.
pub type GradID = u64;

#[cfg(feature = "std")]
#[derive(Clone)]
pub(crate) struct GradSyncContext {
    pub n_required_map: HashMap<NodeId, usize>,
    pub distributed_params: HashMap<NodeId, DistributedParams>,
}

/// Hook type executed when a gradient is registered.
type OnRegisterHook = Box<dyn FnMut(&NodeId, &mut TensorContainer<GradID>) + Send + Sync>;

/// Trait for registering distributed gradients.
pub trait DistributedRegistration: Send + Sync {
    /// Performs distributed registration operations on the tensor with the corresponding [`NodeId`].
    fn on_register(&mut self, node_id: &NodeId, container: &mut TensorContainer<GradID>);
}

#[derive(Default)]
pub(crate) enum BackwardMode {
    #[default]
    Standard,
    // Distributed registration hook.
    #[cfg(feature = "std")]
    Distributed(Box<dyn FnOnce(GradSyncContext) -> Box<dyn DistributedRegistration>>),
}

/// Gradients container used during the backward pass.
pub struct Gradients {
    container: TensorContainer<GradID>,
    /// Optional hook called after each gradient is registered, used to trigger
    /// distributed gradient synchronization operations.
    on_register: Option<OnRegisterHook>,
}

impl Gradients {
    /// Creates a new gradients container.
    pub fn new<B: Backend>(root_node: NodeRef, root_tensor: FloatTensor<B>) -> Self {
        Self::new_with_hook::<B>(root_node, root_tensor, None)
    }

    /// Creates a new gradients container.
    fn new_with_hook<B: Backend>(
        root_node: NodeRef,
        root_tensor: FloatTensor<B>,
        on_register: Option<OnRegisterHook>,
    ) -> Self {
        let mut gradients = Self {
            container: TensorContainer::new(),
            on_register,
        };
        gradients.register::<B>(
            root_node.id,
            B::float_ones(
                root_tensor.shape(),
                &B::float_device(&root_tensor),
                root_tensor.dtype().into(),
            ),
        );
        gradients
    }

    /// Creates a new gradients container with a registration hook for distributed gradients.
    #[cfg(feature = "std")]
    pub fn new_distributed<B: Backend>(
        root_node: NodeRef,
        root_tensor: FloatTensor<B>,
        mut reg: Box<dyn DistributedRegistration>,
    ) -> Self {
        let on_register: Option<OnRegisterHook> = Some(Box::new(move |id, container| {
            reg.on_register(id, container);
        }));
        Self::new_with_hook::<B>(root_node, root_tensor, on_register)
    }

    /// Consumes the gradients for a given tensor.
    ///
    /// Each tensor should be consumed exactly 1 time if its gradients are only required during the
    /// backward pass, otherwise, it may be consume multiple times.
    pub fn consume<B: Backend>(&mut self, node: &NodeRef) -> FloatTensor<B> {
        match node.requirement {
            Requirement::Grad => self
                .container
                .get::<TensorPrimitive<B>>(&node.id.value)
                .map(|tensor| tensor.tensor())
                .expect("Can't consume the gradients before they are registered at least once."),
            Requirement::GradInBackward => self
                .container
                .remove::<TensorPrimitive<B>>(&node.id.value)
                .map(|tensor| tensor.tensor())
                .expect("Can't consume the gradients before they are registered at least once."),
            Requirement::None => panic!("Trying to consume the gradients for an untracked tensor"),
        }
    }

    /// Removes a grad tensor from the container.
    pub fn remove<B: Backend>(&mut self, tensor: &AutodiffTensor<B>) -> Option<FloatTensor<B>> {
        self.container
            .remove::<TensorPrimitive<B>>(&tensor.node.id.value)
            .map(|tensor| tensor.tensor())
    }

    /// Gets a grad tensor from the container.
    pub fn get<B: Backend>(&self, tensor: &AutodiffTensor<B>) -> Option<FloatTensor<B>> {
        self.container
            .get::<TensorPrimitive<B>>(&tensor.node.id.value)
            .map(|tensor| tensor.tensor())
    }

    /// Register a grad tensor in the container.
    ///
    /// If the tensor already exists, add both tensors together before saving the result.
    ///
    /// If the registered tensor is distributed, launches a syncing operation on the gradients.
    pub fn register<B: Backend>(&mut self, node_id: NodeId, value: FloatTensor<B>) {
        let out =
            if let Some(tensor_old) = self.container.remove::<TensorPrimitive<B>>(&node_id.value) {
                B::float_add(value, tensor_old.tensor())
            } else {
                value
            };

        self.container
            .register::<TensorPrimitive<B>>(node_id.value, TensorPrimitive::Float(out));

        if let Some(hook) = &mut self.on_register {
            hook(&node_id, &mut self.container);
        }
    }
}
