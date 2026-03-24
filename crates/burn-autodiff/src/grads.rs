use burn_backend::{
    Backend, DistributedParams, TensorMetadata, TensorPrimitive,
    ops::TensorRef,
    tensor::{FloatTensor, TensorContainer},
};

use crate::{
    NodeId,
    collections::HashMap,
    graph::{NodeRef, Requirement},
    tensor::AutodiffTensor,
};

/// Gradient identifier.
pub type GradID = u64;

/// Submits sync operations on gradient registrations.
#[derive(new)]
pub struct GradientSyncRegistration {
    n_required_map: HashMap<NodeId, usize>,
    sharded_parameters_map: HashMap<NodeId, DistributedParams>,
}

impl GradientSyncRegistration {
    pub(crate) fn on_register<B: Backend>(
        &mut self,
        id: &NodeId,
        container: &mut TensorContainer<GradID>,
    ) {
        if let Some(sharded_params) = self.sharded_parameters_map.get(id) {
            let n_required = self.n_required_map.get_mut(id).unwrap();
            *n_required -= 1;

            if *n_required == 0 {
                let tensor_ref = container.get_mut_ref::<B>(&id.value).unwrap();
                let tensor_ref = TensorRef(tensor_ref.get_mut_ref());
                B::submit_gradient_sync(tensor_ref, sharded_params.clone());
            }
        }
    }
}

/// Gradients container used during the backward pass.
pub struct Gradients {
    container: TensorContainer<GradID>,
    distributed: bool,
    gradient_sync_registration: Option<GradientSyncRegistration>,
}

impl Gradients {
    /// Creates a new gradients container.
    pub fn new<B: Backend>(
        root_node: NodeRef,
        root_tensor: FloatTensor<B>,
        gradient_sync_registration: Option<GradientSyncRegistration>,
    ) -> Self {
        let distributed = gradient_sync_registration.is_some();
        let mut gradients = Self {
            container: TensorContainer::new(),
            distributed,
            gradient_sync_registration,
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

    /// Consumes the gradients for a given tensor.
    ///
    /// Each tensor should be consumed exactly 1 time if its gradients are only required during the
    /// backward pass, otherwise, it may be consume multiple times.
    pub fn consume<B: Backend>(&mut self, node: &NodeRef) -> FloatTensor<B> {
        match node.requirement {
            Requirement::Grad => self
                .container
                .get::<B>(&node.id.value)
                .map(|tensor| tensor.tensor())
                .expect("Can't consume the gradients before they are registered at least once."),
            Requirement::GradInBackward => self
                .container
                .remove::<B>(&node.id.value)
                .map(|tensor| tensor.tensor())
                .expect("Can't consume the gradients before they are registered at least once."),
            Requirement::None => panic!("Trying to consume the gradients for an untracked tensor"),
        }
    }

    /// Removes a grad tensor from the container.
    pub fn remove<B: Backend>(&mut self, tensor: &AutodiffTensor<B>) -> Option<FloatTensor<B>> {
        self.container
            .remove::<B>(&tensor.node.id.value)
            .map(|tensor| tensor.tensor())
    }

    /// Gets a grad tensor from the container.
    pub fn get<B: Backend>(&self, tensor: &AutodiffTensor<B>) -> Option<FloatTensor<B>> {
        self.container
            .get::<B>(&tensor.node.id.value)
            .map(|tensor| tensor.tensor())
    }

    /// Register a grad tensor in the container.
    ///
    /// If the tensor already exists, add both tensors together before saving the result.
    ///
    /// If the registered tensor is distributed across multiple device, performs syncing operations on the gradients.
    pub fn register<B: Backend>(&mut self, node_id: NodeId, value: FloatTensor<B>) {
        let out = if let Some(tensor_old) = self.container.remove::<B>(&node_id.value) {
            B::float_add(value, tensor_old.tensor())
        } else {
            value
        };

        self.container
            .register::<B>(node_id.value, TensorPrimitive::Float(out));
        if let Some(sync_registration) = self.gradient_sync_registration.as_mut() {
            sync_registration.on_register::<B>(&node_id, &mut self.container);
        }
    }

    /// For distributed models, waits for collective operations to be completed so the gradients are synced accross devices.
    pub fn sync_collective<B: Backend>(&self, device: &B::Device) {
        if self.distributed {
            B::submit_sync_collective(device);
        }
    }
}
