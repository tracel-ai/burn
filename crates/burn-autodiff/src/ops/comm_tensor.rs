use burn_backend::{Backend, ops::CommunicationTensorOps};

use crate::{Autodiff, checkpoint::strategy::CheckpointStrategy};

// TODO: Do I need to?
impl<B: Backend, C: CheckpointStrategy> CommunicationTensorOps<Self> for Autodiff<B, C> {}
