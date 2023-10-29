use crate::FusionBackend;
use burn_tensor::{backend::Backend, ops::ActivationOps};

impl<B: Backend> ActivationOps<Self> for FusionBackend<B> {}
