use crate::FusionBackend;
use burn_tensor::{backend::Backend, ops::ActivationOps};

impl<B: Backend> ActivationOps<FusionBackend<B>> for FusionBackend<B> {}
