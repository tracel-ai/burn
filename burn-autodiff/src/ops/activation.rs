use crate::ADBackendDecorator;
use burn_tensor::{backend::Backend, ops::ActivationOps};

impl<B: Backend> ActivationOps<ADBackendDecorator<B>> for ADBackendDecorator<B> {}
