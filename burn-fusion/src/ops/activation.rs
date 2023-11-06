use crate::{FusedBackend, Fusion};
use burn_tensor::ops::ActivationOps;

impl<B: FusedBackend> ActivationOps<Self> for Fusion<B> {}
