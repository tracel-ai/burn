use crate::{graph::FusedBackend, FusionBackend};
use burn_tensor::ops::ActivationOps;

impl<B: FusedBackend> ActivationOps<Self> for FusionBackend<B> {}
