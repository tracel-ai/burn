use crate::{Fusion, FusionBackend};
use burn_tensor::ops::ActivationOps;

impl<B: FusionBackend> ActivationOps<Self> for Fusion<B> {}
