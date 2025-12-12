use crate::{Fusion, FusionBackend};
use burn_backend::ops::ActivationOps;

impl<B: FusionBackend> ActivationOps<Self> for Fusion<B> {}
