use crate::{GpuBackend, Runtime};
use burn_tensor::ops::ActivationOps;

impl<R: Runtime> ActivationOps<Self> for GpuBackend<R> {}
