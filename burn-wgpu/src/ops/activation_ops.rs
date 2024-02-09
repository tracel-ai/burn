use crate::{GpuBackend, JitRuntime};
use burn_tensor::ops::ActivationOps;

impl<B: JitRuntime> ActivationOps<Self> for GpuBackend<B> {}
