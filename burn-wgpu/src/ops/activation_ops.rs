use crate::{GpuBackend, JitGpuBackend};
use burn_tensor::ops::ActivationOps;

impl<B: JitGpuBackend> ActivationOps<Self> for GpuBackend<B> {}
