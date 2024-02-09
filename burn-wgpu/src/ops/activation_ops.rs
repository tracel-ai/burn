use crate::{JitBackend, Runtime};
use burn_tensor::ops::ActivationOps;

impl<R: Runtime> ActivationOps<Self> for JitBackend<R> {}
