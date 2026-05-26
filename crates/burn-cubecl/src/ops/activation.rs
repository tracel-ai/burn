use crate::{CubeBackend, CubeRuntime};
use burn_backend::ops::ActivationOps;

impl<R: CubeRuntime> ActivationOps<Self> for CubeBackend<R> {}
