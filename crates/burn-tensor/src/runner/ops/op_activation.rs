use crate::ops::ActivationOps;
use crate::runner::{BackendRouter, MultiBackendRuntime};

impl<R: MultiBackendRuntime> ActivationOps<Self> for BackendRouter<R> {}
