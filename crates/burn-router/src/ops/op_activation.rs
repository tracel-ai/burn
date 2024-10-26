use crate::{BackendRouter, RunnerChannel};
use burn_tensor::ops::ActivationOps;

impl<R: RunnerChannel> ActivationOps<Self> for BackendRouter<R> {}
