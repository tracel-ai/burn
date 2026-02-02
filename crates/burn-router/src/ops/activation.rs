use crate::{BackendRouter, RunnerChannel};
use burn_backend::ops::ActivationOps;

impl<R: RunnerChannel> ActivationOps<Self> for BackendRouter<R> {}
