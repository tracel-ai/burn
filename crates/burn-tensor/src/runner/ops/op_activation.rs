use crate::ops::ActivationOps;
use crate::runner::{BackendRouter, RunnerChannel};

impl<C: RunnerChannel> ActivationOps<Self> for BackendRouter<C> {}
