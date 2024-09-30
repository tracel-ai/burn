use crate::ops::ActivationOps;
use crate::router::{BackendRouter, RunnerChannel};

impl<R: RunnerChannel> ActivationOps<Self> for BackendRouter<R> {}
