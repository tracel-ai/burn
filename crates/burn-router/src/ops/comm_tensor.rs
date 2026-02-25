use burn_backend::ops::CommunicationTensorOps;

use crate::{BackendRouter, RunnerChannel};

impl<R: RunnerChannel> CommunicationTensorOps<Self> for BackendRouter<R> {}
