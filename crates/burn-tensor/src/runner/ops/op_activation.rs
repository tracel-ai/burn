use crate::runner::Runner;
use crate::{ops::ActivationOps, runner::RunnerBackend};

impl<B: RunnerBackend> ActivationOps<Self> for Runner<B> {}
