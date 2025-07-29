use burn_collective::CollectiveConfig;

use crate::{Learner, components::LearnerComponents};

/// Wrapper for a learner that implements a Distributed Data Parallel, using collective operations
pub struct DdpLearner<LC: LearnerComponents> {
    pub(crate) inner: Learner<LC>,
    pub(crate) collective_config: CollectiveConfig,
}

impl<LC: LearnerComponents> DdpLearner<LC> {
    /// Constructs a new learner that uses DDP
    pub fn new(inner: Learner<LC>, config: CollectiveConfig) -> Self {
        Self {
            inner,
            collective_config: config,
        }
    }
}
