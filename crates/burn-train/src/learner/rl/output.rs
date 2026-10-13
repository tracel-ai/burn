use crate::{
    ItemLazy,
    metric::{Adaptor, CumulativeRewardInput, EpisodeLengthInput},
};
use burn_std::ExecutionError;

/// Summary of an episode.
pub struct EpisodeSummary {
    /// The total length of the episode.
    pub episode_length: usize,
    /// The final cumulative reward.
    pub cum_reward: f64,
}

impl ItemLazy for EpisodeSummary {
    fn sync(self) -> Result<Self, ExecutionError> {
        Ok(self)
    }
}

impl Adaptor<EpisodeLengthInput> for EpisodeSummary {
    fn adapt(&self) -> EpisodeLengthInput {
        EpisodeLengthInput::new(self.episode_length as f64)
    }
}

impl Adaptor<CumulativeRewardInput> for EpisodeSummary {
    fn adapt(&self) -> CumulativeRewardInput {
        CumulativeRewardInput::new(self.cum_reward)
    }
}
