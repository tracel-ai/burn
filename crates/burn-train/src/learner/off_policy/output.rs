use crate::{
    ItemLazy,
    metric::{Adaptor, CumulativeRewardInput, EpisodeLengthInput},
};

/// Summary of an episode.
pub struct EpisodeSummary {
    /// The total length of the episode.
    pub episode_length: usize,
    /// The final cumulative reward.
    pub cum_reward: f64,
}

impl ItemLazy for EpisodeSummary {
    type ItemSync = EpisodeSummary;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl Adaptor<EpisodeLengthInput> for EpisodeSummary {
    fn adapt(&self) -> EpisodeLengthInput {
        EpisodeLengthInput::new(self.episode_length as f64)
    }
}

impl Adaptor<CumulativeRewardInput> for EpisodeSummary {
    fn adapt(&self) -> CumulativeRewardInput {
        CumulativeRewardInput::new(self.cum_reward as f64)
    }
}
