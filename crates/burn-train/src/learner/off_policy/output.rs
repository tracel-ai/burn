use crate::{
    ItemLazy,
    metric::{Adaptor, EpisodeLengthInput},
};

pub struct EpisodeSummary {
    pub episode_length: usize,
    pub total_reward: f64,
}

impl ItemLazy for EpisodeSummary {
    type ItemSync = EpisodeSummary;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl Adaptor<EpisodeLengthInput> for EpisodeSummary {
    fn adapt(&self) -> EpisodeLengthInput {
        EpisodeLengthInput {
            ep_len: self.episode_length as f64,
        }
    }
}
