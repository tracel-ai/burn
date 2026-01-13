use crate::metric::{MetricAdaptor, EpisodeLengthInput, EpisodeLengthMetric, processor::ItemLazy};

/// Simple classification output adapted for multiple metrics.
#[derive(new)]
pub struct EpisodeOuput {
    /// The loss.
    pub run_num: usize,

    /// The output.
    pub reward: f64,

    /// The targets.
    pub episode_len: usize,
}

impl ItemLazy for EpisodeOuput {
    type ItemSync = EpisodeOuput;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl MetricAdaptor<EpisodeLengthMetric> for EpisodeOuput {
    fn adapt(&self) -> EpisodeLengthInput {
        EpisodeLengthInput {
            ep_len: self.episode_len as f64,
        }
    }
}
