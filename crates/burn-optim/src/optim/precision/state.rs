use crate::{RecordState, StateSink, StateSource};
use burn_core::tensor::{Device, Tensor};

const MASTER_KEY: &str = "__precision.master";
const STATELESS_KEY: &str = "__precision.stateless";

#[derive(Clone)]
pub(super) struct PrecisionState<S, const D: usize> {
    pub statistics: Option<S>,
    pub master: Option<Tensor<D>>,
}

impl<S: RecordState, const D: usize> RecordState for PrecisionState<S, D> {
    fn state_flatten(&self, prefix: &str, out: &mut StateSink) {
        match &self.statistics {
            Some(state) => state.state_flatten(prefix, out),
            None => out.push_scalar(prefix, STATELESS_KEY, true.into()),
        }
        if let Some(master) = &self.master {
            out.push_tensor(prefix, MASTER_KEY, master.to_data());
        }
    }

    fn state_unflatten(prefix: &str, src: &mut StateSource, device: &Device) -> Option<Self> {
        src.with_recorded_dtypes(|src| Self::restore(prefix, src, device))
    }
}

impl<S: RecordState, const D: usize> PrecisionState<S, D> {
    fn restore(prefix: &str, src: &mut StateSource, device: &Device) -> Option<Self> {
        let master = src.take_tensor(prefix, MASTER_KEY).map(|data| {
            let dtype = data.dtype();
            Tensor::from_data(data, (device, dtype))
        });
        if src.take_scalar(prefix, STATELESS_KEY).is_some() {
            return master.map(|master| Self {
                master: Some(master),
                statistics: None,
            });
        }
        Some(Self {
            statistics: Some(S::state_unflatten(prefix, src, device)?),
            master,
        })
    }
}
