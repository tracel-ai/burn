use crate::{RecordState, StateSink, StateSource, optim::state::QUANTIZED_STORAGE_KEY};
use burn_core::tensor::{DType, Device, Tensor};

const MASTER_KEY: &str = "__precision.master";
const STATELESS_KEY: &str = "__precision.stateless";

#[derive(Clone)]
pub(super) struct MasterWeightState<S, const D: usize> {
    pub statistics: Option<S>,
    pub master: Option<Tensor<D>>,
}

impl<S: RecordState, const D: usize> RecordState for MasterWeightState<S, D> {
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

/// Keep the storage codec's exact dtypes through deserialization.
#[derive(Clone)]
pub(super) struct StoredState<S>(pub S);

impl<S: RecordState> RecordState for StoredState<S> {
    fn state_flatten(&self, prefix: &str, out: &mut StateSink) {
        let start = out.tensors.len();
        self.0.state_flatten(prefix, out);
        // Bare optimizers decode these statistics when importing a precision checkpoint.
        if out.tensors[start..]
            .iter()
            .any(|(_, data)| matches!(data.dtype(), DType::QFloat(_)))
        {
            out.push_scalar(prefix, QUANTIZED_STORAGE_KEY, true.into());
        }
    }
    fn state_unflatten(prefix: &str, src: &mut StateSource, device: &Device) -> Option<Self> {
        src.with_recorded_dtypes(|src| S::state_unflatten(prefix, src, device))
            .map(Self)
    }
}
