use super::state::MasterWeightState;
use crate::{LearningRate, Optimizer};
use burn_core::tensor::{Device, FloatDType, Tensor};

#[derive(Clone)]
pub(super) struct MasterWeights<O> {
    optimizer: O,
    dtype: FloatDType,
}

impl<O> MasterWeights<O> {
    pub(super) fn new(optimizer: O, dtype: FloatDType) -> Self {
        Self { optimizer, dtype }
    }
}

impl<O: Optimizer> Optimizer for MasterWeights<O> {
    type State<const D: usize> = MasterWeightState<O::State<D>, D>;

    fn prepare_gradient<const D: usize>(&self, grad: Tensor<D>) -> Tensor<D> {
        self.optimizer.prepare_gradient(grad)
    }

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        model: Tensor<D>,
        grad: Tensor<D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        let model_dtype = model.dtype();
        let (parameter, statistics) = match state {
            Some(state) => (state.master.unwrap_or(model), state.statistics),
            None => (model, None),
        };
        let (updated, statistics) =
            self.optimizer
                .step(lr, parameter.cast(self.dtype), grad, statistics);
        let master = (model_dtype != self.dtype.into()).then(|| updated.clone());
        let state = (statistics.is_some() || master.is_some())
            .then_some(MasterWeightState { statistics, master });
        (updated.cast(model_dtype), state)
    }

    fn to_device<const D: usize>(state: Self::State<D>, device: &Device) -> Self::State<D> {
        MasterWeightState {
            statistics: state.statistics.map(|s| O::to_device::<D>(s, device)),
            master: state.master.map(|m| m.to_device(device)),
        }
    }
}
