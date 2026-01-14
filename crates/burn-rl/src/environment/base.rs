use burn_core::prelude::*;

pub trait EnvState: Clone + Send {
    fn to_tensor<B: Backend>(&self, device: &Device<B>) -> Tensor<B, 1>;
}

pub trait EnvAction: Clone + Send {
    fn from_tensor<B: Backend>(tensor: Tensor<B, 2>) -> Self;
    fn from_usize(action: usize) -> Self;
    fn to_tensor<B: Backend>(&self, device: &Device<B>) -> Tensor<B, 1>;
}

pub struct StepResult<S: EnvState> {
    pub next_state: S,
    pub reward: f64,
    pub done: bool,
    pub truncated: bool,
}

pub trait Environment: Sized + Clone {
    type State: EnvState;
    type Action: EnvAction;

    const MAX_STEPS: usize;
    const OBS_SPACE: usize;
    const ACTION_SPACE: usize;

    // TODO: New could be removed in favor of letting the user pass a closure to the launch() fn of the ReinforcmentLearning (And remove sized constraint)
    fn new() -> Self;
    fn state(&self) -> Self::State;
    fn step(&mut self, action: Self::Action) -> StepResult<Self::State>;
    fn reset(&mut self);
}
