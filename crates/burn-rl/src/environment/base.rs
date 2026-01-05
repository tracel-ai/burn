use burn_core::prelude::*;

pub trait EnvState: Clone + Send {
    fn to_tensor<B: Backend>(&self, device: &Device<B>) -> Tensor<B, 1>;
}

pub trait EnvAction: Clone + Send {
    fn from_tensor<B: Backend>(tensor: Tensor<B, 2>) -> Self;
    fn from_usize(action: usize) -> Self;
    fn to_tensor<B: Backend>(&self, device: &Device<B>) -> Tensor<B, 1>;
}

pub struct StepResult<E: Environment + Sized> {
    pub next_state: E::State,
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

    fn new() -> Self;
    fn state(&self) -> Self::State;
    fn step(&mut self, action: Self::Action) -> StepResult<Self>;
    fn reset(&mut self);
}
