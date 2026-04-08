use burn::rl::{Environment, StepResult};
use burn::{
    Tensor,
    prelude::{Backend, ToElement},
};

use crate::agent::{DiscreteActionTensor, ObservationTensor};

#[derive(Clone)]
pub struct CartPoleAction {
    action: usize,
}

impl<B: Backend> From<DiscreteActionTensor<B, 2>> for CartPoleAction {
    fn from(value: DiscreteActionTensor<B, 2>) -> Self {
        Self {
            action: value.actions.int().into_scalar().to_usize(),
        }
    }
}

impl<B: Backend> From<CartPoleAction> for DiscreteActionTensor<B, 2> {
    fn from(value: CartPoleAction) -> Self {
        DiscreteActionTensor {
            actions: Tensor::<B, 1>::from_data([value.action], &Default::default()).unsqueeze(),
        }
    }
}

#[derive(Clone)]
pub struct CartPoleState {
    pub state: [f64; 4],
}

impl CartPoleState {
    fn new(state: [f64; 4]) -> Self {
        Self { state }
    }
}

impl<B: Backend> From<CartPoleState> for ObservationTensor<B, 2> {
    fn from(val: CartPoleState) -> Self {
        ObservationTensor {
            state: Tensor::<B, 1>::from_floats(val.state, &Default::default()).unsqueeze(),
        }
    }
}

#[derive(Clone)]
pub struct CartPoleWrapper {
    state: [f64; 4],
    step_index: usize,
}

impl Default for CartPoleWrapper {
    fn default() -> Self {
        Self::new()
    }
}

impl CartPoleWrapper {
    pub fn new() -> Self {
        Self {
            state: [0.0, 0.0, 0.0, 0.0],
            step_index: 0,
        }
    }
}

impl Environment for CartPoleWrapper {
    type State = CartPoleState;
    type Action = CartPoleAction;

    const MAX_STEPS: usize = 500;

    fn state(&self) -> Self::State {
        CartPoleState::new(self.state)
    }

    fn step(&mut self, _action: Self::Action) -> StepResult<Self::State> {
        self.step_index += 1;

        self.state[0] += 0.01;
        self.state[1] += 0.01;
        self.state[2] += 0.01;
        self.state[3] += 0.01;

        StepResult {
            next_state: CartPoleState::new(self.state),
            reward: 1.0,
            done: false,
            truncated: self.step_index >= Self::MAX_STEPS,
        }
    }

    fn reset(&mut self) {
        self.state = [0.0, 0.0, 0.0, 0.0];
        self.step_index = 0;
    }
}