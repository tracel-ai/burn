use burn::rl::{Environment, StepResult, ToAction, ToObservation};
use burn::tensor::Device;
use burn::{Tensor, prelude::ToElement};
use gym_rs::{
    core::Env,
    envs::classical_control::cartpole::{CartPoleEnv, CartPoleObservation},
};

use crate::agent::{DiscreteActionTensor, ObservationTensor};

#[derive(Clone)]
pub struct CartPoleAction {
    action: usize,
}

impl From<DiscreteActionTensor<2>> for CartPoleAction {
    fn from(value: DiscreteActionTensor<2>) -> Self {
        Self {
            action: value.actions.int().into_scalar::<i32>().to_usize(),
        }
    }
}

impl ToAction<DiscreteActionTensor<2>> for CartPoleAction {
    fn to_action(&self, device: &Device) -> DiscreteActionTensor<2> {
        DiscreteActionTensor {
            actions: Tensor::<1>::from_data([self.action], device).unsqueeze(),
        }
    }
}

#[derive(Clone)]
pub struct CartPoleState {
    pub state: [f64; 4],
}

impl From<CartPoleObservation> for CartPoleState {
    fn from(observation: CartPoleObservation) -> Self {
        let vec = Vec::<f64>::from(observation);
        Self {
            state: [vec[0], vec[1], vec[2], vec[3]],
        }
    }
}

impl ToObservation<ObservationTensor<2>> for CartPoleState {
    fn to_observation(&self, device: &Device) -> ObservationTensor<2> {
        ObservationTensor {
            state: Tensor::<1>::from_floats(self.state, device).unsqueeze(),
        }
    }
}

#[derive(Clone)]
pub struct CartPoleWrapper {
    gym_env: CartPoleEnv,
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
            gym_env: CartPoleEnv::new(gym_rs::utils::renderer::RenderMode::None),
            step_index: 0,
        }
    }
}

impl Environment for CartPoleWrapper {
    type State = CartPoleState;
    type Action = CartPoleAction;

    const MAX_STEPS: usize = 500;

    fn state(&self) -> Self::State {
        CartPoleState::from(self.gym_env.state)
    }

    fn step(&mut self, action: Self::Action) -> StepResult<Self::State> {
        let action_reward = self.gym_env.step(action.action);
        self.step_index += 1;
        StepResult {
            next_state: CartPoleState::from(action_reward.observation),
            reward: action_reward.reward.into_inner(),
            done: action_reward.done,
            truncated: self.step_index >= Self::MAX_STEPS,
        }
    }

    fn reset(&mut self) {
        self.gym_env.reset(None, false, None);
        self.step_index = 0;
    }
}
