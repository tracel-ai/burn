use burn::{
    Tensor,
    prelude::{Backend, ToElement},
};
use burn_rl::{Environment, StepResult};
use gym_rs::{
    core::Env,
    envs::classical_control::cartpole::{CartPoleEnv, CartPoleObservation},
};

#[derive(Clone)]
pub struct CartPoleAction {
    action: usize,
}

impl<B: Backend> From<Tensor<B, 2>> for CartPoleAction {
    fn from(value: Tensor<B, 2>) -> Self {
        Self {
            action: value.int().into_scalar().to_usize(),
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

impl<B: Backend> Into<Tensor<B, 2>> for CartPoleState {
    fn into(self) -> Tensor<B, 2> {
        Tensor::<B, 1>::from_floats(self.state, &Default::default()).unsqueeze()
    }
}

#[derive(Clone)]
pub struct CartPoleWrapper {
    gym_env: CartPoleEnv,
    step_index: usize,
}

impl Environment for CartPoleWrapper {
    type State = CartPoleState;
    type Action = CartPoleAction;

    const MAX_STEPS: usize = 500;
    const OBS_SPACE: usize = 4;
    const ACTION_SPACE: usize = 2;

    fn new() -> Self {
        Self {
            gym_env: CartPoleEnv::new(gym_rs::utils::renderer::RenderMode::None),
            step_index: 0,
        }
    }

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
