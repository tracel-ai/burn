use burn_core::{Tensor, prelude::Backend, tensor::Distribution};

/// A state transition in an environment.
#[derive(Clone)]
pub struct Transition<B: Backend> {
    /// The initial state.
    pub state: Tensor<B, 1>,
    /// The state after the step was taken.
    pub next_state: Tensor<B, 1>,
    /// The action taken in the step.
    pub action: Tensor<B, 1>,
    /// The reward.
    pub reward: f32,
    /// If the environment has reached a terminal state.
    pub done: bool,
}

/// A batch of transitions.
pub struct TransitionBatch<B: Backend> {
    /// Batched initial states.
    pub states: Tensor<B, 2>,
    /// Batched resulting states.
    pub next_states: Tensor<B, 2>,
    /// Batched actions.
    pub actions: Tensor<B, 2>,
    /// Batched rewards.
    pub rewards: Tensor<B, 2>,
    /// Batched flags for terminal states.
    pub dones: Tensor<B, 2>,
}

/// A tensor-backed circular buffer for transitions.
pub struct TransitionBuffer<B: Backend> {
    states: Tensor<B, 2>,
    next_states: Tensor<B, 2>,
    actions: Tensor<B, 2>,
    rewards: Tensor<B, 2>,
    dones: Tensor<B, 2>,
    capacity: usize,
    write_head: usize,
    len: usize,
    device: B::Device,
}

impl<B: Backend> TransitionBuffer<B> {
    /// Creates a new buffer with pre-allocated tensors.
    pub fn new(capacity: usize, state_dim: usize, action_dim: usize, device: &B::Device) -> Self {
        Self {
            states: Tensor::zeros([capacity, state_dim], device),
            next_states: Tensor::zeros([capacity, state_dim], device),
            actions: Tensor::zeros([capacity, action_dim], device),
            rewards: Tensor::zeros([capacity, 1], device),
            dones: Tensor::zeros([capacity, 1], device),
            capacity,
            write_head: 0,
            len: 0,
            device: device.clone(),
        }
    }

    /// Add a transition, overwriting the oldest if full.
    pub fn push(&mut self, transition: Transition<B>) {
        let idx = self.write_head % self.capacity;

        self.states = self
            .states
            .clone()
            .slice_assign(idx..idx + 1, transition.state.unsqueeze_dim(0));
        self.next_states = self
            .next_states
            .clone()
            .slice_assign(idx..idx + 1, transition.next_state.unsqueeze_dim(0));
        self.actions = self
            .actions
            .clone()
            .slice_assign(idx..idx + 1, transition.action.unsqueeze_dim(0));
        self.rewards = self.rewards.clone().slice_assign(
            idx..idx + 1,
            Tensor::from_data([[transition.reward]], &self.device),
        );

        self.dones = self.dones.clone().slice_assign(
            idx..idx + 1,
            Tensor::from_data([[if transition.done { 1.0 } else { 0.0 }]], &self.device),
        );

        self.write_head += 1;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Sample a random batch of transitions.
    pub fn sample(&self, batch_size: usize) -> TransitionBatch<B> {
        assert!(batch_size <= self.len, "batch_size exceeds buffer length");

        let indices = Tensor::<B, 1>::random(
            [batch_size],
            Distribution::Uniform(0.0, self.len as f64),
            &self.device,
        )
        .int();

        TransitionBatch {
            states: self.states.clone().select(0, indices.clone()),
            next_states: self.next_states.clone().select(0, indices.clone()),
            actions: self.actions.clone().select(0, indices.clone()),
            rewards: self.rewards.clone().select(0, indices.clone()),
            dones: self.dones.clone().select(0, indices),
        }
    }

    /// Current number of stored transitions.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Buffer capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;

    fn make_transition(
        device: &<TestBackend as Backend>::Device,
        val: f32,
    ) -> Transition<TestBackend> {
        Transition {
            state: Tensor::from_data([val, val], device),
            next_state: Tensor::from_data([val + 1.0, val + 1.0], device),
            action: Tensor::from_data([val], device),
            reward: val,
            done: false,
        }
    }

    #[test]
    fn push_increment_len() {
        let device = Default::default();
        let mut buffer = TransitionBuffer::<TestBackend>::new(5, 2, 1, &device);

        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());

        buffer.push(make_transition(&device, 1.0));
        assert_eq!(buffer.len(), 1);

        buffer.push(make_transition(&device, 2.0));
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn push_overwrites_when_full() {
        let device = Default::default();
        let mut buffer = TransitionBuffer::<TestBackend>::new(3, 2, 1, &device);

        for i in 0..5 {
            buffer.push(make_transition(&device, i as f32));
        }

        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.capacity(), 3);
    }

    #[test]
    fn sample_returns_correct_shapes() {
        let device = Default::default();
        let mut buffer = TransitionBuffer::<TestBackend>::new(10, 2, 1, &device);

        for i in 0..5 {
            buffer.push(make_transition(&device, i as f32));
        }

        let batch = buffer.sample(3);
        assert_eq!(batch.states.shape().dims, [3, 2]);
        assert_eq!(batch.next_states.shape().dims, [3, 2]);
        assert_eq!(batch.actions.shape().dims, [3, 1]);
        assert_eq!(batch.rewards.shape().dims, [3, 1]);
        assert_eq!(batch.dones.shape().dims, [3, 1]);
    }

    #[test]
    #[should_panic(expected = "batch_size exceeds buffer length")]
    fn sample_panics_when_batch_too_large() {
        let device = Default::default();
        let mut buffer = TransitionBuffer::<TestBackend>::new(5, 2, 1, &device);

        buffer.push(make_transition(&device, 1.0));
        buffer.sample(5);
    }
}
