use burn_core::{Tensor, prelude::Backend, tensor::Distribution};
use derive_new::new;

use super::SliceAccess;

/// A state transition in an environment.
#[derive(Clone, new)]
pub struct Transition<B: Backend, S, A> {
    /// The initial state.
    pub state: S,
    /// The state after the step was taken.
    pub next_state: S,
    /// The action taken in the step.
    pub action: A,
    /// The reward.
    pub reward: Tensor<B, 1>,
    /// If the environment has reached a terminal state.
    pub done: Tensor<B, 1>,
}

/// A batch of transitions.
pub struct TransitionBatch<B: Backend, SB, AB> {
    /// Batched initial states.
    pub states: SB,
    /// Batched resulting states.
    pub next_states: SB,
    /// Batched actions.
    pub actions: AB,
    /// Batched rewards.
    pub rewards: Tensor<B, 2>,
    /// Batched flags for terminal states.
    pub dones: Tensor<B, 2>,
}

/// A tensor-backed circular buffer for transitions.
///
/// Uses [`SliceAccess`] to store state and action batches in contiguous
/// tensor storage, enabling efficient random sampling via `select`.
/// The buffer lazily initializes its storage on the first `push` call.
pub struct TransitionBuffer<B: Backend, SB: SliceAccess<B>, AB: SliceAccess<B>> {
    states: Option<SB>,
    next_states: Option<SB>,
    actions: Option<AB>,
    rewards: Option<Tensor<B, 2>>,
    dones: Option<Tensor<B, 2>>,
    capacity: usize,
    write_head: usize,
    len: usize,
    device: B::Device,
}

impl<B: Backend, SB: SliceAccess<B>, AB: SliceAccess<B>> TransitionBuffer<B, SB, AB> {
    /// Creates a new buffer. Storage is lazily allocated on the first `push`.
    pub fn new(capacity: usize, device: &B::Device) -> Self {
        Self {
            states: None,
            next_states: None,
            actions: None,
            rewards: None,
            dones: None,
            capacity,
            write_head: 0,
            len: 0,
            device: device.clone(),
        }
    }

    fn ensure_init(&mut self, state: &SB, next_state: &SB, action: &AB) {
        if self.states.is_none() {
            self.states = Some(SB::zeros_like(state, self.capacity, &self.device));
            self.next_states = Some(SB::zeros_like(next_state, self.capacity, &self.device));
            self.actions = Some(AB::zeros_like(action, self.capacity, &self.device));
            self.rewards = Some(Tensor::zeros([self.capacity, 1], &self.device));
            self.dones = Some(Tensor::zeros([self.capacity, 1], &self.device));
        }
    }

    /// Add a transition, overwriting the oldest if full.
    pub fn push(&mut self, state: SB, next_state: SB, action: AB, reward: f32, done: bool) {
        self.ensure_init(&state, &next_state, &action);

        let idx = self.write_head % self.capacity;

        self.states
            .as_mut()
            .unwrap()
            .slice_assign_inplace(idx, state);
        self.next_states
            .as_mut()
            .unwrap()
            .slice_assign_inplace(idx, next_state);
        self.actions
            .as_mut()
            .unwrap()
            .slice_assign_inplace(idx, action);

        let reward = Tensor::from_data([[reward]], &self.device);
        self.rewards
            .as_mut()
            .unwrap()
            .inplace(|r| r.slice_assign(idx..idx + 1, reward));

        let done_val = if done { 1.0f32 } else { 0.0 };
        let done = Tensor::from_data([[done_val]], &self.device);
        self.dones
            .as_mut()
            .unwrap()
            .inplace(|d| d.slice_assign(idx..idx + 1, done));

        self.write_head += 1;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Sample a random batch of transitions.
    pub fn sample(&self, batch_size: usize) -> TransitionBatch<B, SB, AB> {
        assert!(batch_size <= self.len, "batch_size exceeds buffer length");

        let indices = Tensor::<B, 1>::random(
            [batch_size],
            Distribution::Uniform(0.0, self.len as f64),
            &self.device,
        )
        .int();

        TransitionBatch {
            states: self
                .states
                .as_ref()
                .unwrap()
                .clone()
                .select(0, indices.clone()),
            next_states: self
                .next_states
                .as_ref()
                .unwrap()
                .clone()
                .select(0, indices.clone()),
            actions: self
                .actions
                .as_ref()
                .unwrap()
                .clone()
                .select(0, indices.clone()),
            rewards: self
                .rewards
                .as_ref()
                .unwrap()
                .clone()
                .select(0, indices.clone()),
            dones: self.dones.as_ref().unwrap().clone().select(0, indices),
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

    type TB = Tensor<TestBackend, 2>;

    fn push_transition(
        buffer: &mut TransitionBuffer<TestBackend, TB, TB>,
        device: &<TestBackend as Backend>::Device,
        val: f32,
    ) {
        let state = Tensor::<TestBackend, 2>::from_data([[val, val]], device);
        let next_state = Tensor::<TestBackend, 2>::from_data([[val + 1.0, val + 1.0]], device);
        let action = Tensor::<TestBackend, 2>::from_data([[val]], device);
        buffer.push(state, next_state, action, val, false);
    }

    #[test]
    fn push_increment_len() {
        let device = Default::default();
        let mut buffer = TransitionBuffer::<TestBackend, TB, TB>::new(5, &device);

        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());

        push_transition(&mut buffer, &device, 1.0);
        assert_eq!(buffer.len(), 1);

        push_transition(&mut buffer, &device, 2.0);
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn push_overwrites_when_full() {
        let device = Default::default();
        let mut buffer = TransitionBuffer::<TestBackend, TB, TB>::new(3, &device);

        for i in 0..5 {
            push_transition(&mut buffer, &device, i as f32);
        }

        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.capacity(), 3);
    }

    #[test]
    fn sample_returns_correct_shapes() {
        let device = Default::default();
        let mut buffer = TransitionBuffer::<TestBackend, TB, TB>::new(10, &device);

        for i in 0..5 {
            push_transition(&mut buffer, &device, i as f32);
        }

        let batch = buffer.sample(3);
        assert_eq!(batch.states.dims(), [3, 2]);
        assert_eq!(batch.next_states.dims(), [3, 2]);
        assert_eq!(batch.actions.dims(), [3, 1]);
        assert_eq!(batch.rewards.dims(), [3, 1]);
        assert_eq!(batch.dones.dims(), [3, 1]);
    }

    #[test]
    #[should_panic(expected = "batch_size exceeds buffer length")]
    fn sample_panics_when_batch_too_large() {
        let device = Default::default();
        let mut buffer = TransitionBuffer::<TestBackend, TB, TB>::new(5, &device);

        push_transition(&mut buffer, &device, 1.0);
        buffer.sample(5);
    }
}
