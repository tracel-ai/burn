use burn_core::{Tensor, prelude::Backend};
use derive_new::new;
use rand::{rng, seq::index::sample};

// TODO : Probably should define an associated type for each agent defining what should be returned in a transition.
// Like an adaptor that takes all those outputs but only stores the ones useful to the learning agent in memory.
#[derive(Clone, new)]
pub struct Transition<B: Backend> {
    pub state: Tensor<B, 1>,
    pub next_state: Tensor<B, 1>,
    pub action: Tensor<B, 1>,
    pub reward: Tensor<B, 1>,
    pub done: Tensor<B, 1>,
    // pub prob: Tensor<B, 1>,
    // pub value: Option<Tensor<B, 1>>,
}

pub struct TransitionBatch<B: Backend> {
    pub states: Tensor<B, 2>,
    pub next_states: Tensor<B, 2>,
    pub actions: Tensor<B, 2>,
    pub rewards: Tensor<B, 2>,
    pub dones: Tensor<B, 2>,
}

impl<B: Backend> From<Vec<&Transition<B>>> for TransitionBatch<B> {
    fn from(value: Vec<&Transition<B>>) -> Self {
        let states: Vec<_> = value.iter().map(|t| t.state.clone()).collect();
        let next_states: Vec<_> = value.iter().map(|t| t.next_state.clone()).collect();
        let actions: Vec<_> = value.iter().map(|t| t.action.clone()).collect();
        let rewards: Vec<_> = value.iter().map(|t| t.reward.clone()).collect();
        let dones: Vec<_> = value.iter().map(|t| t.done.clone()).collect();

        Self {
            states: Tensor::stack(states, 0),
            next_states: Tensor::stack(next_states, 0),
            actions: Tensor::stack(actions, 0),
            rewards: Tensor::stack(rewards, 0),
            dones: Tensor::stack(dones, 0),
        }
    }
}

pub struct TransitionBuffer<T> {
    buffer: Vec<T>,
    capacity: usize,
    cursor: usize,
}

impl<T> TransitionBuffer<T> {
    /// Creates a new circular buffer with a fixed capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            cursor: 0,
        }
    }

    /// Add an item, overwriting the oldest if full.
    pub fn push(&mut self, item: T) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(item);
        } else {
            self.buffer[self.cursor] = item;
            self.cursor = (self.cursor + 1) % self.capacity;
        }
    }

    /// Returns the current number of items stored.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    pub fn sample(&self, indices: Vec<usize>) -> Vec<&T> {
        let mut items = Vec::with_capacity(indices.len());

        for &idx in indices.iter() {
            if let Some(item) = self.buffer.get(idx) {
                items.push(item);
            }
        }
        items
    }

    pub fn random_sample(&self, batch_size: usize) -> Vec<&T> {
        assert!(batch_size <= self.len());
        let mut rng = rng();
        let indices = sample(&mut rng, self.len(), batch_size).into_vec();
        self.sample(indices)
    }
}

#[cfg(test)]
mod tests {
    use burn_core::tensor::TensorData;

    use crate::TestBackend;

    use super::*;

    fn transition() -> Transition<TestBackend> {
        Transition::new(
            Tensor::from_data(TensorData::from([1.0, 2.0]), &Default::default()),
            Tensor::from_data(TensorData::from([1.0, 2.0]), &Default::default()),
            Tensor::from_data(TensorData::from([1.0]), &Default::default()),
            Tensor::from_data(TensorData::from([1.0]), &Default::default()),
            Tensor::from_data(TensorData::from([1.0]), &Default::default()),
        )
    }

    #[test]
    fn len_returns_number_of_elements() {
        let mut buffer: TransitionBuffer<Transition<TestBackend>> = TransitionBuffer::new(2);
        assert_eq!(buffer.len(), 0);

        buffer.push(transition());
        assert_eq!(buffer.len(), 1);

        buffer.push(transition());
        assert_eq!(buffer.len(), 2);

        buffer.push(transition());
        assert_eq!(buffer.len(), 2)
    }
}
