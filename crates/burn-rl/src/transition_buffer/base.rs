use std::io::{Error, ErrorKind};

use burn_core::{Tensor, prelude::Backend, tensor::backend::AutodiffBackend};
use derive_new::new;
use rand::{rng, seq::index::sample};

use crate::Batchable;

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

impl<BT, B, S, A, SB, AB> From<Vec<&Transition<BT, S, A>>> for TransitionBatch<B, SB, AB>
where
    BT: Backend,
    B: AutodiffBackend,
    S: Into<SB> + Clone,
    A: Into<AB> + Clone,
    SB: Batchable,
    AB: Batchable,
{
    fn from(value: Vec<&Transition<BT, S, A>>) -> Self {
        let states: Vec<_> = value.iter().map(|t| t.state.clone().into()).collect();
        let next_states: Vec<_> = value.iter().map(|t| t.next_state.clone().into()).collect();
        let actions: Vec<_> = value.iter().map(|t| t.action.clone().into()).collect();
        let rewards: Vec<_> = value.iter().map(|t| t.reward.clone()).collect();
        let dones: Vec<_> = value.iter().map(|t| t.done.clone()).collect();

        let rewards = Tensor::stack::<2>(rewards, 0);
        let dones = Tensor::stack::<2>(dones, 0);

        Self {
            states: SB::batch(states),
            next_states: SB::batch(next_states),
            actions: AB::batch(actions),
            rewards: Tensor::from_data(rewards.to_data(), &Default::default()),
            dones: Tensor::from_data(dones.to_data(), &Default::default()),
        }
    }
}

/// A circular buffer for transitions.
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

    /// Append a list of items to the current buffer.
    pub fn append(&mut self, items: &mut Vec<T>) {
        let n = items.len();
        let mut is_overflow = false;
        if n > self.capacity {
            self.cursor = self.capacity - (n % self.capacity);
            items.drain(0..n - self.capacity);
            is_overflow = true;
        }
        let n = items.len();

        let first_part = n.min(self.capacity - self.cursor);
        let second_part = n - first_part;

        if is_overflow {
            if self.capacity > self.len() {
                self.buffer
                    .extend(items.drain(first_part..second_part + first_part));
            } else {
                self.buffer[..second_part]
                    .iter_mut()
                    .zip(items.drain(first_part..second_part + first_part))
                    .for_each(|(slot, item)| *slot = item);
            }
        }

        if self.capacity > self.len() {
            self.buffer.extend(items.drain(..first_part));
        } else {
            self.buffer[self.cursor..self.cursor + first_part]
                .iter_mut()
                .zip(items.drain(..first_part))
                .for_each(|(slot, item)| *slot = item);
        }

        if !is_overflow {
            self.buffer[..second_part]
                .iter_mut()
                .zip(items.drain(..second_part))
                .for_each(|(slot, item)| *slot = item);
        }

        self.cursor = (self.cursor + n) % self.capacity
    }

    /// Returns the current number of items stored.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns the current number of items stored.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Sample the buffer at the given indices.
    pub fn sample(&self, indices: Vec<usize>) -> Result<Vec<&T>, Error> {
        let mut items = Vec::with_capacity(indices.len());

        for &idx in indices.iter() {
            match self.buffer.get(idx) {
                Some(item) => items.push(item),
                None => {
                    return Err(Error::new(
                        ErrorKind::InvalidInput,
                        format!(
                            "index out of bound: the len is {} but the index is {}",
                            self.len(),
                            idx
                        ),
                    ));
                }
            }
        }
        Ok(items)
    }

    /// Sample `batch_size` transitions at random.
    pub fn random_sample(&self, batch_size: usize) -> Result<Vec<&T>, Error> {
        if batch_size > self.len() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "batch size {} bigger than buffer length {}",
                    batch_size,
                    self.len()
                ),
            ));
        }
        let mut rng = rng();
        let indices = sample(&mut rng, self.len(), batch_size).into_vec();
        self.sample(indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_fills_and_overwrites() {
        let mut buffer = TransitionBuffer::new(3);
        assert_eq!(buffer.len(), 0);

        buffer.push(0);
        assert_eq!(buffer.len(), 1);
        assert_eq!(buffer.buffer, vec![0]);

        buffer.push(1);
        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.buffer, vec![0, 1]);

        buffer.push(2);
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.buffer, vec![0, 1, 2]);

        buffer.push(3);
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.buffer, vec![3, 1, 2]);

        buffer.push(4);
        buffer.push(5);
        buffer.push(6);
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.buffer, vec![6, 4, 5]);
    }

    #[test]
    fn append_fills_and_overwrites() {
        let mut buffer = TransitionBuffer::new(4);
        assert_eq!(buffer.len(), 0);

        buffer.append(&mut vec![0, 1]);
        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.buffer, vec![0, 1]);

        buffer.append(&mut vec![2, 3, 4, 5]);
        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer.buffer, vec![4, 5, 2, 3]);

        let mut buffer = TransitionBuffer::new(4);
        buffer.append(&mut vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer.buffer, vec![4, 5, 2, 3]);

        buffer.append(&mut vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);
        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer.buffer, vec![20, 17, 18, 19]);

        buffer.append(&mut vec![21, 22]);
        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer.buffer, vec![20, 21, 22, 19]);
    }

    #[test]
    fn clear_removes_all_items() {
        let mut buffer = TransitionBuffer::new(2);
        assert!(buffer.is_empty());

        buffer.push(1);
        assert!(!buffer.is_empty());

        buffer.clear();
        assert!(buffer.is_empty());

        buffer.append(&mut vec![1, 2, 3]);
        assert!(!buffer.is_empty());
        buffer.clear();
        assert!(buffer.is_empty());
    }

    #[test]
    fn sample_with_valid_indices() {
        let buffer = TransitionBuffer {
            buffer: vec![1, 2, 3, 4],
            capacity: 5,
            cursor: 0,
        };

        let samples = buffer.sample(vec![0, 2]).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(*samples[0], 1);
        assert_eq!(*samples[1], 3);
    }

    #[test]
    fn sample_with_out_of_bounds_indices() {
        let buffer = TransitionBuffer {
            buffer: vec![10, 20, 30],
            capacity: 5,
            cursor: 0,
        };

        let result = buffer.sample(vec![0, 5, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn random_sample_returns_correct_size() {
        let buffer = TransitionBuffer {
            buffer: vec![1, 2, 3, 4, 5],
            capacity: 5,
            cursor: 0,
        };

        let samples = buffer.random_sample(3).unwrap();
        assert_eq!(samples.len(), 3);
    }

    #[test]
    fn random_sample_exceeds_buffer_size() {
        let buffer = TransitionBuffer {
            buffer: vec![1, 2, 3],
            capacity: 3,
            cursor: 0,
        };

        let result = buffer.random_sample(5);
        assert!(result.is_err())
    }
}
