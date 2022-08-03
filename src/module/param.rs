use crate::tensor::{back, Tensor};

#[derive(Debug)]
pub struct Param<T> {
    value: T,
}

impl<T> std::ops::Deref for Param<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> Param<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<const D: usize, B: back::Backend> Param<Tensor<D, B>> {
    pub fn num_params(&self) -> usize {
        self.value.shape().num_elements()
    }
}

impl<const D: usize, B: back::Backend> Param<Option<Tensor<D, B>>> {
    pub fn num_params(&self) -> usize {
        if let Some(value) = &self.value {
            return value.shape().num_elements();
        }

        0
    }
}
