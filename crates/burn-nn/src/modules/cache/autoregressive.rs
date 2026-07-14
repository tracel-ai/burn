use alloc::vec;
use burn_core as burn;

use super::{CacheState, TensorCache};
use burn::tensor::Tensor;

impl<const D: usize> TensorCache<D> {
    pub(crate) fn forward_autoregressive<F>(
        &mut self,
        tensor: Tensor<3>,
        dim_cat: usize,
        func: F,
    ) -> Tensor<D>
    where
        F: Fn(Tensor<3>) -> Tensor<D>,
    {
        let mut tensor_old = CacheState::Empty;
        core::mem::swap(&mut self.state, &mut tensor_old);

        let tensor_new = match tensor_old {
            CacheState::Value(tensor_old) => {
                let [batch_size, seq_length, d_model] = tensor.dims();
                let next_seq_token =
                    tensor.slice([0..batch_size, (seq_length - 1)..seq_length, 0..d_model]);
                let next_seq_token = func(next_seq_token);

                Tensor::cat(vec![tensor_old, next_seq_token], dim_cat)
            }
            _ => func(tensor),
        };

        self.state = CacheState::Value(tensor_new.clone());
        tensor_new
    }

    pub(crate) fn forward_full<F>(&mut self, tensor: Tensor<3>, func: F) -> Tensor<D>
    where
        F: Fn(Tensor<3>) -> Tensor<D>,
    {
        let mut tensor_old = CacheState::Empty;
        core::mem::swap(&mut self.state, &mut tensor_old);

        let tensor_new = match tensor_old {
            CacheState::Value(tensor_old) => tensor_old,
            _ => func(tensor),
        };

        self.state = CacheState::Value(tensor_new.clone());
        tensor_new
    }
}
