use alloc::vec;

use super::TensorCache;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

impl<B: Backend, const D: usize> TensorCache<B, D> {
    pub(crate) fn forward_autoregressive<F>(
        &mut self,
        tensor: Tensor<B, 3>,
        dim_cat: usize,
        func: F,
    ) -> Tensor<B, D>
    where
        F: Fn(Tensor<B, 3>) -> Tensor<B, D>,
    {
        let mut tensor_old = None;
        core::mem::swap(&mut self.state, &mut tensor_old);

        let tensor_new = match tensor_old {
            Some(tensor_old) => {
                let [batch_size, seq_length, d_model] = tensor.dims();
                let next_seq_token =
                    tensor.index([0..batch_size, (seq_length - 1)..seq_length, 0..d_model]);
                let next_seq_token = func(next_seq_token);

                Tensor::cat(vec![tensor_old, next_seq_token], dim_cat)
            }
            None => func(tensor),
        };

        self.state = Some(tensor_new.clone());
        tensor_new
    }
}
