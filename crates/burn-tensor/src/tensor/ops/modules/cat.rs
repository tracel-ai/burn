use crate::{backend::Backend, BasicOps, Tensor, TensorKind};
use alloc::vec::Vec;

pub(crate) fn cat_with_slice_assign<B: Backend, const D: usize, K: TensorKind<B> + BasicOps<B>>(
    tensors: Vec<Tensor<B, D, K>>,
    dim: usize,
) -> Tensor<B, D, K> {
    let first_tensor = tensors.first().expect("Tensors should not be empty");
    let mut shape = first_tensor.shape();
    let device = first_tensor.device();

    let output_dim_length: usize = tensors
        .iter()
        .map(|tensor: &Tensor<B, D, K>| tensor.shape().dims[dim])
        .sum();
    shape.dims[dim] = output_dim_length;

    let mut tensor_output = Tensor::empty(shape.clone(), &device);

    let mut i = 0;
    let indices_select_all = [0; D].map(|_| {
        i += 1;
        0..shape.dims[i - 1]
    });

    let mut output_index = 0;
    for tensor in tensors {
        let mut indices = indices_select_all.clone();
        let tensor_dim_length = tensor.shape().dims[dim];
        indices[dim] = output_index..output_index + tensor_dim_length;
        output_index += tensor_dim_length;

        tensor_output = tensor_output.slice_assign(indices, tensor);
    }

    tensor_output
}
