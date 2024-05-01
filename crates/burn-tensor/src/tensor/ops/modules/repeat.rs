use crate::{backend::Backend, BasicOps, Tensor, TensorKind};

pub(crate) fn repeat_with_slice_assign<
    B: Backend,
    const D: usize,
    K: TensorKind<B> + BasicOps<B>,
>(
    tensor: Tensor<B, D, K>,
    dim: usize,
    times: usize,
) -> Tensor<B, D, K> {
    let mut shape = tensor.shape();
    let device = tensor.device();

    let original_dim_length = shape.dims[dim];
    shape.dims[dim] *= times;

    let mut tensor_output = Tensor::empty(shape.clone(), &device);

    let mut i = 0;
    let indices_select_all = [0; D].map(|_| {
        i += 1;
        0..shape.dims[i - 1]
    });

    let mut output_index = 0;
    for _ in 0..times {
        let mut indices = indices_select_all.clone();
        indices[dim] = output_index..output_index + original_dim_length;
        output_index += original_dim_length;

        tensor_output = tensor_output.slice_assign(indices, tensor.clone());
    }

    tensor_output
}
