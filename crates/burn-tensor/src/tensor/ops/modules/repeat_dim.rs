use crate::{BasicOps, TensorKind, TensorMetadata, backend::Backend};
use alloc::vec::Vec;

pub(crate) fn repeat_with_slice_assign<B: Backend, K: TensorKind<B> + BasicOps<B>>(
    tensor: K::Primitive,
    dim: usize,
    times: usize,
) -> K::Primitive {
    let mut shape = tensor.shape();
    let device = K::device(&tensor);
    let dtype = tensor.dtype();

    let original_dim_length = shape.dims[dim];
    shape.dims[dim] *= times;

    let mut tensor_output = K::empty(shape.clone(), &device, dtype);

    let indices_select_all = shape.dims.iter().map(|d| 0..*d).collect::<Vec<_>>();

    let mut output_index = 0;
    for _ in 0..times {
        let mut indices = indices_select_all.clone();
        indices[dim] = output_index..output_index + original_dim_length;
        output_index += original_dim_length;

        tensor_output = K::slice_assign(tensor_output, &indices, tensor.clone());
    }

    tensor_output
}
