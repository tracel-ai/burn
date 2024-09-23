use crate::{backend::Backend, BasicOps, TensorKind};
use alloc::vec::Vec;

pub(crate) fn cat_with_slice_assign<B: Backend, K: TensorKind<B> + BasicOps<B>>(
    tensors: Vec<K::Primitive>,
    dim: usize,
) -> K::Primitive {
    let first_tensor = tensors.first().expect("Tensors should not be empty");
    let mut shape = K::shape(first_tensor);
    let device = K::device(first_tensor);

    let output_dim_length: usize = tensors
        .iter()
        .map(|tensor| K::shape(tensor).dims[dim])
        .sum();
    shape.dims[dim] = output_dim_length;

    let mut tensor_output = K::empty(shape.clone(), &device);

    let indices_select_all = shape.dims.iter().map(|d| 0..*d).collect::<Vec<_>>();

    let mut output_index = 0;
    for tensor in tensors {
        let mut indices = indices_select_all.clone();
        let tensor_dim_length = K::shape(&tensor).dims[dim];
        indices[dim] = output_index..output_index + tensor_dim_length;
        output_index += tensor_dim_length;

        tensor_output = K::slice_assign(tensor_output, &indices, tensor);
    }

    tensor_output
}
