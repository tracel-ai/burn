use crate::{BasicOps, Slice, TensorKind, TensorMetadata, backend::Backend};
use alloc::vec::Vec;

pub(crate) fn cat_with_slice_assign<B: Backend, K: TensorKind<B> + BasicOps<B>>(
    tensors: Vec<K::Primitive>,
    dim: usize,
) -> K::Primitive {
    let first_tensor = tensors.first().expect("Tensors should not be empty");
    let mut shape = first_tensor.shape();
    let device = K::device(first_tensor);
    let dtype = first_tensor.dtype();

    let output_dim_length: usize = tensors.iter().map(|tensor| tensor.shape().dims[dim]).sum();
    shape[dim] = output_dim_length;

    let mut tensor_output = K::empty(shape.clone(), &device, dtype);

    let indices_select_all = shape.iter().map(|d| 0..*d).collect::<Vec<_>>();

    let mut output_index = 0;
    for tensor in tensors {
        let mut indices = indices_select_all.clone();
        let tensor_dim_length = tensor.shape().dims[dim];
        indices[dim] = output_index..output_index + tensor_dim_length;
        output_index += tensor_dim_length;

        // Convert ranges to Slice
        let slices: Vec<Slice> = indices
            .iter()
            .map(|r| Slice::new(r.start as isize, Some(r.end as isize), 1))
            .collect();
        tensor_output = K::slice_assign(tensor_output, &slices, tensor);
    }

    tensor_output
}
