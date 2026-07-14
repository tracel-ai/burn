use crate::{Backend, TensorMetadata, tensor::Device};
use alloc::vec::Vec;
use burn_std::{DType, Shape, Slice};

pub(crate) fn cat_with_slice_assign<B, T, E, SA>(
    tensors: Vec<T>,
    dim: usize,
    device: Device<B>,
    empty: E,
    slice_assign: SA,
) -> T
where
    T: TensorMetadata,
    B: Backend,
    E: Fn(Shape, &Device<B>, DType) -> T,
    SA: Fn(T, &[Slice], T) -> T,
{
    let first_tensor = tensors.first().expect("Tensors should not be empty");
    let mut shape = first_tensor.shape();
    let dtype = first_tensor.dtype();

    let output_dim_length: usize = tensors.iter().map(|tensor| tensor.shape()[dim]).sum();
    shape[dim] = output_dim_length;

    let mut tensor_output = empty(shape.clone(), &device, dtype);

    let indices_select_all = shape.iter().map(|d| 0..*d).collect::<Vec<_>>();

    let mut output_index = 0;
    for tensor in tensors {
        let mut indices = indices_select_all.clone();
        let tensor_dim_length = tensor.shape()[dim];
        indices[dim] = output_index..output_index + tensor_dim_length;
        output_index += tensor_dim_length;

        // Convert ranges to Slice
        let slices: Vec<Slice> = indices
            .iter()
            .map(|r| Slice::new(r.start as isize, Some(r.end as isize), 1))
            .collect();
        tensor_output = slice_assign(tensor_output, &slices, tensor);
    }

    tensor_output
}
