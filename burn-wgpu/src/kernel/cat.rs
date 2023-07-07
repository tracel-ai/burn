use crate::{
    element::WgpuElement,
    kernel::{build_info, elemwise_workgroup, KernelSettings},
    kernel_wgsl,
    tensor::WgpuTensor,
};

kernel_wgsl!(Cat, "../template/cat.wgsl");

pub fn cat<E: WgpuElement, const D: usize>(
    inputs: Vec<WgpuTensor<E, D>>,
    dim: usize,
) -> WgpuTensor<E, D> {
    const WORKGROUP: usize = 32;

    let first_input = inputs.get(0).unwrap();
    let context = &first_input.context;
    let mut shape_output = first_input.shape.clone();
    shape_output.dims[dim] = inputs.iter().map(|input| input.shape.dims[dim]).sum();

    let buffer = first_input
        .context
        .create_buffer(shape_output.num_elements() * std::mem::size_of::<E>());

    let output = WgpuTensor::new(context.clone(), shape_output, buffer);
    let kernel = context.compile_static::<KernelSettings<Cat, E, i32, WORKGROUP, WORKGROUP, 1>>();

    let mut dim_cat_index = 0;

    for input in inputs.iter() {
        let mut info = build_info(&[input, &output]);
        info.push(dim as u32);
        info.push(dim_cat_index as u32);
        dim_cat_index += input.shape.dims[dim];
        let info_buffer = context.create_buffer_with_data(bytemuck::cast_slice(&info));

        context.execute(
            elemwise_workgroup(input.shape.num_elements(), WORKGROUP),
            kernel.clone(),
            &[&input.buffer, &output.buffer, &info_buffer],
        );
    }

    output
}

#[cfg(test)]
mod tests {
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{backend::Backend, Distribution, Tensor};

    #[test]
    fn cat_should_support_multiple_invocations_dim0() {
        test_same_as_reference([6, 256], 2, 0);
    }

    #[test]
    fn cat_should_support_multiple_invocations_dim1() {
        test_same_as_reference([6, 256], 2, 1);
    }

    #[test]
    fn cat_should_support_uneven_launch() {
        test_same_as_reference([1, 137], 2, 0);
    }

    fn test_same_as_reference(shape: [usize; 2], num_tensors: usize, dim: usize) {
        TestBackend::seed(0);
        let tensors = (0..num_tensors)
            .map(|_| Tensor::<TestBackend, 2>::random(shape, Distribution::Default))
            .collect::<Vec<_>>();
        let tensors_ref = tensors
            .iter()
            .map(|tensor| Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data()))
            .collect::<Vec<_>>();

        let tensor = Tensor::<TestBackend, 2>::cat(tensors, dim);
        let tensor_ref = Tensor::<ReferenceBackend, 2>::cat(tensors_ref, dim);

        tensor
            .into_data()
            .assert_approx_eq(&tensor_ref.into_data(), 3);
    }
}
