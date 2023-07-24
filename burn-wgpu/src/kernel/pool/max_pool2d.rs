use crate::{
    element::WgpuElement,
    kernel::{self, elemwise_workgroup, pool::build_output_and_info_pool2d, KernelSettings},
    kernel_wgsl,
    tensor::WgpuTensor,
};

kernel_wgsl!(MaxPool2d, "../../template/pool/max_pool2d.wgsl");
kernel_wgsl!(
    MaxPool2dWithIndicesBackward,
    "../../template/pool/max_pool2d_with_indices_backward.wgsl"
);
kernel_wgsl!(
    MaxPool2dWithIndices,
    "../../template/pool/max_pool2d_with_indices.wgsl"
);

pub(crate) fn max_pool2d<E: WgpuElement>(
    x: WgpuTensor<E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> WgpuTensor<E, 4> {
    const WORKGROUP: usize = 32;

    let (info_buffer, output) = build_output_and_info_pool2d(&x, kernel_size, stride, padding);
    let kernel = x
        .context
        .compile_static::<KernelSettings<MaxPool2d, E, i32, WORKGROUP, WORKGROUP, 1>>();

    x.context.execute(
        elemwise_workgroup(output.shape.num_elements(), WORKGROUP),
        kernel,
        &[&x.buffer, &output.buffer, &info_buffer],
    );

    output
}

pub(crate) fn max_pool2d_with_indices<E: WgpuElement, I: WgpuElement>(
    x: WgpuTensor<E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> (WgpuTensor<E, 4>, WgpuTensor<I, 4>) {
    const WORKGROUP: usize = 32;

    let (info_buffer, output) = build_output_and_info_pool2d(&x, kernel_size, stride, padding);
    let num_elems = output.shape.num_elements();

    let indices = WgpuTensor::new(
        x.context.clone(),
        output.shape.clone(),
        x.context
            .create_buffer(num_elems * std::mem::size_of::<I>()),
    );

    let kernel = x
        .context
        .compile_static::<KernelSettings<MaxPool2dWithIndices, E, i32, WORKGROUP, WORKGROUP, 1>>();

    x.context.execute(
        elemwise_workgroup(output.shape.num_elements(), WORKGROUP),
        kernel,
        &[&x.buffer, &output.buffer, &indices.buffer, &info_buffer],
    );

    (output, indices)
}

pub(crate) fn max_pool2d_with_indices_backward<E: WgpuElement, I: WgpuElement>(
    x: WgpuTensor<E, 4>,
    grad: WgpuTensor<E, 4>,
    indices: WgpuTensor<I, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> WgpuTensor<E, 4> {
    const WORKGROUP: usize = 32;

    let grad = kernel::into_contiguous(grad);
    let indices = kernel::into_contiguous(indices);

    let num_elems = x.shape.num_elements();
    let buffer = x
        .context
        .create_buffer(num_elems * core::mem::size_of::<E>());
    let output = WgpuTensor::new(x.context.clone(), x.shape.clone(), buffer);

    let mut info: [u32; 18] = [0; 18];
    info[0] = x.strides[0] as u32;
    info[1] = x.strides[1] as u32;
    info[2] = x.strides[2] as u32;
    info[3] = x.strides[3] as u32;
    info[4] = x.shape.dims[0] as u32;
    info[5] = x.shape.dims[1] as u32;
    info[6] = x.shape.dims[2] as u32;
    info[7] = x.shape.dims[3] as u32;

    info[8] = grad.strides[0] as u32;
    info[9] = grad.strides[1] as u32;
    info[10] = grad.strides[2] as u32;
    info[11] = grad.strides[3] as u32;

    info[12] = kernel_size[0] as u32;
    info[13] = kernel_size[1] as u32;
    info[14] = stride[0] as u32;
    info[15] = stride[1] as u32;
    info[16] = padding[0] as u32;
    info[17] = padding[1] as u32;

    let info_buffer = x
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    let kernel = x.context.compile_static::<KernelSettings<
        MaxPool2dWithIndicesBackward,
        E,
        I,
        WORKGROUP,
        WORKGROUP,
        1,
    >>();

    x.context.execute(
        elemwise_workgroup(output.shape.num_elements(), WORKGROUP),
        kernel,
        &[&indices.buffer, &grad.buffer, &output.buffer, &info_buffer],
    );
    output
}

#[cfg(test)]
mod tests {
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{module, ops::ModuleOps, Distribution, Tensor};

    #[test]
    pub fn max_pool2d_should_work_with_multiple_invocations() {
        let tensor = Tensor::<TestBackend, 4>::random([32, 32, 32, 32], Distribution::Default);
        let tensor_ref = Tensor::<ReferenceBackend, 4>::from_data(tensor.to_data());
        let kernel_size = [3, 3];
        let stride = [2, 2];
        let padding = [1, 1];

        let pooled = module::max_pool2d(tensor, kernel_size, stride, padding);
        let pooled_ref = module::max_pool2d(tensor_ref, kernel_size, stride, padding);

        pooled
            .into_data()
            .assert_approx_eq(&pooled_ref.into_data(), 3);
    }

    #[test]
    pub fn max_pool2d_with_indices_should_work_with_multiple_invocations() {
        let tensor = Tensor::<TestBackend, 4>::random([32, 32, 32, 32], Distribution::Default);
        let tensor_ref = Tensor::<ReferenceBackend, 4>::from_data(tensor.to_data());
        let kernel_size = [3, 3];
        let stride = [2, 2];
        let padding = [1, 1];

        let (pooled, indices) =
            module::max_pool2d_with_indices(tensor, kernel_size, stride, padding);
        let (pooled_ref, indices_ref) =
            module::max_pool2d_with_indices(tensor_ref, kernel_size, stride, padding);

        pooled
            .into_data()
            .assert_approx_eq(&pooled_ref.into_data(), 3);
        assert_eq!(indices.into_data(), indices_ref.into_data().convert());
    }

    #[test]
    pub fn max_pool2d_with_indices_backward_should_work_with_multiple_invocations() {
        let tensor = Tensor::<TestBackend, 4>::random([32, 32, 32, 32], Distribution::Default);
        let grad_output = Tensor::<TestBackend, 4>::random([32, 32, 16, 16], Distribution::Default);
        let tensor_ref = Tensor::<ReferenceBackend, 4>::from_data(tensor.to_data());
        let grad_output_ref = Tensor::<ReferenceBackend, 4>::from_data(grad_output.to_data());
        let kernel_size = [3, 3];
        let stride = [2, 2];
        let padding = [1, 1];

        let (_, indices) =
            module::max_pool2d_with_indices(tensor.clone(), kernel_size, stride, padding);
        let (_, indices_ref) =
            module::max_pool2d_with_indices(tensor_ref.clone(), kernel_size, stride, padding);
        let grad = TestBackend::max_pool2d_with_indices_backward(
            tensor.into_primitive(),
            kernel_size,
            stride,
            padding,
            grad_output.into_primitive(),
            indices.into_primitive(),
        )
        .x_grad;
        let grad_ref = ReferenceBackend::max_pool2d_with_indices_backward(
            tensor_ref.into_primitive(),
            kernel_size,
            stride,
            padding,
            grad_output_ref.into_primitive(),
            indices_ref.into_primitive(),
        )
        .x_grad;

        Tensor::<TestBackend, 4>::from_primitive(grad)
            .into_data()
            .assert_approx_eq(
                &Tensor::<ReferenceBackend, 4>::from_primitive(grad_ref).into_data(),
                3,
            );
    }
}
