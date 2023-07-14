use crate::{
    element::WgpuElement,
    kernel::{self, elemwise_workgroup, KernelSettings},
    kernel_wgsl,
    tensor::WgpuTensor,
};
use burn_tensor::Shape;
use std::sync::Arc;
use wgpu::Buffer;

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

    let (info_buffer, output) = build_output_and_info(&x, kernel_size, stride, padding);
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

    let (info_buffer, output) = build_output_and_info(&x, kernel_size, stride, padding);
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

    let grad = kernel::into_continuous(grad);
    let indices = kernel::into_continuous(indices);

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

fn build_output_and_info<E: WgpuElement>(
    x: &WgpuTensor<E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> (Arc<Buffer>, WgpuTensor<E, 4>) {
    let [kernel_height, kernel_width] = kernel_size;
    let [padding_height, padding_width] = padding;
    let [stride_height, stride_width] = stride;
    let [batch_size, channels, x_height, x_width] = x.shape.dims;

    let out_height = ((x_height + 2 * padding_height - kernel_height) / stride_height) + 1;
    let out_width = ((x_width + 2 * padding_width - kernel_width) / stride_width) + 1;
    let shape_out = Shape::new([batch_size, channels, out_height, out_width]);
    let num_elems = shape_out.num_elements();

    let buffer = x
        .context
        .create_buffer(num_elems * core::mem::size_of::<E>());
    let output = WgpuTensor::new(x.context.clone(), shape_out, buffer);

    let mut info: [u32; 22] = [0; 22];
    info[0] = x.strides[0] as u32;
    info[1] = x.strides[1] as u32;
    info[2] = x.strides[2] as u32;
    info[3] = x.strides[3] as u32;
    info[4] = x.shape.dims[0] as u32;
    info[5] = x.shape.dims[1] as u32;
    info[6] = x.shape.dims[2] as u32;
    info[7] = x.shape.dims[3] as u32;

    info[8] = output.strides[0] as u32;
    info[9] = output.strides[1] as u32;
    info[10] = output.strides[2] as u32;
    info[11] = output.strides[3] as u32;
    info[12] = output.shape.dims[0] as u32;
    info[13] = output.shape.dims[1] as u32;
    info[14] = output.shape.dims[2] as u32;
    info[15] = output.shape.dims[3] as u32;

    info[16] = kernel_height as u32;
    info[17] = kernel_width as u32;
    info[18] = stride_height as u32;
    info[19] = stride_width as u32;
    info[20] = padding_height as u32;
    info[21] = padding_width as u32;

    let info_buffer = x
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    (info_buffer, output)
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
