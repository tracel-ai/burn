use crate::{
    compute::StaticKernel,
    element::JitElement,
    kernel::{
        self, elemwise_workgroup,
        pool::{build_output_and_info_pool2d, build_pool2d_info},
        KernelSettings, WORKGROUP_DEFAULT,
    },
    kernel_wgsl,
    ops::numeric::empty_device,
    tensor::JitTensor,
    Runtime,
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

pub(crate) fn max_pool2d_old<R: Runtime, E: JitElement>(
    x: JitTensor<R, E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> JitTensor<R, E, 4> {
    let (info_handle, output) =
        build_output_and_info_pool2d(&x, kernel_size, stride, padding, dilation);
    let kernel = StaticKernel::<
        KernelSettings<MaxPool2d, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
    >::new(elemwise_workgroup(
        output.shape.num_elements(),
        WORKGROUP_DEFAULT,
    ));

    x.client
        .execute(Box::new(kernel), &[&x.handle, &output.handle, &info_handle]);

    output
}

pub(crate) fn max_pool2d_with_indices_old<R: Runtime, E: JitElement, I: JitElement>(
    x: JitTensor<R, E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> (JitTensor<R, E, 4>, JitTensor<R, I, 4>) {
    let (info_handle, output) =
        build_output_and_info_pool2d(&x, kernel_size, stride, padding, dilation);
    let indices = empty_device(x.client.clone(), x.device, output.shape.clone());

    let kernel = StaticKernel::<
        KernelSettings<MaxPool2dWithIndices, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
    >::new(elemwise_workgroup(
        output.shape.num_elements(),
        WORKGROUP_DEFAULT,
    ));

    x.client.execute(
        Box::new(kernel),
        &[&x.handle, &output.handle, &indices.handle, &info_handle],
    );

    (output, indices)
}

pub(crate) fn max_pool2d_with_indices_backward<R: Runtime, E: JitElement, I: JitElement>(
    x: JitTensor<R, E, 4>,
    grad: JitTensor<R, E, 4>,
    indices: JitTensor<R, I, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> JitTensor<R, E, 4> {
    let grad = kernel::into_contiguous(grad);
    let indices = kernel::into_contiguous(indices);

    let num_elems = x.shape.num_elements();
    let buffer = x.client.empty(num_elems * core::mem::size_of::<E>());
    let output = JitTensor::new(x.client.clone(), x.device.clone(), x.shape.clone(), buffer);

    let info_handle = build_pool2d_info(&x, &grad, kernel_size, stride, padding, dilation);

    let kernel = StaticKernel::<
        KernelSettings<MaxPool2dWithIndicesBackward, E, I, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
    >::new(elemwise_workgroup(
        output.shape.num_elements(),
        WORKGROUP_DEFAULT,
    ));

    x.client.execute(
        Box::new(kernel),
        &[&indices.handle, &grad.handle, &output.handle, &info_handle],
    );
    output
}
