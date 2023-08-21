use crate::{
    element::WgpuElement,
    kernel::{
        self, elemwise_workgroup,
        pool::{build_output_and_info_pool2d, build_pool2d_info},
        KernelSettings, StaticKernel,
    },
    kernel_wgsl,
    tensor::WgpuTensor,
};

kernel_wgsl!(AvgPool2dRaw, "../../template/pool/avg_pool2d.wgsl");
kernel_wgsl!(
    AvgPool2dBackwardRaw,
    "../../template/pool/avg_pool2d_backward.wgsl"
);

struct AvgPool2dBackward<const COUNT_INCLUDE_PAD: bool>;
struct AvgPool2d<const COUNT_INCLUDE_PAD: bool>;

impl<const COUNT_INCLUDE_PAD: bool> StaticKernel for AvgPool2dBackward<COUNT_INCLUDE_PAD> {
    fn source_template() -> kernel::SourceTemplate {
        AvgPool2dBackwardRaw::source_template()
            .register("count_include_pad", format!("{COUNT_INCLUDE_PAD}"))
    }
}

impl<const COUNT_INCLUDE_PAD: bool> StaticKernel for AvgPool2d<COUNT_INCLUDE_PAD> {
    fn source_template() -> kernel::SourceTemplate {
        AvgPool2dRaw::source_template()
            .register("count_include_pad", format!("{COUNT_INCLUDE_PAD}"))
    }
}

pub(crate) fn avg_pool2d<E: WgpuElement>(
    x: WgpuTensor<E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
) -> WgpuTensor<E, 4> {
    const WORKGROUP: usize = 32;

    let (info_buffer, output) =
        build_output_and_info_pool2d(&x, kernel_size, stride, padding, [1, 1]);
    let kernel = match count_include_pad {
        true => x
            .context
            .compile_static::<KernelSettings<AvgPool2d<true>, E, i32, WORKGROUP, WORKGROUP, 1>>(),
        false => x
            .context
            .compile_static::<KernelSettings<AvgPool2d<false>, E, i32, WORKGROUP, WORKGROUP, 1>>(),
    };

    x.context.execute(
        elemwise_workgroup(output.shape.num_elements(), WORKGROUP),
        kernel,
        &[&x.buffer, &output.buffer, &info_buffer],
    );

    output
}

pub(crate) fn avg_pool2d_backward<E: WgpuElement>(
    x: WgpuTensor<E, 4>,
    grad: WgpuTensor<E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
) -> WgpuTensor<E, 4> {
    const WORKGROUP: usize = 32;

    let grad = kernel::into_contiguous(grad);

    let num_elems = x.shape.num_elements();
    let buffer = x
        .context
        .create_buffer(num_elems * core::mem::size_of::<E>());
    let output = WgpuTensor::new(x.context.clone(), x.shape.clone(), buffer);
    let info_buffer = build_pool2d_info(&x, &grad, kernel_size, stride, padding, [1, 1]);

    let kernel =
        match count_include_pad {
            true => x.context.compile_static::<KernelSettings<
                AvgPool2dBackward<true>,
                E,
                i32,
                WORKGROUP,
                WORKGROUP,
                1,
            >>(),
            false => x.context.compile_static::<KernelSettings<
                AvgPool2dBackward<false>,
                E,
                i32,
                WORKGROUP,
                WORKGROUP,
                1,
            >>(),
        };

    x.context.execute(
        elemwise_workgroup(output.shape.num_elements(), WORKGROUP),
        kernel,
        &[&grad.buffer, &output.buffer, &info_buffer],
    );

    output
}

#[cfg(test)]
mod tests {
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{backend::Backend, module, ops::ModuleOps, Distribution, Tensor};

    #[test]
    fn avg_pool2d_should_work_with_multiple_invocations() {
        let tensor = Tensor::<TestBackend, 4>::random([32, 32, 32, 32], Distribution::Default);
        let tensor_ref = Tensor::<ReferenceBackend, 4>::from_data(tensor.to_data());
        let kernel_size = [3, 4];
        let stride = [1, 2];
        let padding = [1, 2];
        let count_include_pad = true;

        let pooled = module::avg_pool2d(tensor, kernel_size, stride, padding, count_include_pad);
        let pooled_ref =
            module::avg_pool2d(tensor_ref, kernel_size, stride, padding, count_include_pad);

        pooled
            .into_data()
            .assert_approx_eq(&pooled_ref.into_data(), 3);
    }

    #[test]
    fn avg_pool2d_backward_should_work_with_multiple_invocations() {
        TestBackend::seed(0);
        ReferenceBackend::seed(0);
        let tensor = Tensor::<TestBackend, 4>::random([32, 32, 32, 32], Distribution::Default);
        let tensor_ref = Tensor::<ReferenceBackend, 4>::from_data(tensor.to_data());
        let kernel_size = [3, 3];
        let stride = [1, 1];
        let padding = [1, 1];
        let count_include_pad = true;

        let shape_out = module::avg_pool2d(
            tensor.clone(),
            kernel_size,
            stride,
            padding,
            count_include_pad,
        )
        .shape();
        let grad_output = Tensor::<TestBackend, 4>::random(shape_out, Distribution::Default);
        let grad_output_ref = Tensor::<ReferenceBackend, 4>::from_data(grad_output.to_data());

        let grad: Tensor<TestBackend, 4> =
            Tensor::from_primitive(TestBackend::avg_pool2d_backward(
                tensor.into_primitive(),
                grad_output.into_primitive(),
                kernel_size,
                stride,
                padding,
                count_include_pad,
            ));
        let grad_ref: Tensor<ReferenceBackend, 4> =
            Tensor::from_primitive(ReferenceBackend::avg_pool2d_backward(
                tensor_ref.into_primitive(),
                grad_output_ref.into_primitive(),
                kernel_size,
                stride,
                padding,
                count_include_pad,
            ));

        grad.into_data().assert_approx_eq(&grad_ref.into_data(), 3);
    }
}
