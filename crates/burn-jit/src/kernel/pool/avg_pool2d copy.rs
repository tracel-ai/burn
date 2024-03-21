use crate::{
    compute::{Kernel, StaticKernel},
    element::JitElement,
    kernel::{
        self, elemwise_workgroup, pool::build_output_and_info_pool2d, KernelSettings,
        StaticKernelSource, WORKGROUP_DEFAULT,
    },
    kernel_wgsl,
    tensor::JitTensor,
    Runtime,
};

kernel_wgsl!(AvgPool2dRaw, "../../template/pool/avg_pool2d.wgsl");

struct AvgPool2d<const COUNT_INCLUDE_PAD: bool>;

impl<const COUNT_INCLUDE_PAD: bool> StaticKernelSource for AvgPool2d<COUNT_INCLUDE_PAD> {
    fn source() -> kernel::SourceTemplate {
        AvgPool2dRaw::source().register("count_include_pad", format!("{COUNT_INCLUDE_PAD}"))
    }
}

pub(crate) fn avg_pool2d<R: Runtime, E: JitElement>(
    x: JitTensor<R, E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
) -> JitTensor<R, E, 4> {
    let (info_handle, output) =
        build_output_and_info_pool2d(&x, kernel_size, stride, padding, [1, 1]);

    let workgroup = elemwise_workgroup(output.shape.num_elements(), WORKGROUP_DEFAULT);
    let kernel: Box<dyn Kernel> = match count_include_pad {
        true => Box::new(StaticKernel::<
            KernelSettings<AvgPool2d<true>, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
        >::new(workgroup)),
        false => Box::new(StaticKernel::<
            KernelSettings<AvgPool2d<false>, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
        >::new(workgroup)),
    };

    x.client
        .execute(kernel, &[&x.handle, &output.handle, &info_handle]);

    output
}
