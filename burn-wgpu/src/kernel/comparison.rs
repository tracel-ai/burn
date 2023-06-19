use super::{build_info, KernelSettings, StaticKernel};
use crate::{context::WorkGroup, element::WgpuElement, kernel_wgsl, tensor::WgpuTensor};
use burn_tensor::Shape;

kernel_wgsl!(ComparisonRaw, "../template/comparison/binary.wgsl");
kernel_wgsl!(
    ComparisonInplaceRaw,
    "../template/comparison/binary_inplace.wgsl"
);

#[macro_export]
macro_rules! comparison {
    (
        $struct:ident,
        $ops:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::ComparisonRaw::source_template().register(
                    "body",
                    format!(
                        "output[global_id.x] = u32(lhs[index_lhs] {} rhs[index_rhs]);",
                        $ops
                    ),
                )
            }
        }
    };
}

#[macro_export]
macro_rules! comparison_inplace {
    (
        $struct:ident,
        $ops:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::ComparisonInplaceRaw::source_template()
                    .register(
                        "body",
                        "lhs[index_lhs] = compare(lhs[index_lhs], rhs[index_rhs]);",
                    )
                    .add_template(format!(
                        "{}return {{{{ elem }}}}(lhs {} rhs);{}",
                        "fn compare(lhs: {{ elem }}, rhs: {{ elem }}) -> {{ elem }} {\n",
                        $ops,
                        "\n}\n"
                    ))
            }
        }
    };
}

pub fn comparison<K: StaticKernel, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D> {
    lhs.assert_is_on_save_device(&rhs);

    let mut shape_out = [0; D];
    lhs.shape
        .dims
        .iter()
        .zip(rhs.shape.dims.iter())
        .enumerate()
        .for_each(|(index, (dim_lhs, dim_rhs))| {
            shape_out[index] = usize::max(*dim_lhs, *dim_rhs);
        });

    let shape_out = Shape::new(shape_out);

    let buffer = lhs
        .context
        .create_buffer(shape_out.num_elements() * core::mem::size_of::<u32>());
    let output = WgpuTensor::new(lhs.context.clone(), shape_out, buffer);

    let kernel = lhs
        .context
        .compile_static::<KernelSettings<K, E, i32, 256, 1, 1>>();
    let info = build_info(&[&lhs, &rhs, &output]);
    let info_buffers = lhs
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    lhs.context.execute(
        WorkGroup::new(
            f32::ceil(output.shape.num_elements() as f32 / 256_f32) as u32,
            1,
            1,
        ),
        kernel,
        &[&lhs.buffer, &rhs.buffer, &output.buffer, &info_buffers],
    );

    WgpuTensor::new(output.context, output.shape, output.buffer)
}

pub fn comparison_inplace<K: StaticKernel, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D> {
    lhs.assert_is_on_save_device(&rhs);

    let kernel = lhs
        .context
        .compile_static::<KernelSettings<K, E, i32, 256, 1, 1>>();
    let info = build_info(&[&lhs, &rhs]);
    let info_buffers = lhs
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    lhs.context.execute(
        WorkGroup::new(
            f32::ceil(lhs.shape.num_elements() as f32 / 256_f32) as u32,
            1,
            1,
        ),
        kernel,
        &[&lhs.buffer, &rhs.buffer, &info_buffers],
    );

    WgpuTensor::new(lhs.context, lhs.shape, lhs.buffer)
}
