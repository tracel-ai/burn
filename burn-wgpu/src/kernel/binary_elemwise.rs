use super::{build_info, KernelSettings, StaticKernel};
use crate::{context::WorkGroup, element::WgpuElement, kernel_wgsl, tensor::WgpuTensor};
use burn_tensor::Shape;

kernel_wgsl!(BinaryElemwiseRaw, "../template/binary_elemwise.wgsl");
kernel_wgsl!(
    BinaryElemwiseInplaceRaw,
    "../template/binary_elemwise_inplace.wgsl"
);

/// Creates a binary elementwise kernel.
#[macro_export]
macro_rules! binary_elemwise {
    (
        $struct:ident,
        $ops:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::BinaryElemwiseRaw::source_template().register(
                    "body",
                    format!(
                        "output[global_id.x] = lhs[index_lhs] {} rhs[index_rhs];",
                        $ops
                    ),
                )
            }
        }
    };
}

/// Creates a binary elementwise inplace kernel.
#[macro_export]
macro_rules! binary_elemwise_inplace {
    (
        $struct:ident,
        $ops:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::BinaryElemwiseInplaceRaw::source_template().register(
                    "body",
                    format!(
                        "lhs[global_id.x] = lhs[global_id.x] {} rhs[index_rhs];",
                        $ops
                    ),
                )
            }
        }
    };
}

pub fn binary_elemwise<K: StaticKernel, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    const WORKGROUP: usize = 256;
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
        .create_buffer(shape_out.num_elements() * core::mem::size_of::<E>());
    let output = WgpuTensor::new(lhs.context.clone(), shape_out, buffer);

    let kernel = lhs
        .context
        .compile_static::<KernelSettings<K, E, i32, WORKGROUP, 1, 1>>();
    let info = build_info(&[&lhs, &rhs, &output]);
    let info_buffers = lhs
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    lhs.context.execute(
        WorkGroup::new(
            f32::ceil(output.shape.num_elements() as f32 / WORKGROUP as f32) as u32,
            1,
            1,
        ),
        kernel,
        &[&lhs.buffer, &rhs.buffer, &output.buffer, &info_buffers],
    );

    output
}
pub fn binary_elemwise_inplace<K: StaticKernel, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
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

    lhs
}
