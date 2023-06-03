use super::{build_binary_info, build_unary_info, KernelGenerator, KernelSettings};
use crate::{context::WorkGroup, element::WGPUElement, kernel_wgsl, tensor::WGPUTensor};
use burn_tensor::Shape;
use std::sync::Arc;

kernel_wgsl!(BinaryElemwiseRaw, "../template/binary_elemwise.wgsl");
kernel_wgsl!(
    BinaryElemwiseInplaceRaw,
    "../template/binary_elemwise_inplace.wgsl"
);

#[macro_export]
macro_rules! binary_elemwise {
    (
        $struct:ident,
        $ops:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::KernelGenerator for $struct {
            type Source = String;

            fn generate() -> Self::Source {
                let source = $crate::kernel::BinaryElemwiseRaw::generate().to_string();
                let body = format!(
                    "output[global_id.x] = lhs[index_lhs] {} rhs[index_rhs];",
                    $ops
                );
                source.replace("BODY", &body)
            }
        }
    };
}

#[macro_export]
macro_rules! binary_elemwise_inplace {
    (
        $struct:ident,
        $ops:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::KernelGenerator for $struct {
            type Source = String;

            fn generate() -> Self::Source {
                let source = $crate::kernel::BinaryElemwiseInplaceRaw::generate().to_string();
                let body = format!(
                    "lhs[global_id.x] = lhs[global_id.x] {} rhs[index_rhs];",
                    $ops
                );
                source.replace("BODY", &body)
            }
        }
    };
}

pub fn binary_elemwise<K: KernelGenerator, E: WGPUElement, const D: usize>(
    lhs: WGPUTensor<E, D>,
    rhs: WGPUTensor<E, D>,
) -> WGPUTensor<E, D> {
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
    let output = WGPUTensor::new(lhs.context.clone(), shape_out, Arc::new(buffer));

    let kernel = lhs
        .context
        .compile::<KernelSettings<K, E, i32, 256, 1, 1>>();
    let info = build_binary_info(&lhs, &rhs);
    let info_buffers = lhs
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    lhs.context.execute(
        &WorkGroup::new(
            f32::ceil(output.shape.num_elements() as f32 / 256_f32) as u32,
            1,
            1,
        ),
        &kernel,
        &[&lhs.buffer, &rhs.buffer, &output.buffer, &info_buffers],
    );

    output
}
pub fn binary_elemwise_inplace<K: KernelGenerator, E: WGPUElement, const D: usize>(
    lhs: WGPUTensor<E, D>,
    rhs: WGPUTensor<E, D>,
) -> WGPUTensor<E, D> {
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
        .compile::<KernelSettings<K, E, i32, 256, 1, 1>>();
    let info = build_unary_info(&rhs);
    let info_buffers = lhs
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    lhs.context.execute(
        &WorkGroup::new(
            f32::ceil(lhs.shape.num_elements() as f32 / 256_f32) as u32,
            1,
            1,
        ),
        &kernel,
        &[&lhs.buffer, &rhs.buffer, &info_buffers],
    );

    lhs
}
