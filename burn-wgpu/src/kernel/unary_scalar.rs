use super::{KernelSettings, StaticKernel};
use crate::{context::WorkGroup, element::WgpuElement, kernel_wgsl, tensor::WgpuTensor};

kernel_wgsl!(UnaryScalarRaw, "../template/unary_scalar.wgsl");
kernel_wgsl!(
    UnaryScalarInplaceRaw,
    "../template/unary_scalar_inplace.wgsl"
);

/// Creates a unary scalar kernel.
#[macro_export]
macro_rules! unary_scalar {
    (
        $struct:ident,
        ops $ops:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::UnaryScalarRaw::source_template().register(
                    "body",
                    format!("output[global_id.x] = lhs[global_id.x] {} rhs;", $ops),
                )
            }
        }
    };

    (
        $struct:ident,
        func $func:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::UnaryScalarRaw::source_template().register(
                    "body",
                    format!("output[global_id.x] = {}(lhs[global_id.x], rhs);", $func),
                )
            }
        }
    };
}

/// Creates a unary scalar inplace kernel.
#[macro_export]
macro_rules! unary_scalar_inplace {
    (
        $struct:ident,
        ops $ops:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::UnaryScalarInplaceRaw::source_template().register(
                    "body",
                    format!("lhs[global_id.x] = lhs[global_id.x] {} rhs;", $ops),
                )
            }
        }
    };

    (
        $struct:ident,
        func $func:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::UnaryScalarInplaceRaw::source_template().register(
                    "body",
                    format!("lhs[global_id.x] = {}(lhs[global_id.x], rhs);", $func),
                )
            }
        }
    };
}

pub fn unary_scalar<K: StaticKernel, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    scalar: E,
) -> WgpuTensor<E, D> {
    let buffer = lhs
        .context
        .create_buffer(lhs.shape.num_elements() * core::mem::size_of::<E>());
    let output = WgpuTensor::new(lhs.context.clone(), lhs.shape, buffer);
    let kernel = lhs
        .context
        .compile_static::<KernelSettings<K, E, i32, 256, 1, 1>>();
    let rhs_buffer = lhs.context.create_buffer_with_data(E::as_bytes(&[scalar]));

    lhs.context.execute(
        WorkGroup::new(
            f32::ceil(output.shape.num_elements() as f32 / 256_f32) as u32,
            1,
            1,
        ),
        kernel,
        &[&lhs.buffer, &rhs_buffer, &output.buffer],
    );

    output
}

pub fn unary_scalar_inplace<K: StaticKernel, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    scalar: E,
) -> WgpuTensor<E, D> {
    let kernel = lhs
        .context
        .compile_static::<KernelSettings<K, E, i32, 256, 1, 1>>();
    let rhs_buffer = lhs.context.create_buffer_with_data(E::as_bytes(&[scalar]));

    lhs.context.execute(
        WorkGroup::new(
            f32::ceil(lhs.shape.num_elements() as f32 / 256_f32) as u32,
            1,
            1,
        ),
        kernel,
        &[&lhs.buffer, &rhs_buffer],
    );

    lhs
}
