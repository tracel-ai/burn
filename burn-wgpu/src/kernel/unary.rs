use super::{KernelSettings, StaticKernel};
use crate::{context::WorkGroup, element::WgpuElement, kernel_wgsl, tensor::WgpuTensor};

kernel_wgsl!(UnaryRaw, "../template/unary.wgsl");
kernel_wgsl!(UnaryInplaceRaw, "../template/unary_inplace.wgsl");

#[macro_export]
macro_rules! unary {
    (
        $struct:ident,
        func $func:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                let source = $crate::kernel::UnaryRaw::source_template();
                source.register(
                    "body",
                    format!("output[global_id.x] = {}(input[global_id.x]);", $func),
                )
            }
        }
    };
    (
        $struct:ident,
        body $body:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::UnaryRaw::source_template().register("body", $body)
            }
        }
    };
    (
        $struct:ident,
        func $func:expr,
        include $file:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::UnaryRaw::source_template()
                    .register(
                        "body",
                        format!("output[global_id.x] = {}(input[global_id.x]);", $func),
                    )
                    .add_template(include_str!($file))
            }
        }
    };
}

#[macro_export]
macro_rules! unary_inplace {
    (
        $struct:ident,
        func $func:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::UnaryInplaceRaw::source_template().register(
                    "body",
                    format!("input[global_id.x] = {}(input[global_id.x]);", $func),
                )
            }
        }
    };
    (
        $struct:ident,
        body $body:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::UnaryInplaceRaw::source_template().register("body", $body)
            }
        }
    };
    (
        $struct:ident,
        func $func:expr,
        include $file:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::UnaryInplaceRaw::source_template()
                    .register(
                        "body",
                        format!("input[global_id.x] = {}(input[global_id.x]);", $func),
                    )
                    .add_template(include_str!($file))
            }
        }
    };
}

pub fn unary<K: StaticKernel, E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    let buffer = input
        .context
        .create_buffer(input.shape.num_elements() * core::mem::size_of::<E>());
    let mut output = WgpuTensor::new(input.context.clone(), input.shape, buffer);
    // Since we don't handle the stride inside the kernel, the output tensor have the same strides
    // as the input tensor. It might not be in the default format.
    output.strides = input.strides;

    let kernel = input
        .context
        .compile_static::<KernelSettings<K, E, i32, 256, 1, 1>>();

    input.context.execute(
        WorkGroup::new(
            f32::ceil(output.shape.num_elements() as f32 / 256_f32) as u32,
            1,
            1,
        ),
        kernel,
        &[&input.buffer, &output.buffer],
    );

    output
}

pub fn unary_inplace<K: StaticKernel, E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    let kernel = input
        .context
        .compile_static::<KernelSettings<K, E, i32, 256, 1, 1>>();

    input.context.execute(
        WorkGroup::new(
            f32::ceil(input.shape.num_elements() as f32 / 256_f32) as u32,
            1,
            1,
        ),
        kernel,
        &[&input.buffer],
    );

    input
}
