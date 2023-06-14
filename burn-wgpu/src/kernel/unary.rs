use super::{KernelSettings, StaticKernelGenerator};
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

        impl $crate::kernel::StaticKernelGenerator for $struct {
            fn source() -> $crate::kernel::Source {
                let source = $crate::kernel::UnaryRaw::source();
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

        impl $crate::kernel::StaticKernelGenerator for $struct {
            fn source() -> $crate::kernel::Source {
                $crate::kernel::UnaryRaw::source().register("body", $body)
            }
        }
    };
    (
        $struct:ident,
        func $func:expr,
        include $file:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernelGenerator for $struct {
            fn source() -> $crate::kernel::Source {
                $crate::kernel::UnaryRaw::source()
                    .register(
                        "body",
                        format!("output[global_id.x] = {}(input[global_id.x]);", $func),
                    )
                    .add_source(include_str!($file))
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

        impl $crate::kernel::StaticKernelGenerator for $struct {
            fn source() -> $crate::kernel::Source {
                $crate::kernel::UnaryInplaceRaw::source().register(
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

        impl $crate::kernel::StaticKernelGenerator for $struct {
            fn source() -> $crate::kernel::Source {
                $crate::kernel::UnaryInplaceRaw::source().register("body", $body)
            }
        }
    };
    (
        $struct:ident,
        func $func:expr,
        include $file:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernelGenerator for $struct {
            fn source() -> $crate::kernel::Source {
                $crate::kernel::UnaryInplaceRaw::source()
                    .register(
                        "body",
                        format!("input[global_id.x] = {}(input[global_id.x]);", $func),
                    )
                    .add_source(include_str!($file))
            }
        }
    };
}

pub fn unary<K: StaticKernelGenerator, E: WgpuElement, const D: usize>(
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

pub fn unary_inplace<K: StaticKernelGenerator, E: WgpuElement, const D: usize>(
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
