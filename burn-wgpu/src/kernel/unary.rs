use super::{KernelGenerator, KernelSettings};
use crate::{context::WorkGroup, element::WGPUElement, kernel_wgsl, tensor::WGPUTensor};
use std::sync::Arc;

kernel_wgsl!(UnaryRaw, "../template/unary.wgsl");
kernel_wgsl!(UnaryInplaceRaw, "../template/unary_inplace.wgsl");

#[macro_export]
macro_rules! unary {
    (
        $struct:ident,
        func $func:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::KernelGenerator for $struct {
            type Source = String;

            fn generate() -> Self::Source {
                let source = $crate::kernel::UnaryRaw::generate().to_string();
                let body = format!("output[global_id.x] = {}(input[global_id.x]);", $func);
                source.replace("BODY", &body)
            }
        }
    };
    (
        $struct:ident,
        body $body:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::KernelGenerator for $struct {
            type Source = String;

            fn generate() -> Self::Source {
                let source = $crate::kernel::UnaryRaw::generate().to_string();
                source.replace("BODY", $body)
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

        impl $crate::kernel::KernelGenerator for $struct {
            type Source = String;

            fn generate() -> Self::Source {
                let source = $crate::kernel::UnaryInplaceRaw::generate().to_string();
                let body = format!("input[global_id.x] = {}(input[global_id.x]);", $func);
                source.replace("BODY", &body)
            }
        }
    };
    (
        $struct:ident,
        body $body:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::KernelGenerator for $struct {
            type Source = String;

            fn generate() -> Self::Source {
                let source = $crate::kernel::UnaryInplaceRaw::generate().to_string();
                source.replace("BODY", $body)
            }
        }
    };
}

pub fn unary<K: KernelGenerator, E: WGPUElement, const D: usize>(
    input: WGPUTensor<E, D>,
) -> WGPUTensor<E, D> {
    let buffer = input
        .context
        .create_buffer(input.shape.num_elements() * core::mem::size_of::<E>());
    let output = WGPUTensor::new(input.context.clone(), input.shape, Arc::new(buffer));
    let kernel = input
        .context
        .compile::<KernelSettings<K, E, i32, 256, 1, 1>>();

    input.context.execute(
        &WorkGroup::new(
            f32::ceil(output.shape.num_elements() as f32 / 256_f32) as u32,
            1,
            1,
        ),
        &kernel,
        &[&input.buffer, &output.buffer],
    );

    output
}

pub fn unary_inplace<K: KernelGenerator, E: WGPUElement, const D: usize>(
    input: WGPUTensor<E, D>,
) -> WGPUTensor<E, D> {
    let kernel = input
        .context
        .compile::<KernelSettings<K, E, i32, 256, 1, 1>>();

    input.context.execute(
        &WorkGroup::new(
            f32::ceil(input.shape.num_elements() as f32 / 256_f32) as u32,
            1,
            1,
        ),
        &kernel,
        &[&input.buffer],
    );

    input
}
