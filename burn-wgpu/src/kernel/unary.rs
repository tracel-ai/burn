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
                let line = format!("output[global_id.x] = {}(lhs[global_id.x]);", $func);
                source.replace("BODY", &line)
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
                let line = format!("lhs[global_id.x] = {}(lhs[global_id.x]);", $func);
                source.replace("BODY", &line)
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
    lhs: WGPUTensor<E, D>,
) -> WGPUTensor<E, D> {
    let buffer = lhs
        .context
        .create_buffer(lhs.shape.num_elements() * core::mem::size_of::<E>());
    let output = WGPUTensor::new(lhs.context.clone(), lhs.shape, Arc::new(buffer));
    let kernel = lhs
        .context
        .compile::<KernelSettings<K, E, i32, 256, 1, 1>>();

    lhs.context.execute(
        &WorkGroup::new(
            f32::ceil(output.shape.num_elements() as f32 / 256_f32) as u32,
            1,
            1,
        ),
        &kernel,
        &[&lhs.buffer, &output.buffer],
    );

    output
}

pub fn unary_inplace<K: KernelGenerator, E: WGPUElement, const D: usize>(
    lhs: WGPUTensor<E, D>,
) -> WGPUTensor<E, D> {
    let kernel = lhs
        .context
        .compile::<KernelSettings<K, E, i32, 256, 1, 1>>();

    lhs.context.execute(
        &WorkGroup::new(
            f32::ceil(lhs.shape.num_elements() as f32 / 256_f32) as u32,
            1,
            1,
        ),
        &kernel,
        &[&lhs.buffer],
    );

    lhs
}
