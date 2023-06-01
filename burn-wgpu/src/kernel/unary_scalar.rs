use super::{KernelGenerator, KernelSettings};
use crate::{context::WorkGroup, element::WGPUElement, kernel_wgsl, tensor::WGPUTensor};
use std::sync::Arc;

kernel_wgsl!(UnaryScalarRaw, "../template/unary_scalar.wgsl");
kernel_wgsl!(
    UnaryScalarInplaceRaw,
    "../template/unary_scalar_inplace.wgsl"
);

#[macro_export]
macro_rules! unary_scalar {
    (
        $struct:ident,
        ops $ops:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::KernelGenerator for $struct {
            type Source = String;

            fn generate() -> Self::Source {
                let source = $crate::kernel::UnaryScalarRaw::generate().to_string();
                let body = format!("output[global_id.x] = lhs[global_id.x] {} rhs;", $ops);

                source.replace("BODY", &body)
            }
        }
    };

    (
        $struct:ident,
        func $func:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::KernelGenerator for $struct {
            type Source = String;

            fn generate() -> Self::Source {
                let source = $crate::kernel::UnaryScalarRaw::generate().to_string();
                let body = format!("output[global_id.x] = {}(lhs[global_id.x], rhs);", $func);

                source.replace("BODY", &body)
            }
        }
    };
}

#[macro_export]
macro_rules! unary_scalar_inplace {
    (
        $struct:ident,
        ops $ops:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::KernelGenerator for $struct {
            type Source = String;

            fn generate() -> Self::Source {
                let source = $crate::kernel::UnaryScalarInplaceRaw::generate().to_string();
                let body = format!("lhs[global_id.x] = lhs[global_id.x] {} rhs;", $ops);

                source.replace("BODY", &body)
            }
        }
    };

    (
        $struct:ident,
        func $func:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::KernelGenerator for $struct {
            type Source = String;

            fn generate() -> Self::Source {
                let source = $crate::kernel::UnaryScalarInplaceRaw::generate().to_string();
                let body = format!("lhs[global_id.x] = {}(lhs[global_id.x], rhs);", $func);

                source.replace("BODY", &body)
            }
        }
    };
}

pub fn unary_scalar<K: KernelGenerator, E: WGPUElement, const D: usize>(
    lhs: WGPUTensor<E, D>,
    scalar: E,
) -> WGPUTensor<E, D> {
    let buffer = lhs
        .context
        .create_buffer(lhs.shape.num_elements() * core::mem::size_of::<E>());
    let output = WGPUTensor::new(lhs.context.clone(), lhs.shape, Arc::new(buffer));
    let kernel = lhs
        .context
        .compile::<KernelSettings<K, E, i32, 256, 1, 1>>();
    let rhs_buffer = lhs.context.create_buffer_with_data(E::as_bytes(&[scalar]));

    lhs.context.execute(
        &WorkGroup::new(
            f32::ceil(output.shape.num_elements() as f32 / 256_f32) as u32,
            1,
            1,
        ),
        &kernel,
        &[&lhs.buffer, &rhs_buffer, &output.buffer],
    );

    output
}

pub fn unary_scalar_inplace<K: KernelGenerator, E: WGPUElement, const D: usize>(
    lhs: WGPUTensor<E, D>,
    scalar: E,
) -> WGPUTensor<E, D> {
    let kernel = lhs
        .context
        .compile::<KernelSettings<K, E, i32, 256, 1, 1>>();
    let rhs_buffer = lhs.context.create_buffer_with_data(E::as_bytes(&[scalar]));

    lhs.context.execute(
        &WorkGroup::new(
            f32::ceil(lhs.shape.num_elements() as f32 / 256_f32) as u32,
            1,
            1,
        ),
        &kernel,
        &[&lhs.buffer, &rhs_buffer],
    );

    lhs
}
