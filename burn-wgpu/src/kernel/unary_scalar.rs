use super::{KernelSettings, StaticKernelGenerator};
use crate::{context::WorkGroup, element::WgpuElement, kernel_wgsl, tensor::WgpuTensor};

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

        impl $crate::kernel::StaticKernelGenerator for $struct {
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

        impl $crate::kernel::StaticKernelGenerator for $struct {
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

        impl $crate::kernel::StaticKernelGenerator for $struct {
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

        impl $crate::kernel::StaticKernelGenerator for $struct {
            type Source = String;

            fn generate() -> Self::Source {
                let source = $crate::kernel::UnaryScalarInplaceRaw::generate().to_string();
                let body = format!("lhs[global_id.x] = {}(lhs[global_id.x], rhs);", $func);

                source.replace("BODY", &body)
            }
        }
    };
}

pub fn unary_scalar<K: StaticKernelGenerator, E: WgpuElement, const D: usize>(
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

pub fn unary_scalar_inplace<K: StaticKernelGenerator, E: WgpuElement, const D: usize>(
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
