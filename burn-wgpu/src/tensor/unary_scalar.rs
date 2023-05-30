use crate::{
    context::{WorkGroup, WorkGroupSize},
    element::WGPUElement,
    kernel::{KernelTemplate, RenderOptions},
    kernel_wgsl,
    tensor::WGPUTensor,
};

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
        $ops:expr
    ) => {
        pub struct $struct {
            raw: $crate::tensor::UnaryScalarRaw,
        }

        impl $crate::tensor::UnaryScalarOps for $struct {
            fn template(options: $crate::kernel::RenderOptions) -> Self {
                Self {
                    raw: $crate::tensor::UnaryScalarRaw::new(options),
                }
            }
        }

        impl KernelTemplate for $struct {
            fn id(&self) -> String {
                let id = self.raw.id();
                id + $ops
            }

            fn render(&self) -> String {
                let source = self.raw.render();
                source.replace("OPS", $ops)
            }
        }
    };
}

#[macro_export]
macro_rules! unary_scalar_inplace {
    (
        $struct:ident,
        $ops:expr
    ) => {
        pub struct $struct {
            raw: $crate::tensor::UnaryScalarInplaceRaw,
        }

        impl $crate::tensor::UnaryScalarOps for $struct {
            fn template(options: $crate::kernel::RenderOptions) -> Self {
                Self {
                    raw: $crate::tensor::UnaryScalarInplaceRaw::new(options),
                }
            }
        }

        impl KernelTemplate for $struct {
            fn id(&self) -> String {
                let id = self.raw.id();
                id + $ops
            }

            fn render(&self) -> String {
                let source = self.raw.render();
                source.replace("OPS", $ops)
            }
        }
    };
}

pub trait UnaryScalarOps: KernelTemplate {
    fn template(options: RenderOptions) -> Self;
}

pub fn unary_scalar<K: UnaryScalarOps, E: WGPUElement, const D: usize>(
    lhs: WGPUTensor<E, D>,
    scalar: E,
) -> WGPUTensor<E, D> {
    let buffer = lhs
        .context
        .create_buffer(lhs.shape.num_elements() * core::mem::size_of::<E>());
    let output = WGPUTensor::new(lhs.context.clone(), lhs.shape, Arc::new(buffer));

    let kernel = lhs.context.compile(K::template(RenderOptions::new(
        WorkGroupSize::new(256, 1, 1),
        Some(E::type_name().to_string()),
        None,
    )));
    let rhs_buffer = lhs
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&[scalar]));

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

pub fn unary_scalar_inplace<K: UnaryScalarOps, E: WGPUElement, const D: usize>(
    lhs: WGPUTensor<E, D>,
    scalar: E,
) -> WGPUTensor<E, D> {
    let kernel = lhs.context.compile(K::template(RenderOptions::new(
        WorkGroupSize::new(256, 1, 1),
        Some(E::type_name().to_string()),
        None,
    )));
    let rhs_buffer = lhs
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&[scalar]));

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
