use crate::{
    context::{WorkGroup, WorkGroupSize},
    element::WGPUElement,
    kernel::{KernelTemplate, RenderOptions},
    kernel_wgsl,
    tensor::WGPUTensor,
};
use burn_tensor::Shape;
use num_traits::ToPrimitive;
use std::sync::Arc;

kernel_wgsl!(ElemwiseRaw, "../template/elemwise.wgsl");
kernel_wgsl!(ElemwiseInplaceRaw, "../template/elemwise_inplace.wgsl");

#[macro_export]
macro_rules! kernel_elemwise {
    (
        $struct:ident,
        $ops:expr
    ) => {
        pub struct $struct {
            raw: $crate::tensor::elemwise::ElemwiseRaw,
        }

        impl $crate::tensor::elemwise::ElemwiseOps for $struct {
            fn template(options: $crate::kernel::RenderOptions) -> Self {
                Self {
                    raw: $crate::tensor::elemwise::ElemwiseRaw::new(options),
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
macro_rules! kernel_elemwise_inplace {
    (
        $struct:ident,
        $ops:expr
    ) => {
        pub struct $struct {
            raw: $crate::tensor::elemwise::ElemwiseInplaceRaw,
        }

        impl $crate::tensor::elemwise::ElemwiseOps for $struct {
            fn template(options: $crate::kernel::RenderOptions) -> Self {
                Self {
                    raw: $crate::tensor::elemwise::ElemwiseInplaceRaw::new(options),
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

pub trait ElemwiseOps: KernelTemplate {
    fn template(options: RenderOptions) -> Self;
}

pub fn execute_elemwise<K: ElemwiseOps, E: WGPUElement, const D: usize>(
    lhs: WGPUTensor<E, D>,
    rhs: WGPUTensor<E, D>,
) -> WGPUTensor<E, D> {
    if lhs.context.device_wgpu != rhs.context.device_wgpu {
        panic!(
            "Both tensors should be on the same device {:?} != {:?}",
            lhs.context.device_wgpu, rhs.context.device_wgpu
        );
    }

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

    let kernel = lhs.context.compile(K::template(RenderOptions::new(
        WorkGroupSize::new(256, 1, 1),
        Some(E::type_name().to_string()),
        None,
    )));
    let mut info: Vec<u32> = vec![D.to_u32().unwrap()];
    lhs.strides
        .into_iter()
        .for_each(|v| info.push(v.to_u32().unwrap()));
    rhs.strides
        .into_iter()
        .for_each(|v| info.push(v.to_u32().unwrap()));
    lhs.shape
        .dims
        .into_iter()
        .for_each(|v| info.push(v.to_u32().unwrap()));
    rhs.shape
        .dims
        .into_iter()
        .for_each(|v| info.push(v.to_u32().unwrap()));
    let info_buffers = lhs
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    lhs.context.execute(
        &WorkGroup::new(
            f32::ceil(output.shape.num_elements() as f32 / 256 as f32) as u32,
            1,
            1,
        ),
        &kernel,
        &[&lhs.buffer, &rhs.buffer, &output.buffer, &info_buffers],
    );

    output
}

pub fn execute_elemwise_inplace<K: ElemwiseOps, E: WGPUElement, const D: usize>(
    lhs: WGPUTensor<E, D>,
    rhs: WGPUTensor<E, D>,
) -> WGPUTensor<E, D> {
    if lhs.context.device_wgpu != rhs.context.device_wgpu {
        panic!(
            "Both tensors should be on the same device {:?} != {:?}",
            lhs.context.device_wgpu, rhs.context.device_wgpu
        );
    }

    let mut shape_out = [0; D];
    lhs.shape
        .dims
        .iter()
        .zip(rhs.shape.dims.iter())
        .enumerate()
        .for_each(|(index, (dim_lhs, dim_rhs))| {
            shape_out[index] = usize::max(*dim_lhs, *dim_rhs);
        });

    let kernel = lhs.context.compile(K::template(RenderOptions::new(
        WorkGroupSize::new(256, 1, 1),
        Some(E::type_name().to_string()),
        None,
    )));
    let mut info: Vec<u32> = vec![D.to_u32().unwrap()];
    rhs.strides
        .into_iter()
        .for_each(|v| info.push(v.to_u32().unwrap()));
    rhs.shape
        .dims
        .into_iter()
        .for_each(|v| info.push(v.to_u32().unwrap()));
    let info_buffers = lhs
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    lhs.context.execute(
        &WorkGroup::new(
            f32::ceil(lhs.shape.num_elements() as f32 / 256 as f32) as u32,
            1,
            1,
        ),
        &kernel,
        &[&lhs.buffer, &rhs.buffer, &info_buffers],
    );

    lhs.clone()
}
