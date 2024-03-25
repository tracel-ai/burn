use std::marker::PhantomData;

use crate::{
    codegen::{execute_dynamic, EagerHandle, WorkgroupLaunch},
    kernel::{DynamicKernelSource, SourceTemplate},
    tensor::JitTensor,
    JitElement, Runtime,
};

#[derive(new)]
struct InterpolateBicubicEagerKernel<R, E> {
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

impl<R: Runtime, E: JitElement> DynamicKernelSource for InterpolateBicubicEagerKernel<R, E> {
    fn source(&self) -> SourceTemplate {
        todo!()
    }

    fn id(&self) -> String {
        format!("{:?}", core::any::TypeId::of::<Self>())
    }
}

pub(crate) fn interpolate_bicubic_launch<R: Runtime, E: JitElement>(
    input: JitTensor<R, E, 4>,
    output: JitTensor<R, E, 4>,
) -> JitTensor<R, E, 4> {
    let kernel = InterpolateBicubicEagerKernel::new();

    execute_dynamic::<R, InterpolateBicubicEagerKernel<R, E>, u32>(
        &[EagerHandle::new(
            &input.handle,
            &input.strides,
            &input.shape.dims,
        )],
        &[EagerHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )],
        None,
        kernel,
        WorkgroupLaunch::Output { pos: 0 },
        input.client,
    );

    output
}
