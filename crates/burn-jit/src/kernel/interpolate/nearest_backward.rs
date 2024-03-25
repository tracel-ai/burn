use std::marker::PhantomData;

use crate::{
    codegen::{execute_dynamic, EagerHandle, WorkgroupLaunch},
    kernel::{DynamicKernelSource, SourceTemplate},
    tensor::JitTensor,
    JitElement, Runtime,
};

#[derive(new)]
struct InterpolateNearestBackwardEagerKernel<R, E> {
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

impl<R: Runtime, E: JitElement> DynamicKernelSource
    for InterpolateNearestBackwardEagerKernel<R, E>
{
    fn source(&self) -> SourceTemplate {
        todo!()
    }

    fn id(&self) -> String {
        format!("{:?}", core::any::TypeId::of::<Self>())
    }
}

pub(crate) fn interpolate_nearest_backward_launch<R: Runtime, E: JitElement>(
    out_grad: JitTensor<R, E, 4>,
    output: JitTensor<R, E, 4>,
) -> JitTensor<R, E, 4> {
    let kernel = InterpolateNearestBackwardEagerKernel::new();

    execute_dynamic::<R, InterpolateNearestBackwardEagerKernel<R, E>, u32>(
        &[EagerHandle::new(
            &out_grad.handle,
            &out_grad.strides,
            &out_grad.shape.dims,
        )],
        &[EagerHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )],
        None,
        kernel,
        WorkgroupLaunch::Output { pos: 0 },
        out_grad.client,
    );

    output
}
