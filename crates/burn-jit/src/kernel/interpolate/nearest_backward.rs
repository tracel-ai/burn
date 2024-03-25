use std::marker::PhantomData;

use crate::{
    codegen::{
        execute_dynamic, Compilation, CompilationInfo, CompilationSettings, EagerHandle, InputInfo,
        OutputInfo, WorkgroupLaunch,
    },
    gpu::{Scope, Variable, Visibility},
    kernel::{DynamicKernelSource, SourceTemplate},
    tensor::JitTensor,
    Compiler, JitElement, Runtime,
};

#[derive(new)]
struct InterpolateNearestBackwardEagerKernel<R, E> {
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

struct InterpolateNearestBackwardShader {
    out_grad: Variable,
    output: Variable,
}

impl InterpolateNearestBackwardShader {
    pub(crate) fn expand(self, scope: &mut Scope) {
        let out_grad = self.out_grad;
        let output = self.output;
        let id = Variable::Id;
    }
}

impl<R: Runtime, E: JitElement> DynamicKernelSource
    for InterpolateNearestBackwardEagerKernel<R, E>
{
    fn source(&self) -> SourceTemplate {
        let mut scope = Scope::root();
        let item = E::gpu_elem().into();

        let out_grad = Variable::GlobalInputArray(0, item);
        let output = Variable::GlobalOutputArray(0, item);

        InterpolateNearestBackwardShader { out_grad, output }.expand(&mut scope);

        scope.write_global_custom(output);

        let input = InputInfo::Array {
            item,
            visibility: Visibility::Read,
        };

        let out = OutputInfo::Array { item };

        let info = CompilationInfo {
            inputs: vec![input],
            outputs: vec![out],
            scope,
        };

        let settings = CompilationSettings::default();
        let shader = Compilation::new(info).compile(settings);
        let shader = <R::Compiler as Compiler>::compile(shader);
        SourceTemplate::new(shader.to_string())
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
