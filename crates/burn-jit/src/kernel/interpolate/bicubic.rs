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
struct InterpolateBicubicEagerKernel<R, E> {
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

struct InterpolateBicubicShader {
    input: Variable,
    output: Variable,
}

impl InterpolateBicubicShader {
    pub(crate) fn expand(self, scope: &mut Scope) {
        let input = self.input;
        let output = self.output;
        let id = Variable::Id;
    }
}

impl<R: Runtime, E: JitElement> DynamicKernelSource for InterpolateBicubicEagerKernel<R, E> {
    fn source(&self) -> SourceTemplate {
        let mut scope = Scope::root();
        let item = E::gpu_elem().into();

        let input = Variable::GlobalInputArray(0, item);
        let output = Variable::GlobalOutputArray(0, item);

        InterpolateBicubicShader { input, output }.expand(&mut scope);

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
