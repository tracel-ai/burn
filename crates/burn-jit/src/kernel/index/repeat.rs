use crate::{
    codegen::{
        dialect::gpu::{gpu, Elem, Scope, Variable, Visibility},
        Compilation, CompilationInfo, CompilationSettings, EagerHandle, Execution, InputInfo,
        OutputInfo, WorkgroupLaunch,
    },
    element::JitElement,
    gpu::ComputeShader,
    kernel::GpuComputeShaderPhase,
    tensor::JitTensor,
    Runtime,
};
use std::marker::PhantomData;

pub struct RepeatComputeShader {
    input: Variable,
    output: Variable,
    dim: usize,
    rank: usize,
}

#[derive(new)]
struct RepeatEagerKernel<R: Runtime, E: JitElement> {
    dim: usize,
    rank: usize,
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

impl RepeatComputeShader {
    pub fn expand(self, scope: &mut Scope) {
        let input = self.input;
        let output = self.output;
        let id = Variable::Id;

        let offset_input = scope.zero(Elem::UInt);
        let offset_local = scope.zero(Elem::UInt);

        let stride_input = scope.create_local(Elem::UInt);
        let stride_output = scope.create_local(Elem::UInt);
        let shape = scope.create_local(Elem::UInt);

        for i in 0..self.rank {
            gpu!(scope, stride_input = stride(input, i));
            gpu!(scope, stride_output = stride(output, i));
            if i != self.dim {
                gpu!(scope, shape = shape(output, i));
            } else {
                gpu!(scope, shape = shape(input, i));
            }

            gpu!(scope, offset_local = id / stride_output);
            gpu!(scope, offset_local = offset_local % shape);
            gpu!(scope, offset_local = offset_local * stride_input);
            gpu!(scope, offset_input += offset_local);
        }

        let result = scope.create_local(input.item());
        gpu!(scope, result = input[offset_input]);
        gpu!(scope, output[id] = result);
    }
}
impl<R: Runtime, E: JitElement> GpuComputeShaderPhase for RepeatEagerKernel<R, E> {
    fn compile(&self) -> ComputeShader {
        let mut scope = Scope::root();
        let item = E::gpu_elem().into();

        let input = Variable::GlobalInputArray(0, item);
        let output = Variable::GlobalOutputArray(0, item);

        scope.write_global_custom(output);

        RepeatComputeShader {
            input,
            output,
            rank: self.rank,
            dim: self.dim,
        }
        .expand(&mut scope);

        let input = InputInfo::Array {
            item,
            visibility: Visibility::Read,
        };
        let output = OutputInfo::Array { item };

        let info = CompilationInfo {
            inputs: vec![input],
            outputs: vec![output],
            scope,
        };

        let settings = CompilationSettings::default();
        Compilation::new(info).compile(settings)
    }

    fn id(&self) -> String {
        format!(
            "{:?}d={}r={}",
            core::any::TypeId::of::<Self>(),
            self.dim,
            self.rank
        )
    }
}

pub(crate) fn repeat<R: Runtime, E: JitElement, const D1: usize>(
    input: JitTensor<R, E, D1>,
    dim: usize,
    times: usize,
) -> JitTensor<R, E, D1> {
    let mut shape = input.shape.clone();

    // Create output handle
    shape.dims[dim] *= times;
    let num_elems_output = shape.num_elements();
    let handle = input
        .client
        .empty(num_elems_output * core::mem::size_of::<E>());
    let output = JitTensor::new(
        input.client.clone(),
        input.device.clone(),
        shape.clone(),
        handle,
    );

    let kernel = RepeatEagerKernel::<R, E>::new(dim, D1);

    Execution::start(kernel, input.client)
        .inputs(&[EagerHandle::<R>::new(
            &input.handle,
            &input.strides,
            &input.shape.dims,
        )])
        .outputs(&[EagerHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )])
        .execute(WorkgroupLaunch::Output { pos: 0 });

    output
}
