use std::marker::PhantomData;

use crate::{
    codegen::{
        Compilation, CompilationInfo, CompilationSettings, EagerHandle, Execution, InputInfo,
        OutputInfo, WorkgroupLaunch,
    },
    gpu::{gpu, ComputeShader, Elem, Item, Scope, Variable, Visibility},
    kernel::GpuComputeShaderPhase,
    tensor::JitTensor,
    JitElement, Runtime,
};

/// Cast a bool tensor to the given element type.
///
/// This alternative to cast is necessary because bool are represented as u32
/// where any non-zero value means true. Depending how it was created
/// it may hold an uncanny bit combination. Naively casting it would not
/// necessarily yield 0 or 1.
pub fn bool_cast<R: Runtime, EO: JitElement, const D: usize>(
    tensor: JitTensor<R, u32, D>,
) -> JitTensor<R, EO, D> {
    let kernel = BoolCastEagerKernel::<R, EO>::new();
    let num_elems = tensor.shape.num_elements();
    let buffer = tensor.client.empty(num_elems * core::mem::size_of::<EO>());
    let output = JitTensor::new(
        tensor.client.clone(),
        tensor.device,
        tensor.shape.clone(),
        buffer,
    );

    Execution::start(kernel, tensor.client)
        .inputs(&[EagerHandle::<R>::new(
            &tensor.handle,
            &tensor.strides,
            &tensor.shape.dims,
        )])
        .outputs(&[EagerHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )])
        .execute(WorkgroupLaunch::Output { pos: 0 });

    output
}

pub(crate) struct BoolCastShader {
    tensor: Variable,
    output: Variable,
}

#[derive(new)]
pub(crate) struct BoolCastEagerKernel<R: Runtime, EO: JitElement> {
    _runtime: PhantomData<R>,
    _elem_out: PhantomData<EO>,
}

impl<R: Runtime, EO: JitElement> GpuComputeShaderPhase for BoolCastEagerKernel<R, EO> {
    fn compile(&self) -> ComputeShader {
        let mut scope = Scope::root();
        let item_input = Item::Scalar(Elem::Bool);
        let item_output = EO::gpu_elem().into();

        let tensor = Variable::GlobalInputArray(0, item_input);
        let output = Variable::GlobalOutputArray(0, item_output);

        BoolCastShader { tensor, output }.expand(&mut scope);

        scope.write_global_custom(output);

        let tensor = InputInfo::Array {
            item: item_input,
            visibility: Visibility::Read,
        };

        let out = OutputInfo::Array { item: item_output };

        let info = CompilationInfo {
            inputs: vec![tensor],
            outputs: vec![out],
            scope,
        };

        let settings = CompilationSettings::default();
        Compilation::new(info).compile(settings)
    }

    fn id(&self) -> String {
        format!("{:?}", core::any::TypeId::of::<Self>())
    }
}

impl BoolCastShader {
    pub(crate) fn expand(self, scope: &mut Scope) {
        let tensor = self.tensor;
        let id = Variable::Id;
        let output = self.output;

        let represents_true = scope.create_local(Elem::Bool);
        gpu!(scope, represents_true = tensor[id]);
        gpu!(scope, if(represents_true).then(|scope|{
            gpu!(scope, output[id] = 1);
        }).else(|scope|{
            gpu!(scope, output[id] = 0);
        }));
    }
}
