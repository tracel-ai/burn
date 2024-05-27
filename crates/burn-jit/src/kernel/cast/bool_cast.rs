use crate::{kernel::GpuComputeShaderPhase, tensor::JitTensor, JitElement, JitRuntime};
use burn_cube::{
    cpa,
    dialect::{ComputeShader, Elem, Item, Scope, Variable, Visibility},
    Compilation, CompilationInfo, CompilationSettings, Execution, InputInfo, OutputInfo,
    TensorHandle, WorkgroupLaunch,
};
use std::marker::PhantomData;

/// Cast a bool tensor to the given element type.
///
/// This alternative to cast is necessary because bool are represented as u32
/// where any non-zero value means true. Depending how it was created
/// it may hold an uncanny bit combination. Naively casting it would not
/// necessarily yield 0 or 1.
pub fn bool_cast<R: JitRuntime, EO: JitElement, const D: usize>(
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
        .inputs(&[TensorHandle::<R>::new(
            &tensor.handle,
            &tensor.strides,
            &tensor.shape.dims,
        )])
        .outputs(&[TensorHandle::new(
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
pub(crate) struct BoolCastEagerKernel<R: JitRuntime, EO: JitElement> {
    _runtime: PhantomData<R>,
    _elem_out: PhantomData<EO>,
}

impl<R: JitRuntime, EO: JitElement> GpuComputeShaderPhase for BoolCastEagerKernel<R, EO> {
    fn compile(&self) -> ComputeShader {
        let mut scope = Scope::root();
        let item_input = Item::new(Elem::Bool);
        let item_output = EO::cube_elem().into();

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
        cpa!(scope, represents_true = tensor[id]);
        cpa!(scope, if(represents_true).then(|scope|{
            cpa!(scope, output[id] = 1);
        }).else(|scope|{
            cpa!(scope, output[id] = 0);
        }));
    }
}
