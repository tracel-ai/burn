use super::{
    DynamicKernelSource, KernelSettings, SourceTemplate, StaticKernelSource, WORKGROUP_DEFAULT,
};
use crate::{
    codegen::{
        dialect::gpu::{gpu, Elem, Item, Scope, Variable, Visibility},
        execute_dynamic, Compilation, CompilationInfo, CompilationSettings, Compiler, EagerHandle,
        InputInfo, OutputInfo, WorkgroupLaunch,
    },
    compute::StaticKernel,
    element::JitElement,
    kernel::elemwise_workgroup,
    kernel_wgsl,
    tensor::JitTensor,
    Runtime,
};
use std::{any::TypeId, marker::PhantomData};

kernel_wgsl!(CastRaw, "../template/cast.wgsl");

struct Cast<InputElem: JitElement, OutputElem: JitElement> {
    _i: PhantomData<InputElem>,
    _o: PhantomData<OutputElem>,
}

impl<InputElem: JitElement, OutputElem: JitElement> StaticKernelSource
    for Cast<InputElem, OutputElem>
{
    fn source() -> SourceTemplate {
        CastRaw::source()
            .register("input_elem", InputElem::type_name())
            .register("output_elem", OutputElem::type_name())
    }
}

/// Cast a tensor to the given element type.
pub fn cast<R: Runtime, InputElem: JitElement, OutputElem: JitElement, const D: usize>(
    tensor: JitTensor<R, InputElem, D>,
) -> JitTensor<R, OutputElem, D> {
    if TypeId::of::<InputElem>() == TypeId::of::<OutputElem>() {
        return JitTensor::new(tensor.client, tensor.device, tensor.shape, tensor.handle);
    }

    let num_elems = tensor.shape.num_elements();
    let kernel = StaticKernel::<
        KernelSettings<
            Cast<InputElem, OutputElem>,
            f32,
            i32,
            WORKGROUP_DEFAULT,
            WORKGROUP_DEFAULT,
            1,
        >,
    >::new(elemwise_workgroup(num_elems, WORKGROUP_DEFAULT));

    let handle = tensor
        .client
        .empty(num_elems * core::mem::size_of::<OutputElem>());
    let output = JitTensor::new(
        tensor.client.clone(),
        tensor.device,
        tensor.shape.clone(),
        handle,
    );

    tensor
        .client
        .execute(Box::new(kernel), &[&tensor.handle, &output.handle]);

    output
}

/// Cast a bool tensor to the given element type.
///
/// This alternative to cast is necessary because bool are represented as u32
/// where any non-zero value means true. Depending how it was created
/// it may hold an uncanny bit combination. Naively casting it would not
/// necessarily yield 0 or 1.
pub fn bool_cast<R: Runtime, EO: JitElement, const D: usize>(
    tensor: JitTensor<R, u32, D>,
) -> JitTensor<R, EO, D> {
    let kernel = BoolCastEagerKernel::new();
    let num_elems = tensor.shape.num_elements();
    let buffer = tensor.client.empty(num_elems * core::mem::size_of::<EO>());
    let output = JitTensor::new(
        tensor.client.clone(),
        tensor.device,
        tensor.shape.clone(),
        buffer,
    );

    execute_dynamic::<R, BoolCastEagerKernel<R, EO>, u32>(
        &[EagerHandle::new(
            &tensor.handle,
            &tensor.strides,
            &tensor.shape.dims,
        )],
        &[EagerHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )],
        None,
        kernel,
        WorkgroupLaunch::Output { pos: 0 },
        tensor.client,
    );

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

impl<R: Runtime, EO: JitElement> DynamicKernelSource for BoolCastEagerKernel<R, EO> {
    fn source(&self) -> crate::kernel::SourceTemplate {
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
        let shader = Compilation::new(info).compile(settings);
        let shader = <R::Compiler as Compiler>::compile(shader);
        SourceTemplate::new(shader.to_string())
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
