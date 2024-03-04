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

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::tests::{TestBackend, TestRuntime};
//     use burn_tensor::{Data, Int, Tensor};
//
//     #[test]
//     fn should_cast_int_to_float() {
//         const START: usize = 0;
//         const END: usize = 100;
//
//         let device = Default::default();
//         let tensor = Tensor::<TestBackend, 1, Int>::arange(START as i64..END as i64, &device);
//         let tensor_float = cast::<TestRuntime, i32, f32, 1>(tensor.clone().into_primitive());
//
//         let data_int = tensor.into_data();
//         let data_float = Tensor::<TestBackend, 1>::from_primitive(tensor_float).into_data();
//
//         for i in START..END {
//             assert_eq!(data_int.value[i], i as i32);
//             assert_eq!(data_float.value[i], i as f32);
//         }
//     }
//
//     #[test]
//     fn should_cast_bool_to_int() {
//         let device = Default::default();
//
//         let tensor_1 =
//             Tensor::<TestBackend, 2>::from_floats([[1., 0., 3.], [0., 0., 900.]], &device);
//         let tensor_2: Tensor<TestBackend, 2, Int> = tensor_1.clone().greater_elem(0.0).int();
//
//         assert_eq!(tensor_2.to_data(), Data::from([[1, 0, 1], [0, 0, 1]]))
//     }
//
//     #[test]
//     fn should_cast_bool_to_float() {
//         let device = Default::default();
//
//         let tensor_1 =
//             Tensor::<TestBackend, 2>::from_floats([[1., 0., 3.], [0., 0., 900.]], &device);
//         let tensor_2: Tensor<TestBackend, 2> = tensor_1.clone().greater_elem(0.0).float();
//
//         assert_eq!(tensor_2.to_data(), Data::from([[1., 0., 1.], [0., 0., 1.]]))
//     }
// }
