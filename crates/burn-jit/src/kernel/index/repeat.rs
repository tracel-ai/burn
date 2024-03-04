use crate::{
    codegen::{
        dialect::gpu::{gpu, Elem, Scope, Variable, Visibility},
        execute_dynamic, Compilation, CompilationInfo, CompilationSettings, Compiler, EagerHandle,
        InputInfo, OutputInfo, WorkgroupLaunch,
    },
    element::JitElement,
    kernel::{self, DynamicKernelSource, SourceTemplate},
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
        let shape_output = scope.create_local(Elem::UInt);

        for i in 0..self.rank {
            if i != self.dim {
                gpu!(scope, stride_input = stride(input, i));
                gpu!(scope, stride_output = stride(output, i));
                gpu!(scope, shape_output = shape(output, i));

                gpu!(scope, offset_local = id / stride_output);
                gpu!(scope, offset_local = offset_local % shape_output);
                gpu!(scope, offset_local = offset_local * stride_input);
                gpu!(scope, offset_input += offset_local);
            }
        }

        let result = scope.create_local(input.item());
        gpu!(scope, result = input[offset_input]);
        gpu!(scope, output[id] = result);
    }
}
impl<R: Runtime, E: JitElement> DynamicKernelSource for RepeatEagerKernel<R, E> {
    fn source(&self) -> kernel::SourceTemplate {
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
        let shader = Compilation::new(info).compile(settings);
        let shader = <R::Compiler as Compiler>::compile(shader);
        SourceTemplate::new(shader.to_string())
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
    if shape.dims[dim] != 1 {
        panic!("Can only repeat dimension with dim=1");
    }

    // Create output handle
    shape.dims[dim] = times;
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

    let kernel = RepeatEagerKernel::new(dim, D1);

    execute_dynamic::<R, RepeatEagerKernel<R, E>, E>(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{Distribution, Tensor};

    #[test]
    fn repeat_dim_0_few_times() {
        let tensor =
            Tensor::<TestBackend, 3>::random([1, 6, 6], Distribution::Default, &Default::default());
        let dim = 0;
        let times = 4;
        let tensor_ref =
            Tensor::<ReferenceBackend, 3>::from_data(tensor.to_data(), &Default::default());

        let actual = repeat(tensor.into_primitive(), dim, times);
        let expected = tensor_ref.repeat(dim, times);

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 3>::from_primitive(actual).into_data(),
            3,
        );
    }

    #[test]
    fn repeat_dim_1_few_times() {
        let tensor =
            Tensor::<TestBackend, 3>::random([6, 1, 6], Distribution::Default, &Default::default());
        let dim = 1;
        let times = 4;
        let tensor_ref =
            Tensor::<ReferenceBackend, 3>::from_data(tensor.to_data(), &Default::default());

        let actual = repeat(tensor.into_primitive(), dim, times);
        let expected = tensor_ref.repeat(dim, times);

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 3>::from_primitive(actual).into_data(),
            3,
        );
    }

    #[test]
    fn repeat_dim_2_few_times() {
        let tensor =
            Tensor::<TestBackend, 3>::random([6, 6, 1], Distribution::Default, &Default::default());
        let dim = 2;
        let times = 4;
        let tensor_ref =
            Tensor::<ReferenceBackend, 3>::from_data(tensor.to_data(), &Default::default());

        let actual = repeat(tensor.into_primitive(), dim, times);
        let expected = tensor_ref.repeat(dim, times);

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 3>::from_primitive(actual).into_data(),
            3,
        );
    }

    #[test]
    fn repeat_dim_2_many_times() {
        let tensor = Tensor::<TestBackend, 3>::random(
            [10, 10, 1],
            Distribution::Default,
            &Default::default(),
        );
        let dim = 2;
        let times = 200;
        let tensor_ref =
            Tensor::<ReferenceBackend, 3>::from_data(tensor.to_data(), &Default::default());

        let actual = repeat(tensor.into_primitive(), dim, times);
        let expected = tensor_ref.repeat(dim, times);

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 3>::from_primitive(actual).into_data(),
            3,
        );
    }
}
