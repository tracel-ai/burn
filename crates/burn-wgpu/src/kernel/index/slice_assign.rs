use crate::{
    codegen::{
        dialect::gpu::{gpu, Elem, Scope, Variable, Visibility},
        execute_dynamic, Compilation, CompilationInfo, CompilationSettings, Compiler, EagerHandle,
        InputInfo, WorkgroupLaunch,
    },
    element::JitElement,
    kernel::{DynamicKernelSource, SourceTemplate},
    tensor::JitTensor,
    Runtime,
};
use burn_tensor::ElementConversion;
use std::{marker::PhantomData, ops::Range};

#[derive(new)]
struct SliceAssignEagerKernel<R: Runtime, E: JitElement> {
    rank: usize,
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

pub struct SliceAssignComputeShader {
    input: Variable,
    value: Variable,
    rank: usize,
}

impl SliceAssignComputeShader {
    pub fn expand(self, scope: &mut Scope) {
        let input = self.input;
        let value = self.value;
        let id = Variable::Id;

        let offset_input = scope.zero(Elem::UInt);
        let offset_value = scope.zero(Elem::UInt);

        let offset_local = scope.create_local(Elem::UInt);
        let offset_local_value = scope.create_local(Elem::UInt);
        let offset_local_input = scope.create_local(Elem::UInt);

        let stride_input = scope.create_local(Elem::UInt);
        let stride_value = scope.create_local(Elem::UInt);
        let shape_value = scope.create_local(Elem::UInt);
        let shape_input = scope.create_local(Elem::UInt);
        let range_start = scope.create_local(Elem::UInt);

        for i in 0..self.rank {
            gpu!(scope, stride_input = stride(input, i));
            gpu!(scope, stride_value = stride(value, i));
            gpu!(scope, shape_value = shape(value, i));
            gpu!(scope, shape_input = shape(input, i));
            gpu!(
                scope,
                range_start = cast(Variable::GlobalScalar(i as u16, Elem::UInt))
            );

            gpu!(scope, offset_local = id / stride_value);
            gpu!(scope, offset_local = offset_local % shape_value);

            gpu!(scope, offset_local_value = offset_local * stride_value);
            gpu!(scope, offset_local_input = offset_local + range_start);
            gpu!(
                scope,
                offset_local_input = offset_local_input * stride_input
            );

            gpu!(scope, offset_value += offset_local_value);
            gpu!(scope, offset_input += offset_local_input);
        }

        let result = scope.create_local(input.item());
        gpu!(scope, result = value[offset_value]);
        gpu!(scope, input[offset_input] = result);
    }
}

impl<R: Runtime, E: JitElement> DynamicKernelSource for SliceAssignEagerKernel<R, E> {
    fn source(&self) -> crate::kernel::SourceTemplate {
        let mut scope = Scope::root();
        let item = E::gpu_elem().into();

        let input = Variable::GlobalInputArray(0, item);
        let value = Variable::GlobalInputArray(1, item);

        scope.write_global_custom(input);

        SliceAssignComputeShader {
            input,
            value,
            rank: self.rank,
        }
        .expand(&mut scope);

        let input = InputInfo::Array {
            item,
            visibility: Visibility::ReadWrite,
        };
        let value = InputInfo::Array {
            item,
            visibility: Visibility::Read,
        };
        let ranges = InputInfo::Scalar {
            elem: Elem::UInt,
            size: self.rank,
        };

        let info = CompilationInfo {
            inputs: vec![input, value, ranges],
            outputs: vec![],
            scope,
        };

        let settings = CompilationSettings::default();
        let shader = Compilation::new(info).compile(settings);
        let shader = <R::Compiler as Compiler>::compile(shader);
        SourceTemplate::new(shader.to_string())
    }

    fn id(&self) -> String {
        format!("{:?}-rank={:?}", core::any::TypeId::of::<Self>(), self.rank)
    }
}

type IntType<R> = <<R as Runtime>::Compiler as Compiler>::Int;

pub(crate) fn slice_assign<R: Runtime, E: JitElement, const D1: usize, const D2: usize>(
    tensor: JitTensor<R, E, D1>,
    indices: [Range<usize>; D2],
    value: JitTensor<R, E, D1>,
) -> JitTensor<R, E, D1> {
    let tensor = match tensor.can_mut() {
        true => tensor,
        false => tensor.copy(),
    };
    let mut scalars = Vec::with_capacity(D1);

    for i in 0..D1 {
        let start = indices.get(i).map(|index| index.start).unwrap_or(0);
        scalars.push((start as i32).elem());
    }

    let kernel = SliceAssignEagerKernel::new(D1);

    execute_dynamic::<R, SliceAssignEagerKernel<R, E>, IntType<R>>(
        &[
            EagerHandle::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
            EagerHandle::new(&value.handle, &value.strides, &value.shape.dims),
        ],
        &[],
        Some(&scalars),
        kernel,
        WorkgroupLaunch::Input { pos: 0 },
        value.client,
    );

    tensor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{ReferenceBackend, TestBackend, TestRuntime};
    use burn_tensor::{Distribution, Tensor};

    #[test]
    fn slice_assign_should_work_with_multiple_workgroups() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default, &Default::default());
        let value =
            Tensor::<TestBackend, 2>::random([2, 211], Distribution::Default, &Default::default());
        let indices = [3..5, 45..256];
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());
        let value_ref =
            Tensor::<ReferenceBackend, 2>::from_data(value.to_data(), &Default::default());

        let actual = slice_assign::<TestRuntime, _, 2, 2>(
            tensor.into_primitive(),
            indices.clone(),
            value.into_primitive(),
        );
        let expected = tensor_ref.slice_assign(indices, value_ref);

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 2>::from_primitive(actual).into_data(),
            3,
        );
    }
}
