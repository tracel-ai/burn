use crate::{
    codegen::{
        dialect::gpu::{gpu, Elem, Scope, Variable, Visibility},
        execute_dynamic, Compilation, CompilationInfo, CompilationSettings, Compiler, EagerHandle,
        InputInfo, OutputInfo, WorkgroupLaunch,
    },
    element::JitElement,
    kernel::{DynamicKernelSource, SourceTemplate},
    ops::numeric::empty_device,
    tensor::JitTensor,
    Runtime,
};
use burn_tensor::{ElementConversion, Shape};
use std::{marker::PhantomData, ops::Range};

#[derive(new)]
struct SliceEagerKernel<R: Runtime, E: JitElement> {
    rank: usize,
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

pub struct SliceComputeShader {
    input: Variable,
    output: Variable,
    rank: usize,
}

impl SliceComputeShader {
    pub fn expand(self, scope: &mut Scope) {
        let input = self.input;
        let output = self.output;
        let id = Variable::Id;

        let offset_input = scope.zero(Elem::UInt);
        let offset_local = scope.create_local(Elem::UInt);

        let stride_input = scope.create_local(Elem::UInt);
        let stride_output = scope.create_local(Elem::UInt);
        let shape_output = scope.create_local(Elem::UInt);
        let range_start = scope.create_local(Elem::UInt);

        for i in 0..self.rank {
            gpu!(scope, stride_input = stride(input, i));
            gpu!(scope, stride_output = stride(output, i));
            gpu!(scope, shape_output = shape(output, i));
            gpu!(
                scope,
                range_start = cast(Variable::GlobalScalar(i as u16, Elem::UInt))
            );

            gpu!(scope, offset_local = id / stride_output);
            gpu!(scope, offset_local = offset_local % shape_output);
            gpu!(scope, offset_local = offset_local + range_start);
            gpu!(scope, offset_local = offset_local * stride_input);

            gpu!(scope, offset_input += offset_local);
        }

        let result = scope.create_local(input.item());
        gpu!(scope, result = input[offset_input]);
        gpu!(scope, output[id] = result);
    }
}

impl<R: Runtime, E: JitElement> DynamicKernelSource for SliceEagerKernel<R, E> {
    fn source(&self) -> crate::kernel::SourceTemplate {
        let mut scope = Scope::root();
        let item = E::gpu_elem().into();

        let input = Variable::GlobalInputArray(0, item);
        let output = Variable::GlobalOutputArray(0, item);

        scope.write_global_custom(output);

        SliceComputeShader {
            input,
            output,
            rank: self.rank,
        }
        .expand(&mut scope);

        let input = InputInfo::Array {
            item,
            visibility: Visibility::Read,
        };
        let ranges = InputInfo::Scalar {
            elem: Elem::UInt,
            size: self.rank,
        };
        let output = OutputInfo::Array { item };

        let info = CompilationInfo {
            inputs: vec![input, ranges],
            outputs: vec![output],
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

pub(crate) fn slice<R: Runtime, E: JitElement, const D1: usize, const D2: usize>(
    tensor: JitTensor<R, E, D1>,
    indices: [Range<usize>; D2],
) -> JitTensor<R, E, D1> {
    let mut dims = tensor.shape.dims;
    for i in 0..D2 {
        dims[i] = indices[i].end - indices[i].start;
    }
    let shape_output = Shape::new(dims);
    let output = empty_device(tensor.client.clone(), tensor.device.clone(), shape_output);
    slice_on_output(tensor, output, indices)
}

type IntType<R> = <<R as Runtime>::Compiler as Compiler>::Int;

pub(crate) fn slice_on_output<R: Runtime, E: JitElement, const D1: usize, const D2: usize>(
    tensor: JitTensor<R, E, D1>,
    output: JitTensor<R, E, D1>,
    indices: [Range<usize>; D2],
) -> JitTensor<R, E, D1> {
    let mut scalars = Vec::with_capacity(D1);

    for i in 0..D1 {
        let start = indices.get(i).map(|index| index.start).unwrap_or(0);
        scalars.push((start as i32).elem());
    }

    let kernel = SliceEagerKernel::new(D1);

    execute_dynamic::<R, SliceEagerKernel<R, E>, IntType<R>>(
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
        Some(&scalars),
        kernel,
        WorkgroupLaunch::Output { pos: 0 },
        tensor.client,
    );

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{Distribution, Tensor};

    #[test]
    fn slice_should_work_with_multiple_workgroups() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default, &Default::default());
        let indices = [3..5, 45..256];
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());

        let actual = slice(tensor.into_primitive(), indices.clone());
        let expected = tensor_ref.slice(indices);

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 2>::from_primitive(actual).into_data(),
            3,
        );
    }
}
