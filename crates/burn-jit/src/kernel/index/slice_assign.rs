use crate::{
    codegen::{
        dialect::gpu::{gpu, Elem, Scope, Variable, Visibility},
        Compilation, CompilationInfo, CompilationSettings, EagerHandle, Execution, InputInfo,
        WorkgroupLaunch,
    },
    element::JitElement,
    gpu::ComputeShader,
    kernel::GpuComputeShaderPhase,
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

impl<R: Runtime, E: JitElement> GpuComputeShaderPhase for SliceAssignEagerKernel<R, E> {
    fn compile(&self) -> ComputeShader {
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
        Compilation::new(info).compile(settings)
    }

    fn id(&self) -> String {
        format!("{:?}-rank={:?}", core::any::TypeId::of::<Self>(), self.rank)
    }
}

pub(crate) fn slice_assign<R: Runtime, E: JitElement, const D1: usize, const D2: usize>(
    tensor: JitTensor<R, E, D1>,
    indices: [Range<usize>; D2],
    value: JitTensor<R, E, D1>,
) -> JitTensor<R, E, D1> {
    let tensor = match tensor.can_mut() {
        true => tensor,
        false => tensor.copy(),
    };
    let mut scalars: Vec<i32> = Vec::with_capacity(D1);

    for i in 0..D1 {
        let start = indices.get(i).map(|index| index.start).unwrap_or(0);
        scalars.push((start as i32).elem());
    }

    let kernel = SliceAssignEagerKernel::<R, E>::new(D1);

    Execution::start(kernel, value.client)
        .inputs(&[
            EagerHandle::<R>::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
            EagerHandle::new(&value.handle, &value.strides, &value.shape.dims),
        ])
        .with_scalars(&scalars)
        .execute(WorkgroupLaunch::Input { pos: 0 });

    tensor
}
