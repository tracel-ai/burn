use crate::codegen::dialect::gpu::{gpu, Branch, Elem, Scope, Variable};
use crate::codegen::Execution;
use crate::gpu::ComputeShader;
use crate::kernel::{elemwise_workgroup, WORKGROUP_DEFAULT};
use crate::{
    codegen::{
        dialect::gpu, Compilation, CompilationInfo, CompilationSettings, EagerHandle, InputInfo,
        WorkgroupLaunch,
    },
    element::JitElement,
    kernel::{self, GpuComputeShaderPhase},
    tensor::JitTensor,
    Runtime,
};
use std::marker::PhantomData;

#[derive(new)]
struct ScatterEagerKernel<R: Runtime, E: JitElement> {
    dim: usize,
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

struct ScatterComputeShader {
    input: Variable,
    indices: Variable,
    value: Variable,
    dim: usize,
}

impl ScatterComputeShader {
    pub fn expand(self, scope: &mut Scope) {
        match self.input {
            Variable::GlobalInputArray(_, _) => (),
            Variable::GlobalOutputArray(_, _) => (),
            _ => panic!("Input variable must be an global array."),
        };
        match self.value {
            Variable::GlobalInputArray(_, _) => (),
            Variable::GlobalOutputArray(_, _) => (),
            _ => panic!("Value variable must be an global array."),
        };

        let input = self.input;
        let value = self.value;
        let indices = self.indices;

        let stride_input = scope.create_local(Elem::UInt);
        let shape_value = scope.create_local(Elem::UInt);

        gpu!(scope, stride_input = stride(input, self.dim));
        gpu!(scope, shape_value = shape(value, self.dim));

        let id = Variable::Id;
        let offset_input = scope.zero(Elem::UInt);
        let offset_value = scope.zero(Elem::UInt);

        let num_elems = scope.create_local(Elem::UInt);
        gpu!(scope, num_elems = cast(1usize));
        gpu!(
            scope,
            range(0u32, Variable::Rank).for_each(|i, scope| {
                let should_skip = scope.create_local(Elem::Bool);
                gpu!(scope, should_skip = i == self.dim);

                gpu!(scope, if(should_skip).then(|_| {
                    // Nothing to do.
                }).else(|scope| {
                    let shape_input_loop = scope.create_local(Elem::UInt);
                    let shape_value_loop = scope.create_local(Elem::UInt);
                    let stride_value_loop = scope.create_local(Elem::UInt);

                    let stride_tmp = scope.create_local(Elem::UInt);
                    let num_blocks = scope.create_local(Elem::UInt);
                    let offset_tmp = scope.create_local(Elem::UInt);
                    let stride_input_loop = scope.create_local(Elem::UInt);

                    gpu!(scope, stride_value_loop = stride(value, i));
                    gpu!(scope, stride_input_loop = stride(input, i));
                    gpu!(scope, stride_tmp = stride(indices, i));

                    gpu!(scope, shape_value_loop = shape(value, i));
                    gpu!(scope, shape_input_loop = shape(input, i));

                    gpu!(scope, num_blocks = id / stride_tmp);
                    gpu!(scope, num_blocks = num_blocks % shape_input_loop);

                    gpu!(scope, offset_tmp = num_blocks * stride_input_loop);
                    gpu!(scope, offset_input += offset_tmp);

                    gpu!(scope, offset_tmp = num_blocks * stride_value_loop);
                    gpu!(scope, offset_value += offset_tmp);

                    gpu!(scope, num_elems = num_elems * shape_value_loop);
                }));
            })
        );

        let should_stop = scope.create_local(Elem::Bool);
        gpu!(scope, should_stop = id >= num_elems);
        gpu!(scope, if (should_stop).then(|scope|{
            scope.register(Branch::Return);
        }));

        let index_input = scope.create_local(Elem::UInt);
        let index = scope.create_local(Elem::UInt);

        let result_input = scope.create_local(input.item());
        let result_value = scope.create_local(value.item());
        let result_indices = scope.create_local(Elem::UInt);

        gpu!(
            scope,
            range(0u32, shape_value).for_each(|i, scope| {
                gpu!(scope, index = stride_input * i);
                gpu!(scope, index += offset_value);

                gpu!(scope, result_value = value[index]);
                gpu!(scope, result_indices = indices[index]);

                gpu!(scope, index_input = stride_input * result_indices);
                gpu!(scope, index_input += offset_input);

                gpu!(scope, result_input = input[index_input]);
                gpu!(scope, result_input += result_value);
                gpu!(scope, input[index_input] = result_input);
            })
        );
    }
}

impl<R: Runtime, E: JitElement> GpuComputeShaderPhase for ScatterEagerKernel<R, E> {
    fn compile(&self) -> ComputeShader {
        let mut scope = gpu::Scope::root();
        let item_value = E::gpu_elem().into();
        let item_indices: gpu::Item = gpu::Elem::Int(gpu::IntKind::I32).into();

        let input_output = gpu::Variable::GlobalInputArray(0, item_value);
        let indices = gpu::Variable::GlobalInputArray(1, Elem::Int(gpu::IntKind::I32).into());
        let value = gpu::Variable::GlobalInputArray(2, item_value);

        scope.write_global_custom(input_output);

        ScatterComputeShader {
            input: input_output,
            indices,
            value,
            dim: self.dim,
        }
        .expand(&mut scope);

        let input_output = InputInfo::Array {
            item: item_value,
            visibility: gpu::Visibility::ReadWrite,
        };
        let indices = InputInfo::Array {
            item: item_indices,
            visibility: gpu::Visibility::Read,
        };
        let value = InputInfo::Array {
            item: item_value,
            visibility: gpu::Visibility::Read,
        };

        let info = CompilationInfo {
            inputs: vec![input_output, indices, value],
            outputs: vec![],
            scope,
        };

        let settings = CompilationSettings::default();
        Compilation::new(info).compile(settings)
    }

    fn id(&self) -> String {
        format!("{:?}dim={}", core::any::TypeId::of::<Self>(), self.dim)
    }
}

pub(crate) fn scatter<R: Runtime, E: JitElement, I: JitElement, const D: usize>(
    dim: usize,
    tensor: JitTensor<R, E, D>,
    indices: JitTensor<R, I, D>,
    value: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    let mut indices = kernel::into_contiguous(indices);
    let tensor = kernel::into_contiguous(tensor);
    let value = kernel::into_contiguous(value);

    let tensor = match tensor.can_mut() {
        true => tensor,
        false => tensor.copy(),
    };

    let kernel = ScatterEagerKernel::<R, E>::new(dim);
    let mut strides = [0; D];
    let mut current = 1;
    let mut num_elems_per_workgroup = 1;

    tensor
        .shape
        .dims
        .iter()
        .enumerate()
        .rev()
        .filter(|(index, _val)| *index != dim)
        .for_each(|(index, val)| {
            strides[index] = current;
            current *= val;
            num_elems_per_workgroup *= tensor.shape.dims[index];
        });

    // Fake strides of the virtual output where the strides of dim is hardcoded to one.
    indices.strides = strides;

    let workgroup = elemwise_workgroup(num_elems_per_workgroup, WORKGROUP_DEFAULT);

    Execution::start(kernel, indices.client)
        .inputs(&[
            EagerHandle::<R>::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
            EagerHandle::new(&indices.handle, &indices.strides, &indices.shape.dims),
            EagerHandle::new(&value.handle, &value.strides, &value.shape.dims),
        ])
        .execute(WorkgroupLaunch::Custom(workgroup));

    tensor
}
