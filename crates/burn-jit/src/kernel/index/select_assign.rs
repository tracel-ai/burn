use crate::{
    codegen::{
        dialect::gpu::{cube_inline, Branch, Elem, IntKind, Item, Scope, Variable, Visibility},
        Compilation, CompilationInfo, CompilationSettings, EagerHandle, Execution, InputInfo,
        WorkgroupLaunch,
    },
    element::JitElement,
    gpu::ComputeShader,
    kernel::{elemwise_workgroup, GpuComputeShaderPhase, WORKGROUP_DEFAULT},
    tensor::JitTensor,
    Runtime,
};
use std::marker::PhantomData;

pub struct SelectAssignComputeShader {
    tensor: Variable,
    indices: Variable,
    value: Variable,
    dim: usize,
}

#[derive(new)]
pub struct SelectAssignEagerKernel<R: Runtime, E: JitElement> {
    dim: usize,
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

impl SelectAssignComputeShader {
    pub fn expand(self, scope: &mut Scope) {
        let tensor = self.tensor;
        let value = self.value;
        let indices = self.indices;
        let id = Variable::Id;

        let offset_tensor = scope.zero(Elem::UInt);
        let offset_value = scope.zero(Elem::UInt);

        let stride_tensor_dim = scope.create_local(Elem::UInt);
        let stride_value_dim = scope.create_local(Elem::UInt);
        let shape_value_dim = scope.create_local(Elem::UInt);

        let num_elems = scope.create_local(Elem::UInt);
        cube_inline!(scope, num_elems = cast(1u32));

        cube_inline!(
            scope,
            range(0u32, Variable::Rank).for_each(|i, scope| {
                let shape_value = scope.create_local(Elem::UInt);
                let stride_tensor = scope.create_local(Elem::UInt);
                let stride_value = scope.create_local(Elem::UInt);

                cube_inline!(scope, stride_tensor = stride(tensor, i));
                cube_inline!(scope, stride_value = stride(value, i));
                cube_inline!(scope, shape_value = shape(value, i));

                let dim_index = scope.create_local(Elem::Bool);
                cube_inline!(scope, dim_index = i == self.dim);

                cube_inline!(scope, if(dim_index).then(|scope| {
                    cube_inline!(scope, shape_value_dim = shape_value);
                    cube_inline!(scope, stride_tensor_dim = stride_tensor);
                    cube_inline!(scope, stride_value_dim = stride_value);
                }).else(|scope| {
                    let stride_tmp = scope.create_local(Elem::UInt);
                    let shape_tensor = scope.create_local(Elem::UInt);

                    cube_inline!(scope, stride_tmp = stride(indices, i));
                    cube_inline!(scope, shape_tensor = shape(tensor, i));

                    cube_inline!(scope, num_elems = num_elems * shape_tensor);

                    let offset_local = scope.create_local(Elem::UInt);
                    let offset_local_tensor = scope.create_local(Elem::UInt);
                    let offset_local_value = scope.create_local(Elem::UInt);

                    cube_inline!(scope, offset_local = id / stride_tmp);

                    cube_inline!(scope, offset_local_tensor = offset_local % shape_tensor);
                    cube_inline!(
                        scope,
                        offset_local_tensor = offset_local_tensor * stride_tensor
                    );
                    cube_inline!(scope, offset_tensor += offset_local_tensor);

                    cube_inline!(scope, offset_local_value = offset_local % shape_value);
                    cube_inline!(
                        scope,
                        offset_local_value = offset_local_value * stride_value
                    );
                    cube_inline!(scope, offset_value += offset_local_value);
                }));
            })
        );

        let should_stop = scope.create_local(Elem::Bool);
        cube_inline!(scope, should_stop = id >= num_elems);
        cube_inline!(scope, if(should_stop).then(|scope| {
            scope.register(Branch::Return);
        }));

        cube_inline!(
            scope,
            range(0u32, shape_value_dim).for_each(|i, scope| {
                let index = scope.create_local(Elem::UInt);
                let index_tensor = scope.create_local(Elem::UInt);
                let index_value = scope.create_local(Elem::UInt);

                let result_tensor = scope.create_local(tensor.item());
                let result_value = scope.create_local(value.item());
                let result = scope.create_local(tensor.item());

                cube_inline!(scope, index = indices[i]);

                cube_inline!(scope, index_tensor = index * stride_tensor_dim);
                cube_inline!(scope, index_tensor += offset_tensor);

                cube_inline!(scope, index_value = i * stride_value_dim);
                cube_inline!(scope, index_value += offset_value);

                cube_inline!(scope, result_tensor = tensor[index_tensor]);
                cube_inline!(scope, result_value = value[index_value]);
                cube_inline!(scope, result = result_value + result_tensor);

                cube_inline!(scope, tensor[index_tensor] = result);
            })
        );
    }
}

impl<R: Runtime, E: JitElement> GpuComputeShaderPhase for SelectAssignEagerKernel<R, E> {
    fn compile(&self) -> ComputeShader {
        let mut scope = Scope::root();
        let item = E::gpu_elem().into();
        let item_indices: Item = Elem::Int(IntKind::I32).into();

        let tensor = Variable::GlobalInputArray(0, item);
        let value = Variable::GlobalInputArray(1, item);
        let indices = Variable::GlobalInputArray(2, item_indices);

        scope.write_global_custom(tensor);

        SelectAssignComputeShader {
            tensor,
            indices,
            value,
            dim: self.dim,
        }
        .expand(&mut scope);

        let tensor = InputInfo::Array {
            item,
            visibility: Visibility::ReadWrite,
        };
        let value = InputInfo::Array {
            item,
            visibility: Visibility::Read,
        };
        let indices = InputInfo::Array {
            item: item_indices,
            visibility: Visibility::Read,
        };

        let info = CompilationInfo {
            inputs: vec![tensor, value, indices],
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

pub(crate) fn select_assign<R: Runtime, E: JitElement, I: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
    dim: usize,
    indices: JitTensor<R, I, 1>,
    value: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    let tensor = match tensor.can_mut() {
        true => tensor,
        false => tensor.copy(),
    };

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

    let kernel = SelectAssignEagerKernel::<R, E>::new(dim);
    let workgroup = elemwise_workgroup(num_elems_per_workgroup, WORKGROUP_DEFAULT);

    Execution::start(kernel, indices.client)
        .inputs(&[
            EagerHandle::<R>::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
            EagerHandle::new(&value.handle, &value.strides, &value.shape.dims),
            // We use the custom strides here instead of the shape, since we don't use it in the
            // kernel, but we need to put the right number of dimensions (rank).
            EagerHandle::new(&indices.handle, &strides, &strides),
        ])
        .execute(WorkgroupLaunch::Custom(workgroup));

    tensor
}
