use std::marker::PhantomData;

use crate::{
    codegen::{
        dialect::gpu::{gpu, Elem, Scope, Synchronization, Variable, Visibility, WorkgroupSize},
        execute_dynamic, Compilation, CompilationInfo, CompilationSettings, Compiler, EagerHandle,
        InputInfo, OutputInfo, WorkgroupLaunch,
    },
    element::JitElement,
    kernel::{DynamicKernelSource, SourceTemplate, WORKGROUP_DEFAULT},
    tensor::JitTensor,
    Runtime,
};

pub(crate) struct WorkgroupReduceDimComputeShader {
    tensor: Variable,
    dim: usize,
    shared_memory_size: usize,
    output: Variable,
}

#[derive(new)]
pub(crate) struct WorkgroupReduceDimEagerKernel<R: Runtime, EI: JitElement, EO: JitElement> {
    dim: usize,
    workgroup_size_x: usize,
    workgroup_size_y: usize,
    _runtime: PhantomData<R>,
    _elem_in: PhantomData<EI>,
    _elem_out: PhantomData<EO>,
}

impl<R: Runtime, EI: JitElement, EO: JitElement> DynamicKernelSource
    for WorkgroupReduceDimEagerKernel<R, EI, EO>
{
    fn source(&self) -> crate::kernel::SourceTemplate {
        let mut scope = Scope::root();
        let item_input = EI::gpu_elem().into();
        let item_output = EO::gpu_elem().into();

        let tensor = Variable::GlobalInputArray(0, item_input);
        let output = Variable::GlobalOutputArray(0, item_output);

        WorkgroupReduceDimComputeShader {
            tensor,
            dim: self.dim,
            shared_memory_size: self.workgroup_size_x * self.workgroup_size_y,
            output,
        }
        .expand(&mut scope);

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

        let settings = CompilationSettings::default().workgroup_size(WorkgroupSize::new(
            self.workgroup_size_x as u32,
            self.workgroup_size_y as u32,
            1,
        ));
        let shader = Compilation::new(info).compile(settings);
        let shader = <R::Compiler as Compiler>::compile(shader);
        SourceTemplate::new(shader.to_string())
    }

    fn id(&self) -> String {
        format!("{:?}dim={}", core::any::TypeId::of::<Self>(), self.dim)
    }
}

impl WorkgroupReduceDimComputeShader {
    pub(crate) fn expand(self, scope: &mut Scope) {
        // let tensor = self.tensor;
        // let dim: Variable = self.dim.into();
        // let id = Variable::Id;
        // let output = self.output;

        let tensor = self.tensor;
        let dim: Variable = self.dim.into();
        // let id = Variable::Id;
        let output = self.output;

        let offset_input = scope.zero(Elem::UInt);
        let stride_input_dim = scope.create_local(Elem::UInt);
        let shape_input_dim = scope.create_local(Elem::UInt);

        let workgroup_size_x = Variable::WorkgroupSizeX;
        let workgroup_size_y = Variable::WorkgroupSizeY;

        let nw = Variable::NumWorkgroupsZ;
        let wg_id = Variable::WorkgroupIdX;
        let lii = Variable::LocalInvocationIdX;
        let liiy = Variable::LocalInvocationIdY;

        // let shared_memory_size =
        let n_input_values_per_thread = scope.create_local(Elem::UInt);
        // u32(ceil(f32(shape_dim)/f32(workgroup_size_x*workgroup_size_y)))

        let sm = scope.create_shared(tensor.item(), self.shared_memory_size as u32);
        let sm2 = scope.create_shared(tensor.item(), 800);

        gpu!(
            scope,
            range(0u32, Variable::Rank).for_each(|i, scope| {
                let stride_input = scope.create_local(Elem::UInt);
                let stride_output = scope.create_local(Elem::UInt);
                let shape_output = scope.create_local(Elem::UInt);

                gpu!(scope, stride_input = stride(tensor, i));
                gpu!(scope, stride_output = stride(output, i));
                gpu!(scope, shape_output = shape(output, i));

                let offset_local = scope.create_local(Elem::UInt);
                gpu!(scope, offset_local = nw / shape_output);
                gpu!(
                    scope,
                    offset_local = offset_local % n_input_values_per_thread
                );

                let is_dim_reduce = scope.create_local(Elem::Bool);
                gpu!(scope, is_dim_reduce = i == dim);

                scope.register(Synchronization::WorkgroupBarrier);
                gpu!(scope, while(is_dim_reduce).then(|scope|{
                    gpu!(scope, shape_input_dim = shape(tensor, i));
                    gpu!(scope, stride_input_dim = wg_id);
                    gpu!(scope, offset_input += lii);
                }));

                gpu!(scope, if(is_dim_reduce).then(|scope|{
                    gpu!(scope, shape_input_dim = shape(tensor, i));
                    gpu!(scope, stride_input_dim = wg_id);
                    gpu!(scope, offset_input += lii);
                }).else(|scope|{
                    gpu!(scope, offset_local = offset_local * liiy);
                    gpu!(scope, offset_input += offset_local);
                }));

                let s = scope.create_local(tensor.item());
                gpu!(scope, s = sm[0]);
                gpu!(scope, s = sm2[0]);
            })
        );
    }
}

pub(crate) fn reduce_dim_workgroup<R: Runtime, EI: JitElement, EO: JitElement, const D: usize>(
    input: JitTensor<R, EI, D>,
    output: JitTensor<R, EO, D>,
    dim: usize,
) -> JitTensor<R, EO, D> {
    let kernel = WorkgroupReduceDimEagerKernel::new(dim, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT);

    execute_dynamic::<R, WorkgroupReduceDimEagerKernel<R, EI, EO>, EI>(
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
    use crate::{
        kernel::reduce::init_reduce_output,
        tests::{ReferenceBackend, TestBackend},
    };
    use burn_tensor::{Distribution, Tensor};

    #[test]
    fn workgroup_reduce() {
        let tensor =
            Tensor::<TestBackend, 1>::random([700], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 1>::from_data(tensor.to_data(), &Default::default());
        let reduce_dim = 0;
        let output = init_reduce_output(&tensor.clone().into_primitive(), reduce_dim);

        let val = Tensor::<TestBackend, 1>::from_primitive(reduce_dim_workgroup(
            tensor.into_primitive(),
            output,
            reduce_dim,
        ));
        let val_ref = tensor_ref.sum_dim(reduce_dim);

        val_ref.into_data().assert_approx_eq(&val.into_data(), 3);
    }
}
