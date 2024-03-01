use std::marker::PhantomData;

use serde::de::value;

use crate::{
    codegen::{
        dialect::gpu::{
            gpu, Branch, Elem, Item, Scope, Synchronization, Variable, Visibility, WorkgroupSize,
        },
        execute_dynamic, Compilation, CompilationInfo, CompilationSettings, Compiler, EagerHandle,
        InputInfo, OutputInfo, WorkgroupLaunch,
    },
    compute::WorkGroup,
    element::JitElement,
    kernel::{DynamicKernelSource, SourceTemplate, WORKGROUP_DEFAULT},
    tensor::JitTensor,
    Runtime,
};

pub(crate) trait SharedReduceDim: Send + Sync + 'static {
    type Accumulator: Copy;

    fn initialize(
        scope: &mut Scope,
        shared_memory_size: u32,
        write_position: Variable,
        input_item: Item,
    ) -> Self::Accumulator;

    fn write_to_shared(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        write_position: Variable,
        value: Self::Accumulator,
    );

    fn read_from_input(
        scope: &mut Scope,
        input: Variable,
        read_position: Variable,
    ) -> Self::Accumulator;

    fn read_from_shared(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        read_position: Variable,
    ) -> Self::Accumulator;

    fn assign(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        output: Variable,
        write_position: Variable,
        shape_reduce_dim: Variable,
    );
}

pub(crate) struct SumDim;

impl SharedReduceDim for SumDim {
    type Accumulator = Variable;

    fn initialize(
        scope: &mut Scope,
        shared_memory_size: u32,
        write_position: Variable,
        input_item: Item,
    ) -> Self::Accumulator {
        let shared_memory = scope.create_shared(input_item, shared_memory_size);
        let neutral_element = scope.zero(shared_memory.item());
        gpu!(scope, shared_memory[write_position] = neutral_element);
        shared_memory
    }

    fn write_to_shared(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        write_position: Variable,
        value: Self::Accumulator,
    ) {
        let current_value = scope.create_local(value.item());
        let computed = scope.create_local(value.item());
        gpu!(scope, current_value = shared_memory[write_position]);
        gpu!(scope, computed = current_value + value);
        gpu!(scope, shared_memory[write_position] = computed);
    }

    fn read_from_input(
        scope: &mut Scope,
        input: Variable,
        read_position: Variable,
    ) -> Self::Accumulator {
        let value = scope.create_local(input.item());
        gpu!(scope, value = input[read_position]);
        value
    }

    fn read_from_shared(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        read_position: Variable,
    ) -> Self::Accumulator {
        let read_value = scope.create_local(shared_memory.item());
        gpu!(scope, read_value = shared_memory[read_position]);
        read_value
    }

    fn assign(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        output: Variable,
        write_position: Variable,
        _shape_reduce_dim: Variable,
    ) {
        let final_value = scope.create_local(output.item());
        gpu!(scope, final_value = shared_memory[0]);
        gpu!(scope, output[write_position] = final_value);
    }
}

pub(crate) struct MeanDim;

impl SharedReduceDim for MeanDim {
    type Accumulator = Variable;

    fn initialize(
        scope: &mut Scope,
        shared_memory_size: u32,
        write_position: Variable,
        input_item: Item,
    ) -> Self::Accumulator {
        let shared_memory = scope.create_shared(input_item, shared_memory_size);
        let neutral_element = scope.zero(shared_memory.item());
        gpu!(scope, shared_memory[write_position] = neutral_element);
        shared_memory
    }

    fn write_to_shared(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        write_position: Variable,
        value: Self::Accumulator,
    ) {
        let current_value = scope.create_local(value.item());
        let computed = scope.create_local(value.item());
        gpu!(scope, current_value = shared_memory[write_position]);
        gpu!(scope, computed = current_value + value);
        gpu!(scope, shared_memory[write_position] = computed);
    }

    fn read_from_input(
        scope: &mut Scope,
        input: Variable,
        read_position: Variable,
    ) -> Self::Accumulator {
        let value = scope.create_local(input.item());
        gpu!(scope, value = input[read_position]);
        value
    }

    fn read_from_shared(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        read_position: Variable,
    ) -> Variable {
        let read_value = scope.create_local(shared_memory.item());
        gpu!(scope, read_value = shared_memory[read_position]);
        read_value
    }

    fn assign(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        output: Variable,
        write_position: Variable,
        shape_reduce_dim: Variable,
    ) {
        let final_value = scope.create_local(output.item());
        gpu!(scope, final_value = shared_memory[0]);

        let denominator = scope.create_local(output.item());
        gpu!(scope, denominator = cast(shape_reduce_dim));
        gpu!(scope, final_value = final_value / denominator);
        gpu!(scope, output[write_position] = final_value);
    }
}

pub(crate) struct ArgMin;

impl SharedReduceDim for ArgMin {
    type Accumulator = (Variable, Variable);

    fn initialize(
        scope: &mut Scope,
        shared_memory_size: u32,
        write_position: Variable,
        input_item: Item,
    ) -> Self::Accumulator {
        let value_shared_memory = scope.create_shared(input_item, shared_memory_size);
        let index_shared_memory = scope.create_shared(Elem::UInt, shared_memory_size);
        let min = scope.create_local(input_item);
        gpu!(scope, min = cast(32767.0));
        gpu!(scope, value_shared_memory[write_position] = min);
        (value_shared_memory, index_shared_memory)
    }

    fn write_to_shared(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        write_position: Variable,
        (value, index): Self::Accumulator,
    ) {
        let (value_shared_memory, index_shared_memory) = shared_memory;
        let current_value = scope.create_local(value.item());
        gpu!(scope, current_value = value_shared_memory[write_position]);

        let condition = scope.create_local(Elem::Bool);
        gpu!(scope, condition = value < current_value);
        gpu!(scope, if(condition).then(|scope| {
            gpu!(scope, value_shared_memory[write_position] = value);
            gpu!(scope, index_shared_memory[write_position] = index);
        }));
    }

    fn read_from_input(
        scope: &mut Scope,
        input: Variable,
        read_position: Variable,
    ) -> Self::Accumulator {
        let value = scope.create_local(input.item());
        gpu!(scope, value = input[read_position]);
        todo!()
        // let index = // TODO modulo other shapes??
        // (value, index)
        
    }

    fn read_from_shared(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        read_position: Variable,
    ) -> Self::Accumulator {
        let (value_shared_memory, index_shared_memory) = shared_memory;
        let value = scope.create_local(value_shared_memory.item());
        gpu!(scope, value = value_shared_memory[read_position]);
        let index = scope.create_local(index_shared_memory.item());
        gpu!(scope, index = index_shared_memory[read_position]);
        (value, index)
    }

    fn assign(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        output: Variable,
        write_position: Variable,
        _shape_reduce_dim: Variable,
    ) {
        let (_, index_shared_memory) = shared_memory;
        let final_value = scope.create_local(output.item());
        gpu!(scope, final_value = index_shared_memory[0]);
        gpu!(scope, output[write_position] = final_value);
    }
}

pub(crate) struct SharedReduceDimComputeShader<RD: SharedReduceDim> {
    tensor: Variable,
    dim: usize,
    shared_memory_size: usize,
    n_input_values_per_thread: u32,
    output: Variable,
    _reduce_dim: PhantomData<RD>,
}

#[derive(new)]
pub(crate) struct SharedReduceDimEagerKernel<
    RD: SharedReduceDim,
    R: Runtime,
    EI: JitElement,
    EO: JitElement,
> {
    dim: usize,
    workgroup_size_x: usize,
    workgroup_size_y: usize,
    n_input_values_per_thread: u32,
    _reduce_dim: PhantomData<RD>,
    _runtime: PhantomData<R>,
    _elem_in: PhantomData<EI>,
    _elem_out: PhantomData<EO>,
}

impl<RD: SharedReduceDim, R: Runtime, EI: JitElement, EO: JitElement> DynamicKernelSource
    for SharedReduceDimEagerKernel<RD, R, EI, EO>
{
    fn source(&self) -> crate::kernel::SourceTemplate {
        let mut scope = Scope::root();
        let item_input = EI::gpu_elem().into();
        let item_output = EO::gpu_elem().into();

        let tensor = Variable::GlobalInputArray(0, item_input);
        let output = Variable::GlobalOutputArray(0, item_output);

        // Reduce groups are elements that are aligned along the reduce dim
        SharedReduceDimComputeShader {
            tensor,
            dim: self.dim,
            shared_memory_size: self.workgroup_size_x * self.workgroup_size_y,
            n_input_values_per_thread: self.n_input_values_per_thread,
            output,
            _reduce_dim: PhantomData::<RD>,
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
        format!(
            "{:?}dim={}x={}y={}n={}",
            core::any::TypeId::of::<Self>(),
            self.dim,
            self.workgroup_size_x,
            self.workgroup_size_y,
            self.n_input_values_per_thread
        )
    }
}

impl<RD: SharedReduceDim> SharedReduceDimComputeShader<RD> {
    pub(crate) fn expand(self, scope: &mut Scope) {
        let tensor = self.tensor;
        let output = self.output;

        let rank = Variable::Rank;
        let dim: Variable = self.dim.into();

        // threads are responsible of how many inputs in one reduce_group
        let n_input_values_per_thread: Variable = self.n_input_values_per_thread.into();

        let workgroup_id_x = Variable::WorkgroupIdX;
        let workgroup_id_y = Variable::WorkgroupIdY;
        let num_workgroups_x = Variable::NumWorkgroupsX;
        let local_invocation_id_x = Variable::LocalInvocationIdX;
        let local_invocation_id_y = Variable::LocalInvocationIdY;
        let workgroup_size_x = Variable::WorkgroupSizeX;
        let workgroup_size_y = Variable::WorkgroupSizeY;

        let stride_reduce_dim_input = scope.create_local(Elem::UInt);
        gpu!(scope, stride_reduce_dim_input = stride(tensor, dim));
        let shape_reduce_dim_input = scope.create_local(Elem::UInt);
        gpu!(scope, shape_reduce_dim_input = shape(tensor, dim));

        // To determine which reduce_group (not position, but absolute id)
        let reduce_group_id = scope.create_local(Elem::UInt);
        gpu!(scope, reduce_group_id = workgroup_id_y * num_workgroups_x);
        gpu!(scope, reduce_group_id += workgroup_id_x);

        // nth thread in the workgroup
        let local_id = scope.create_local(Elem::UInt);
        gpu!(scope, local_id = local_invocation_id_y * workgroup_size_x);
        gpu!(scope, local_id += local_invocation_id_x);

        let n_threads = scope.create_local(Elem::UInt);
        gpu!(scope, n_threads = workgroup_size_x * workgroup_size_y);

        // let offset_tensor = scope.create_local(Elem::UInt);
        let index_offset = scope.zero(Elem::UInt);

        gpu!(
            scope,
            range(0u32, rank).for_each(|i, scope| {
                let stride_input = scope.create_local(Elem::UInt);
                let stride_output = scope.create_local(Elem::UInt);
                let shape_output = scope.create_local(Elem::UInt);

                gpu!(scope, stride_input = stride(tensor, i));
                gpu!(scope, stride_output = stride(output, i));
                gpu!(scope, shape_output = shape(output, i));

                let num_block = scope.create_local(Elem::UInt);
                gpu!(scope, num_block = reduce_group_id / stride_output);
                gpu!(scope, num_block = num_block % shape_output);
                gpu!(scope, num_block = num_block * stride_input);
                gpu!(scope, index_offset += num_block);
            })
        );

        let shared_memory = RD::initialize(
            scope,
            self.shared_memory_size as u32,
            local_id,
            tensor.item(),
        );

        // Load to shared memory
        gpu!(
            // TODO UNROLLABLE, then maybe don't even need n_input__vpt in kernel
            scope,
            range(0u32, n_input_values_per_thread).for_each(|i, scope| {
                let nth = scope.create_local(Elem::UInt);
                gpu!(scope, nth = i * n_threads);
                gpu!(scope, nth += local_id);

                let within_shape = scope.create_local(Elem::Bool);
                gpu!(scope, within_shape = nth < shape_reduce_dim_input);
                gpu!(scope, if(within_shape).then(|scope|{
                    let current_position = scope.create_local(Elem::UInt);
                    gpu!(scope, current_position = nth * stride_reduce_dim_input);
                    gpu!(scope, current_position += index_offset);


                    let new_value = RD::read_from_input(scope, tensor, current_position);
                    RD::write_to_shared(scope, shared_memory, local_id, new_value);
                }));
            })
        );

        scope.register(Synchronization::WorkgroupBarrier);

        let several_threads_active = scope.create_local(Elem::Bool);

        // TODO remove an if when n_input_values_per_thread arrive just

        gpu!(scope, loop(|scope|{
            gpu!(scope, several_threads_active = n_threads <= 1u32);
            gpu!(scope, if(several_threads_active).then(|scope|{
                scope.register(Branch::Break);
            }));

            gpu!(scope, n_threads = n_threads / 2u32);

            let updating_thread = scope.create_local(Elem::Bool);
            gpu!(scope, updating_thread = local_id < n_threads);
            gpu!(scope, if(updating_thread).then(|scope|{
                let read_position = scope.create_local(Elem::UInt);
                gpu!(scope, read_position = n_threads + local_id);

                let read_value = RD::read_from_shared(scope, shared_memory, read_position);
                RD::write_to_shared(scope, shared_memory, local_id, read_value);
            }));

            scope.register(Synchronization::WorkgroupBarrier);
        }));

        let is_first_thread = scope.create_local(Elem::Bool);
        gpu!(scope, is_first_thread = local_id == 0u32);
        gpu!(scope, if(is_first_thread).then(|scope|{
            RD::assign(scope, shared_memory, output, reduce_group_id, shape_reduce_dim_input);
        }));
    }
}

pub(crate) fn reduce_dim_shared<
    RD: SharedReduceDim,
    R: Runtime,
    EI: JitElement,
    EO: JitElement,
    const D: usize,
>(
    input: JitTensor<R, EI, D>,
    output: JitTensor<R, EO, D>,
    dim: usize,
) -> JitTensor<R, EO, D> {
    let num_elems_output = output.shape.num_elements();
    let n_workgroups_x = f32::ceil(f32::sqrt(num_elems_output as f32));
    let n_workgroups_y = f32::ceil(num_elems_output as f32 / n_workgroups_x);
    let grid = WorkGroup::new(n_workgroups_x as u32, n_workgroups_y as u32, 1);

    let reduce_group_size = input.shape.dims[dim];
    let n_invocation_per_workgroup = WORKGROUP_DEFAULT * WORKGROUP_DEFAULT;
    let n_input_values_per_thread =
        f32::ceil(reduce_group_size as f32 / n_invocation_per_workgroup as f32) as u32;

    let kernel = SharedReduceDimEagerKernel::new(
        dim,
        WORKGROUP_DEFAULT,
        WORKGROUP_DEFAULT,
        n_input_values_per_thread,
    );

    execute_dynamic::<R, SharedReduceDimEagerKernel<RD, R, EI, EO>, EI>(
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
        WorkgroupLaunch::Custom(grid),
        input.client,
    );

    output
}

/// Execute the sum dim kernel using shared strategy.
pub fn sum_dim_shared<R: Runtime, E: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
    output: JitTensor<R, E, D>,
    dim: usize,
) -> JitTensor<R, E, D> {
    reduce_dim_shared::<SumDim, R, E, E, D>(input, output, dim)
}

/// Execute the mean dim kernel using shared strategy.
pub fn mean_dim_shared<R: Runtime, E: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
    output: JitTensor<R, E, D>,
    dim: usize,
) -> JitTensor<R, E, D> {
    reduce_dim_shared::<MeanDim, R, E, E, D>(input, output, dim)
}

/// Execute the argmin dim kernel using shared strategy.
pub fn argmin_shared<R: Runtime, EI: JitElement, EO: JitElement, const D: usize>(
    input: JitTensor<R, EI, D>,
    output: JitTensor<R, EO, D>,
    dim: usize,
) -> JitTensor<R, EO, D> {
    reduce_dim_shared::<ArgMin, R, EI, EO, D>(input, output, dim)
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
    fn reduction_sum_dim_shared_memory_small() {
        let tensor =
            Tensor::<TestBackend, 1>::random([700], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 1>::from_data(tensor.to_data(), &Default::default());
        let reduce_dim = 0;
        let output = init_reduce_output(&tensor.clone().into_primitive(), reduce_dim);

        let val = Tensor::<TestBackend, 1>::from_primitive(sum_dim_shared(
            tensor.into_primitive(),
            output,
            reduce_dim,
        ));
        let val_ref = tensor_ref.sum_dim(reduce_dim);

        val_ref.into_data().assert_approx_eq(&val.into_data(), 3);
    }

    #[test]
    fn reduction_sum_dim_shared_memory_medium() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 1024], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());
        let reduce_dim = 1;
        let output = init_reduce_output(&tensor.clone().into_primitive(), reduce_dim);

        let val = Tensor::<TestBackend, 2>::from_primitive(sum_dim_shared(
            tensor.into_primitive(),
            output,
            reduce_dim,
        ));
        let val_ref = tensor_ref.sum_dim(reduce_dim);

        val_ref.into_data().assert_approx_eq(&val.into_data(), 3);
    }

    #[test]
    fn reduction_sum_dim_shared_memory_large() {
        let tensor = Tensor::<TestBackend, 3>::random(
            [4, 1024, 50],
            Distribution::Default,
            &Default::default(),
        );
        let tensor_ref =
            Tensor::<ReferenceBackend, 3>::from_data(tensor.to_data(), &Default::default());
        let reduce_dim = 2;
        let output = init_reduce_output(&tensor.clone().into_primitive(), reduce_dim);

        let val = Tensor::<TestBackend, 3>::from_primitive(sum_dim_shared(
            tensor.into_primitive(),
            output,
            reduce_dim,
        ));
        let val_ref = tensor_ref.sum_dim(reduce_dim);

        val_ref.into_data().assert_approx_eq(&val.into_data(), 3);
    }

    #[test]
    fn reduction_mean_dim_shared_memory_medium() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 1024], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());
        let reduce_dim = 0;
        let output = init_reduce_output(&tensor.clone().into_primitive(), reduce_dim);

        let val = Tensor::<TestBackend, 2>::from_primitive(mean_dim_shared(
            tensor.into_primitive(),
            output,
            reduce_dim,
        ));
        let val_ref = tensor_ref.mean_dim(reduce_dim);

        val_ref.into_data().assert_approx_eq(&val.into_data(), 3);
    }

    #[test]
    fn reduction_argmin_shared_memory_medium() {
        let tensor =
            Tensor::<TestBackend, 2>::random([2, 2], Distribution::Default, &Default::default());
        println!("{:?}", tensor.to_data());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());
        let reduce_dim = 0;
        let output = init_reduce_output(&tensor.clone().into_primitive(), reduce_dim);

        let val = Tensor::<TestBackend, 2>::from_primitive(argmin_shared(
            tensor.into_primitive(),
            output,
            reduce_dim,
        ));
        let val_ref = tensor_ref.argmin(reduce_dim);

        assert_eq!(val_ref.into_data().convert(), val.into_data());
    }
}
