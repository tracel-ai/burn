use cubecl::{
    cpa, ir::KernelDefinition, prelude::CubeCount, CubeCountSettings, Execution, InputInfo,
    KernelExpansion, KernelIntegrator, KernelSettings, OutputInfo,
};
use std::marker::PhantomData;

use crate::{
    element::JitElement,
    kernel::{Kernel, SUBCUBE_DIM_APPROX},
    tensor::JitTensor,
    JitRuntime,
};
use cubecl::ir::{Branch, CubeDim, Elem, Scope, Synchronization, Variable, Visibility};

use super::base::ReduceDimShared;

pub(crate) struct SharedReduceDimComputeShader<E: JitElement, RD: ReduceDimShared<E>> {
    tensor: Variable,
    dim: usize,
    shared_memory_size: usize,
    n_input_values_per_thread: u32,
    output: Variable,
    divisible_shape: bool,
    _reduce_dim: PhantomData<RD>,
    _elem: PhantomData<E>,
}

#[derive(new)]
pub(crate) struct SharedReduceDimEagerKernel<
    RD: ReduceDimShared<EI>,
    R: JitRuntime,
    EI: JitElement,
    EO: JitElement,
> {
    dim: usize,
    cube_dim_x: usize,
    cube_dim_y: usize,
    n_input_values_per_thread: u32,
    divisible_shape: bool,
    _reduce_dim: PhantomData<RD>,
    _runtime: PhantomData<R>,
    _elem_in: PhantomData<EI>,
    _elem_out: PhantomData<EO>,
}

impl<RD: ReduceDimShared<EI>, R: JitRuntime, EI: JitElement, EO: JitElement> Kernel
    for SharedReduceDimEagerKernel<RD, R, EI, EO>
{
    fn define(&self) -> KernelDefinition {
        let mut scope = Scope::root();
        let item_input = EI::cube_elem().into();
        let item_output = EO::cube_elem().into();

        let tensor = Variable::GlobalInputArray {
            id: 0,
            item: item_input,
        };
        let output = Variable::GlobalOutputArray {
            id: 0,
            item: item_output,
        };

        // Reduce groups are elements that are aligned along the reduce dim
        SharedReduceDimComputeShader {
            tensor,
            dim: self.dim,
            shared_memory_size: self.cube_dim_x * self.cube_dim_y,
            n_input_values_per_thread: self.n_input_values_per_thread,
            output,
            divisible_shape: self.divisible_shape,
            _reduce_dim: PhantomData::<RD>,
            _elem: PhantomData::<EI>,
        }
        .expand(&mut scope);

        scope.write_global_custom(output);

        let tensor = InputInfo::Array {
            item: item_input,
            visibility: Visibility::Read,
        };

        let out = OutputInfo::Array { item: item_output };

        let info = KernelExpansion {
            inputs: vec![tensor],
            outputs: vec![out],
            scope,
        };

        let settings = KernelSettings::default().cube_dim(CubeDim::new(
            self.cube_dim_x as u32,
            self.cube_dim_y as u32,
            1,
        ));
        KernelIntegrator::new(info).integrate(settings)
    }

    fn id(&self) -> cubecl::KernelId {
        cubecl::KernelId::new::<Self>().info((
            self.dim,
            self.cube_dim_x,
            self.cube_dim_y,
            self.n_input_values_per_thread,
            self.divisible_shape,
        ))
    }
}

impl<E: JitElement, RD: ReduceDimShared<E>> SharedReduceDimComputeShader<E, RD> {
    pub(crate) fn expand(self, scope: &mut Scope) {
        let tensor = self.tensor;
        let output = self.output;

        let rank = Variable::Rank;
        let dim: Variable = self.dim.into();

        let cube_pos_x = Variable::CubePosX;
        let cube_pos_y = Variable::CubePosY;
        let cube_count_x = Variable::CubeCountX;
        let local_invocation_id_x = Variable::UnitPosX;
        let local_invocation_id_y = Variable::UnitPosY;
        let cube_dim_x = Variable::CubeDimX;
        let cube_dim_y = Variable::CubeDimY;

        let stride_reduce_dim_input = scope.create_local(Elem::UInt);
        cpa!(scope, stride_reduce_dim_input = stride(tensor, dim));
        let shape_reduce_dim_input = scope.create_local(Elem::UInt);
        cpa!(scope, shape_reduce_dim_input = shape(tensor, dim));

        // To determine which reduce_group (not position, but absolute id)
        let reduce_group_id = scope.create_local(Elem::UInt);
        cpa!(scope, reduce_group_id = cube_pos_y * cube_count_x);
        cpa!(scope, reduce_group_id += cube_pos_x);

        // nth thread in the cube
        let local_id = scope.create_local(Elem::UInt);
        cpa!(scope, local_id = local_invocation_id_y * cube_dim_x);
        cpa!(scope, local_id += local_invocation_id_x);

        let n_threads = scope.create_local(Elem::UInt);
        cpa!(scope, n_threads = cube_dim_x * cube_dim_y);

        let index_offset = scope.zero(Elem::UInt);

        cpa!(
            scope,
            range(0u32, rank).for_each(|i, scope| {
                let stride_input = scope.create_local(Elem::UInt);
                let stride_output = scope.create_local(Elem::UInt);
                let shape_output = scope.create_local(Elem::UInt);

                cpa!(scope, stride_input = stride(tensor, i));
                cpa!(scope, stride_output = stride(output, i));
                cpa!(scope, shape_output = shape(output, i));

                let num_block = scope.create_local(Elem::UInt);
                cpa!(scope, num_block = reduce_group_id / stride_output);
                cpa!(scope, num_block = num_block % shape_output);
                cpa!(scope, num_block = num_block * stride_input);
                cpa!(scope, index_offset += num_block);
            })
        );

        let shared_memory = RD::initialize_shared(
            scope,
            self.shared_memory_size as u32,
            local_id,
            tensor.item(),
        );

        // Load to shared memory, unrolled
        cpa!(
            scope,
            range(0u32, self.n_input_values_per_thread).for_each(|i, scope| {
                let nth = scope.create_local(Elem::UInt);
                cpa!(scope, nth = i * n_threads);
                cpa!(scope, nth += local_id);

                let within_shape = scope.create_local(Elem::Bool);

                if self.divisible_shape {
                    let current_position = scope.create_local(Elem::UInt);
                    cpa!(scope, current_position = nth * stride_reduce_dim_input);
                    cpa!(scope, current_position += index_offset);

                    let new_value = RD::read_from_input(scope, tensor, current_position, nth);
                    RD::write_to_shared(scope, shared_memory, local_id, new_value);
                } else {
                    cpa!(scope, within_shape = nth < shape_reduce_dim_input);
                    cpa!(scope, if(within_shape).then(|scope|{
                        let current_position = scope.create_local(Elem::UInt);
                        cpa!(scope, current_position = nth * stride_reduce_dim_input);
                        cpa!(scope, current_position += index_offset);

                        let new_value = RD::read_from_input(scope, tensor, current_position, nth);
                        RD::write_to_shared(scope, shared_memory, local_id, new_value);
                    }));
                }
            })
        );

        scope.register(Synchronization::SyncUnits);

        let several_threads_active = scope.create_local(Elem::Bool);

        cpa!(scope, loop(|scope|{
            cpa!(scope, several_threads_active = n_threads <= 1u32);
            cpa!(scope, if(several_threads_active).then(|scope|{
                scope.register(Branch::Break);
            }));

            cpa!(scope, n_threads = n_threads / 2u32);

            let updating_thread = scope.create_local(Elem::Bool);
            cpa!(scope, updating_thread = local_id < n_threads);
            cpa!(scope, if(updating_thread).then(|scope|{
                let read_position = scope.create_local(Elem::UInt);
                cpa!(scope, read_position = n_threads + local_id);

                let read_value = RD::read_from_shared(scope, shared_memory, read_position);
                RD::write_to_shared(scope, shared_memory, local_id, read_value);
            }));

            scope.register(Synchronization::SyncUnits);
        }));

        let is_first_thread = scope.create_local(Elem::Bool);
        cpa!(scope, is_first_thread = local_id == 0u32);
        cpa!(scope, if(is_first_thread).then(|scope|{
            RD::assign_shared(scope, shared_memory, output, reduce_group_id, shape_reduce_dim_input);
        }));
    }
}

/// Executes the shared memory kernel for reduce dim
pub fn reduce_dim_shared<RD: ReduceDimShared<EI>, R: JitRuntime, EI: JitElement, EO: JitElement>(
    input: JitTensor<R, EI>,
    output: JitTensor<R, EO>,
    dim: usize,
) -> JitTensor<R, EO> {
    let num_elems_output = output.shape.num_elements();
    let cube_count_x = f32::ceil(f32::sqrt(num_elems_output as f32));
    let cube_count_y = f32::ceil(num_elems_output as f32 / cube_count_x);
    let grid = CubeCount::Static(cube_count_x as u32, cube_count_y as u32, 1);

    let reduce_group_size = input.shape.dims[dim];
    let n_invocation_per_cube = SUBCUBE_DIM_APPROX * SUBCUBE_DIM_APPROX;
    let n_input_values_per_thread =
        f32::ceil(reduce_group_size as f32 / n_invocation_per_cube as f32) as u32;

    let divisible_shape =
        n_invocation_per_cube as u32 * n_input_values_per_thread == reduce_group_size as u32;

    let kernel = SharedReduceDimEagerKernel::<RD, R, EI, EO>::new(
        dim,
        SUBCUBE_DIM_APPROX,
        SUBCUBE_DIM_APPROX,
        n_input_values_per_thread,
        divisible_shape,
    );

    Execution::start(kernel, input.client.clone())
        .inputs(&[input.as_handle_ref()])
        .outputs(&[output.as_handle_ref()])
        .execute(CubeCountSettings::Custom(grid));

    output
}
