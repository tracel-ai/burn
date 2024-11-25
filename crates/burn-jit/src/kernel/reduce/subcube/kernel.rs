use cubecl::{prelude::*, CubeCount, CubeDim, Feature};

use crate::{
    kernel::reduce::{init_reduce_output, shared::kernel::reduce_dim_shared, ReduceDimAlgorithm},
    tensor::JitTensor,
    JitElement, JitRuntime,
};

use super::base::ReduceDimSubcube;

#[cube(launch)]
pub fn reduce_dim_subcube_kernel<
    RD: ReduceDimSubcube<EIn, EOut>,
    EIn: JitElement,
    EOut: JitElement,
>(
    input: &Tensor<EIn>,
    output: &mut Tensor<EOut>,
    #[comptime] dim: u32,
    #[comptime] subcube_size: u32,
    #[comptime] elems_per_thread: u32,
    #[comptime] divisible_shape: bool,
) {
    let reduce_group_id = CUBE_POS;

    let stride_reduce_dim_input = input.stride(dim);
    let shape_reduce_dim_input = input.shape(dim);

    let should_unroll = elems_per_thread <= 8;

    let warp_id = UNIT_POS / PLANE_DIM;

    let mut shared_memory = RD::init_shared(subcube_size);

    let mut index_offset = 0;

    for i in 0..input.rank() {
        let num_block = reduce_group_id / output.stride(i) % output.shape(i);
        index_offset += num_block * input.stride(i);
    }

    let mut value = RD::init_value();

    #[unroll(should_unroll)]
    for i in 0..elems_per_thread {
        let nth = i * CUBE_DIM + UNIT_POS;
        let current_pos = nth * stride_reduce_dim_input + index_offset;

        #[allow(clippy::collapsible_else_if)]
        if divisible_shape {
            let next = RD::read_value(input, current_pos, nth);
            RD::update_value(&mut value, next);
        } else {
            if nth < shape_reduce_dim_input {
                let next = RD::read_value(input, current_pos, nth);
                RD::update_value(&mut value, next);
            }
        }
    }

    RD::reduce_subcube(&mut shared_memory, warp_id, value);

    sync_units();

    if UNIT_POS >= PLANE_DIM {
        return;
    }

    let value = RD::read_from_shared(&shared_memory, UNIT_POS);
    RD::reduce_subcube(&mut shared_memory, 0, value);

    if UNIT_POS == 0 {
        RD::store(
            &shared_memory,
            output,
            reduce_group_id,
            shape_reduce_dim_input,
        );
    }
}

/// Executes the shared memory kernel for reduce dim
pub fn reduce_dim_subcube<
    RD: ReduceDimAlgorithm<EI, EO>,
    R: JitRuntime,
    EI: JitElement,
    EO: JitElement,
>(
    input: JitTensor<R>,
    dim: usize,
) -> JitTensor<R> {
    let topology = input.client.properties().hardware_properties();

    if !input.client.properties().feature_enabled(Feature::Plane)
        || topology.plane_size_min != topology.plane_size_max
    {
        return reduce_dim_shared::<RD, R, EI, EO>(input, dim);
    }

    let subcube_size = topology.plane_size_min;

    let output = init_reduce_output::<R, EI, EO>(&input, dim);

    let num_elems_output = output.shape.num_elements();
    let cube_dim = CubeDim {
        x: subcube_size,
        y: subcube_size,
        z: 1,
    };
    let cube_count_x = f32::ceil(f32::sqrt(num_elems_output as f32));
    let cube_count_y = f32::ceil(num_elems_output as f32 / cube_count_x);
    let cube_count = CubeCount::Static(cube_count_x as u32, cube_count_y as u32, 1);

    let reduce_group_size = input.shape.dims[dim];
    let n_invocation_per_cube = cube_dim.num_elems();
    let elems_per_thread =
        f32::ceil(reduce_group_size as f32 / n_invocation_per_cube as f32) as u32;

    let divisible_shape = n_invocation_per_cube * elems_per_thread == reduce_group_size as u32;

    reduce_dim_subcube_kernel::launch::<RD, EI, EO, R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg::<EI>(1),
        output.as_tensor_arg::<EO>(1),
        dim as u32,
        subcube_size,
        elems_per_thread,
        divisible_shape,
    );

    output
}
