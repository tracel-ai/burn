use crate::gpu::{gpu, Elem, Item, Scope, Variable};

use super::{MatmulTiling2dShader, Tiling2dState};

pub(crate) fn gather_shader_information(
    scope: &mut Scope,
    shader: &MatmulTiling2dShader,
) -> Tiling2dState {
    // Inputs
    let lhs = shader.variables.lhs;
    let rhs = shader.variables.rhs;
    let out = shader.variables.out;

    // Config variables
    let block_size_m: Variable = shader.config.block_size_m.into();
    let block_size_k: Variable = shader.config.block_size_k.into();
    let block_size_n: Variable = shader.config.block_size_n.into();
    let tile_size_m: Variable = shader.config.tile_size_m.into();
    let tile_size_n: Variable = shader.config.tile_size_n.into();
    let n_threads_per_row: Variable =
        (((shader.config.block_size_n - 1) / shader.config.tile_size_n) + 1).into();
    let results_size = (shader.config.tile_size_m * shader.config.tile_size_n) as u32;

    // Shader info
    let local_idx = Variable::LocalInvocationIndex;
    let batch = Variable::GlobalInvocationIdZ;

    // Shapes
    let rank = Variable::Rank;
    let last_dim = scope.create_local(Elem::UInt);
    let second_to_last_dim = scope.create_local(Elem::UInt);
    let dim_m = scope.create_local(Elem::UInt);
    let dim_k = scope.create_local(Elem::UInt);
    let dim_n = scope.create_local(Elem::UInt);
    gpu!(scope, last_dim = rank - 1u32);
    gpu!(scope, second_to_last_dim = rank - 2u32);
    gpu!(scope, dim_m = shape(lhs, second_to_last_dim));
    gpu!(scope, dim_k = shape(lhs, last_dim));
    gpu!(scope, dim_n = shape(rhs, last_dim));

    // Strides
    let lhs_stride_row = scope.create_local(Elem::UInt);
    let lhs_stride_col = scope.create_local(Elem::UInt);
    let rhs_stride_row = scope.create_local(Elem::UInt);
    let rhs_stride_col = scope.create_local(Elem::UInt);
    let out_stride_row = scope.create_local(Elem::UInt);
    let out_stride_col = scope.create_local(Elem::UInt);
    gpu!(scope, lhs_stride_row = stride(lhs, second_to_last_dim));
    gpu!(scope, lhs_stride_col = stride(lhs, last_dim));
    gpu!(scope, rhs_stride_row = stride(rhs, second_to_last_dim));
    gpu!(scope, rhs_stride_col = stride(rhs, last_dim));
    gpu!(scope, out_stride_row = stride(out, second_to_last_dim));
    gpu!(scope, out_stride_col = stride(out, last_dim));

    // Workgroup offset
    let skip_row = scope.create_local(Elem::UInt);
    let skip_col = scope.create_local(Elem::UInt);
    let workgroup_id_x = Variable::WorkgroupIdX;
    let workgroup_id_y = Variable::WorkgroupIdY;
    gpu!(scope, skip_row = workgroup_id_x);
    gpu!(scope, skip_row *= block_size_m);
    gpu!(scope, skip_col = workgroup_id_y);
    gpu!(scope, skip_col *= block_size_n);

    // Position of the first element of the thread, relative to the block
    let thread_row = scope.create_local(Elem::UInt);
    let thread_col = scope.create_local(Elem::UInt);
    gpu!(scope, thread_row = local_idx / n_threads_per_row);
    gpu!(scope, thread_row *= tile_size_m);
    gpu!(scope, thread_col = local_idx % n_threads_per_row);
    gpu!(scope, thread_col *= tile_size_n);

    // Position of the first element of the thread, in absolute (in one batch)
    let row = scope.create_local(Elem::UInt);
    let col = scope.create_local(Elem::UInt);
    gpu!(scope, row = skip_row + thread_row);
    gpu!(scope, col = skip_col + thread_col);

    // Calculate offset.
    let offset_lhs = scope.create_local(Elem::UInt);
    let offset_rhs = scope.create_local(Elem::UInt);
    gpu!(scope, offset_lhs = skip_row * lhs_stride_row);
    gpu!(scope, offset_rhs = skip_col * rhs_stride_col);

    // Batch offset for the output.
    let offset_output = scope.create_local(Elem::UInt);
    let batch_dims = scope.create_local(Elem::UInt);
    gpu!(scope, offset_output = dim_m * dim_n);
    gpu!(scope, offset_output = offset_output * batch);

    // Batch offset for the lhs & rhs matrices.
    let stride_lhs = scope.create_local(Elem::UInt);
    let stride_rhs = scope.create_local(Elem::UInt);
    let stride_output = scope.create_local(Elem::UInt);
    let shape_lhs = scope.create_local(Elem::UInt);
    let shape_rhs = scope.create_local(Elem::UInt);
    let tmp = scope.create_local(Elem::UInt);
    let tmp_lhs = scope.create_local(Elem::UInt);
    let tmp_rhs = scope.create_local(Elem::UInt);
    gpu!(scope, batch_dims = rank - 2u32);
    gpu!(
        scope,
        range(0u32, batch_dims).for_each(|b, scope| {
            gpu!(scope, stride_lhs = stride(lhs, b));
            gpu!(scope, stride_rhs = stride(rhs, b));
            gpu!(scope, stride_output = stride(out, b));
            gpu!(scope, shape_lhs = shape(lhs, b));
            gpu!(scope, shape_rhs = shape(rhs, b));

            gpu!(scope, tmp = offset_output / stride_output);
            gpu!(scope, tmp_lhs = tmp % shape_lhs);
            gpu!(scope, tmp_lhs = tmp_lhs * stride_lhs);
            gpu!(scope, offset_lhs += tmp_lhs);

            gpu!(scope, tmp_rhs = tmp % shape_rhs);
            gpu!(scope, tmp_rhs = tmp_rhs * stride_rhs);
            gpu!(scope, offset_rhs += tmp_rhs);
        })
    );

    let elem = lhs.item().elem();

    // Registers used in the compute pass
    let results = scope.create_local_array(elem, results_size);
    let register_m = scope.create_local(Item::Vec4(elem));
    let register_n = scope.create_local(Item::Vec4(elem));
    let shared_lhs = scope.create_shared(
        Item::Vec4(elem),
        shader.config.block_size_m as u32 * shader.config.block_size_k as u32 / 4u32,
    );
    let shared_rhs = scope.create_shared(
        Item::Vec4(elem),
        shader.config.block_size_k as u32 * shader.config.block_size_n as u32 / 4u32,
    );

    // Calculate exact number of loop iterations
    let n_loops = scope.create_local(Elem::UInt);
    let k = scope.create_local(Elem::UInt);
    if shader.bounds_check_required {
        let dim_k_float = scope.create_local(elem);
        let block_size_k_float = scope.create_local(elem);
        let n_loops_float = scope.create_local(elem);
        gpu!(scope, dim_k_float = dim_k);
        gpu!(scope, block_size_k_float = block_size_k);
        gpu!(scope, n_loops_float = dim_k_float / block_size_k_float);
        gpu!(scope, n_loops_float = ceil(n_loops_float));
        gpu!(scope, n_loops = n_loops_float);
    } else {
        gpu!(scope, n_loops = dim_k / block_size_k);
    }

    Tiling2dState {
        n_loops,
        k,
        lhs,
        rhs,
        out,
        offset_lhs,
        offset_rhs,
        offset_output,
        row,
        col,
        dim_m,
        dim_k,
        dim_n,
        thread_col,
        thread_row,
        shared_lhs,
        shared_rhs,
        register_m,
        register_n,
        results,
        lhs_stride_col,
        lhs_stride_row,
        rhs_stride_col,
        rhs_stride_row,
        out_stride_row,
        out_stride_col,
    }
}
