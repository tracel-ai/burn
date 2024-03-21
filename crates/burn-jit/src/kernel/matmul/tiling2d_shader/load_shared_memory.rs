use crate::gpu::{gpu, Elem, Scope, Variable};

use super::{MatmulTiling2dShader, Tiling2dState};

enum InputIdentifier {
    Lhs,
    Rhs,
}

pub(crate) fn load_shared_memory(
    scope: &mut Scope,
    shader: &MatmulTiling2dShader,
    shader_state: &Tiling2dState,
) {
    if shader.bounds_check_required {
        load_shared_memory_with_bound_check(scope, shader, shader_state, InputIdentifier::Lhs);
        load_shared_memory_with_bound_check(scope, shader, shader_state, InputIdentifier::Rhs);
    } else {
        load_shared_memory_no_bound_check(scope, shader, shader_state, InputIdentifier::Lhs);
        load_shared_memory_no_bound_check(scope, shader, shader_state, InputIdentifier::Rhs);
    }
}

#[allow(clippy::too_many_arguments)]
fn load_shared_memory_with_bound_check(
    scope: &mut Scope,
    shader: &MatmulTiling2dShader,
    shader_state: &Tiling2dState,
    input_identifier: InputIdentifier,
) {
    let (
        input,
        input_offset,
        shared_memory,
        thread_idx_1,
        thread_idx_2,
        stride_1,
        stride_2,
        dim,
        pos_in_dim,
    ) = match input_identifier {
        InputIdentifier::Lhs => (
            shader_state.lhs,
            shader_state.offset_lhs,
            shader_state.shared_lhs,
            shader_state.thread_col,
            shader_state.thread_row,
            shader_state.lhs_stride_col,
            shader_state.lhs_stride_row,
            shader_state.dim_m,
            shader_state.row,
        ),
        InputIdentifier::Rhs => (
            shader_state.rhs,
            shader_state.offset_rhs,
            shader_state.shared_rhs,
            shader_state.thread_row,
            shader_state.thread_col,
            shader_state.rhs_stride_row,
            shader_state.rhs_stride_col,
            shader_state.dim_n,
            shader_state.col,
        ),
    };
    let k = shader_state.k;
    let dim_k = shader_state.dim_k;

    // How close is the thread to the end of the matrix.
    // If < 4 then it is an edge case
    let remain = scope.create_local(Elem::UInt);
    gpu!(scope, remain = dim - pos_in_dim);

    let block_size_k: Variable = shader.config.block_size_k.into();
    let block_size_n: Variable = shader.config.block_size_n.into();
    let elem = input.item().elem();

    let current = scope.create_local(Elem::UInt);
    let aligned_with_shared_memory = scope.create_local(Elem::Bool);
    let sm_position = scope.create_local(Elem::UInt);
    let within_input = scope.create_local(Elem::Bool);
    let current_with_k = scope.create_local(Elem::UInt);
    let remain_at_least_1 = scope.create_local(Elem::Bool);
    let read_condition = scope.create_local(Elem::Bool);
    let val_vec4 = scope.create_local(shared_memory.item());

    let tmp = scope.create_local(Elem::UInt);
    let position_0 = scope.create_local(Elem::UInt);
    let position_1 = scope.create_local(Elem::UInt);
    let position_2 = scope.create_local(Elem::UInt);
    let position_3 = scope.create_local(Elem::UInt);
    let remain_n = scope.create_local(Elem::Bool);

    let val_0 = scope.create_local(elem);
    let val_1 = scope.create_local(elem);
    let val_2 = scope.create_local(elem);
    let val_3 = scope.create_local(elem);
    let zero: Variable = 0u32.into();

    gpu!(
        scope,
        range(0_u32, 4u32, shader.unroll).for_each(|j, scope| {
            gpu!(scope, current = thread_idx_1 + j);

            gpu!(scope, aligned_with_shared_memory = current < block_size_k);

            // To avoid overwriting following row in shared memory
            gpu!(scope, if(aligned_with_shared_memory).then(|scope|{

                // Position in shared memory
                match input_identifier {
                    InputIdentifier::Lhs => {
                        gpu!(scope, sm_position = thread_idx_2 / 4u32);
                        gpu!(scope, sm_position *= block_size_k);
                        gpu!(scope, sm_position += current);
                },
                    InputIdentifier::Rhs => {
                        gpu!(scope, sm_position = current * block_size_n);
                        gpu!(scope, sm_position += thread_idx_2);
                        gpu!(scope, sm_position = sm_position / 4u32);
                    }
                }

                // To pad with zeros if outside lhs
                gpu!(scope, current_with_k = current + k);
                gpu!(scope, within_input = current_with_k < dim_k);
                gpu!(scope, remain_at_least_1 = remain >= 1u32);
                gpu!(scope, read_condition = within_input && remain_at_least_1);

                gpu!(scope, if(read_condition).then(|scope| {
                    gpu!(scope, position_0 = k + current);
                    gpu!(scope, position_0 *= stride_1);
                    gpu!(scope, tmp = thread_idx_2 * stride_2);
                    gpu!(scope, position_0 += tmp);
                    gpu!(scope, position_0 += input_offset);
                    gpu!(scope, position_1 = position_0 + stride_2);
                    gpu!(scope, position_2 = position_1 + stride_2);
                    gpu!(scope, position_3 = position_2 + stride_2);

                    gpu!(scope, remain_n = remain >= 4u32);
                    gpu!(scope, if(remain_n).then(|scope|{
                        gpu!(scope, val_0 = input[position_0]);
                        gpu!(scope, val_1 = input[position_1]);
                        gpu!(scope, val_2 = input[position_2]);
                        gpu!(scope, val_3 = input[position_3]);

                    }).else(|scope|{
                        gpu!(scope, remain_n = remain == 3u32);
                        gpu!(scope, if(remain_n).then(|scope|{
                            gpu!(scope, val_0 = input[position_0]);
                            gpu!(scope, val_1 = input[position_1]);
                            gpu!(scope, val_2 = input[position_2]);
                            gpu!(scope, val_3 = zero);

                        }).else(|scope|{
                            gpu!(scope, remain_n = remain == 2u32);
                            gpu!(scope, if(remain_n).then(|scope|{
                                gpu!(scope, val_0 = input[position_0]);
                                gpu!(scope, val_1 = input[position_1]);
                                gpu!(scope, val_2 = zero);
                                gpu!(scope, val_3 = zero);

                            }).else(|scope|{
                                gpu!(scope, remain_n = remain == 1u32);
                                gpu!(scope, if(remain_n).then(|scope|{
                                    gpu!(scope, val_0 = input[position_0]);
                                    gpu!(scope, val_1 = zero);
                                    gpu!(scope, val_2 = zero);
                                    gpu!(scope, val_3 = zero);
                                }));
                            }));
                        }));
                    }));

                    gpu!(scope, val_vec4 = vec4(val_0, val_1, val_2, val_3));
                    gpu!(scope, shared_memory[sm_position] = val_vec4);

                }).else(|scope|{
                    gpu!(scope, val_0 = zero);
                    gpu!(scope, val_vec4 = vec4(val_0, val_0, val_0, val_0));
                    gpu!(scope, shared_memory[sm_position] = val_vec4);
                }));
            }));
        })
    );
}

#[allow(clippy::too_many_arguments)]
fn load_shared_memory_no_bound_check(
    scope: &mut Scope,
    shader: &MatmulTiling2dShader,
    shader_state: &Tiling2dState,
    input_identifier: InputIdentifier,
) {
    let (input, input_offset, shared_memory, thread_idx_1, thread_idx_2, stride_1, stride_2) =
        match input_identifier {
            InputIdentifier::Lhs => (
                shader_state.lhs,
                shader_state.offset_lhs,
                shader_state.shared_lhs,
                shader_state.thread_col,
                shader_state.thread_row,
                shader_state.lhs_stride_col,
                shader_state.lhs_stride_row,
            ),
            InputIdentifier::Rhs => (
                shader_state.rhs,
                shader_state.offset_rhs,
                shader_state.shared_rhs,
                shader_state.thread_row,
                shader_state.thread_col,
                shader_state.rhs_stride_row,
                shader_state.rhs_stride_col,
            ),
        };
    let k = shader_state.k;

    let block_size_k: Variable = shader.config.block_size_k.into();
    let block_size_n: Variable = shader.config.block_size_n.into();
    let elem = input.item().elem();

    let current = scope.create_local(Elem::UInt);
    let aligned_with_shared_memory = scope.create_local(Elem::Bool);
    let sm_position = scope.create_local(Elem::UInt);

    let tmp = scope.create_local(Elem::UInt);
    let position_0 = scope.create_local(Elem::UInt);
    let position_1 = scope.create_local(Elem::UInt);
    let position_2 = scope.create_local(Elem::UInt);
    let position_3 = scope.create_local(Elem::UInt);
    let val_0 = scope.create_local(elem);
    let val_1 = scope.create_local(elem);
    let val_2 = scope.create_local(elem);
    let val_3 = scope.create_local(elem);
    let val_vec4 = scope.create_local(shared_memory.item());

    gpu!(
        scope,
        range(0_u32, 4u32, shader.unroll).for_each(|j, scope| {
            gpu!(scope, current = thread_idx_1 + j);

            gpu!(scope, aligned_with_shared_memory = current < block_size_k);

            // To avoid overwriting following row in shared memory
            gpu!(scope, if(aligned_with_shared_memory).then(|scope|{

                match input_identifier {
                    InputIdentifier::Lhs => {
                        gpu!(scope, sm_position = thread_idx_2 / 4u32);
                        gpu!(scope, sm_position *= block_size_k);
                        gpu!(scope, sm_position += current);
                },
                    InputIdentifier::Rhs => {
                        gpu!(scope, sm_position = current * block_size_n);
                        gpu!(scope, sm_position += thread_idx_2);
                        gpu!(scope, sm_position = sm_position / 4u32);
                    }
                }

                gpu!(scope, position_0 = k + current);
                gpu!(scope, position_0 *= stride_1);
                gpu!(scope, tmp = thread_idx_2 * stride_2);
                gpu!(scope, position_0 += tmp);
                gpu!(scope, position_0 += input_offset);
                gpu!(scope, position_1 = position_0 + stride_2);
                gpu!(scope, position_2 = position_1 + stride_2);
                gpu!(scope, position_3 = position_2 + stride_2);

                gpu!(scope, val_0 = input[position_0]);
                gpu!(scope, val_1 = input[position_1]);
                gpu!(scope, val_2 = input[position_2]);
                gpu!(scope, val_3 = input[position_3]);

                gpu!(scope, val_vec4 = vec4(val_0, val_1, val_2, val_3));
                gpu!(scope, shared_memory[sm_position] = val_vec4);
            }));
        })
    );
}
