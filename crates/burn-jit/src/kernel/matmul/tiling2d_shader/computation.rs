use crate::gpu::{gpu, Elem, Scope, Variable};

use super::{MatmulTiling2dShader, Tiling2dState};

#[allow(clippy::too_many_arguments)]
pub fn computation_loop(
    scope: &mut Scope,
    shader: &MatmulTiling2dShader,
    shader_state: &Tiling2dState,
) {
    let thread_col = shader_state.thread_col;
    let thread_row = shader_state.thread_row;
    let shared_lhs = shader_state.shared_lhs;
    let shared_rhs = shader_state.shared_rhs;
    let register_m = shader_state.register_m;
    let register_n = shader_state.register_n;
    let results = shader_state.results;

    let block_size_k: Variable = shader.config.block_size_k.into();
    let block_size_n: Variable = shader.config.block_size_n.into();
    let elem = results.item().elem();

    let lhs_sm_position = scope.create_local(Elem::UInt);
    let rhs_sm_position = scope.create_local(Elem::UInt);

    let registered_m = scope.create_local(elem);
    let registered_n = scope.create_local(elem);

    let multiplied = scope.create_local(elem);
    let results_position = scope.create_local(Elem::UInt);
    let results_before = scope.create_local(elem);
    let results_after = scope.create_local(elem);

    gpu!(
        scope,
        range(0u32, shader.config.block_size_k as u32, shader.unroll).for_each(
            |dot_index, scope| {
                // Load a subcolumn of values from lhs
                gpu!(scope, lhs_sm_position = thread_row / 4u32);
                gpu!(scope, lhs_sm_position *= block_size_k);
                gpu!(scope, lhs_sm_position += dot_index);
                gpu!(scope, register_m = shared_lhs[lhs_sm_position]);

                // Load a subrow of values from rhs
                gpu!(scope, rhs_sm_position = dot_index * block_size_n);
                gpu!(scope, rhs_sm_position += thread_col);
                gpu!(scope, rhs_sm_position = rhs_sm_position / 4u32);
                gpu!(scope, register_n = shared_rhs[rhs_sm_position]);

                gpu!(
                    scope,
                    range(0u32, shader.config.tile_size_m as u32, shader.unroll).for_each(
                        |res_idx_m, scope| {
                            gpu!(
                                scope,
                                range(0u32, shader.config.tile_size_n as u32, shader.unroll)
                                    .for_each(|res_idx_n, scope| {
                                        gpu!(scope, registered_m = register_m[res_idx_m]);
                                        gpu!(scope, registered_n = register_n[res_idx_n]);

                                        gpu!(scope, multiplied = registered_m * registered_n);

                                        gpu!(
                                            scope,
                                            results_position =
                                                res_idx_m * shader.config.tile_size_n
                                        );
                                        gpu!(scope, results_position += res_idx_n);

                                        gpu!(scope, results_before = results[results_position]);
                                        gpu!(scope, results_after = results_before + multiplied);

                                        gpu!(scope, results[results_position] = results_after);
                                    })
                            );
                        }
                    )
                );
            }
        )
    );
}
