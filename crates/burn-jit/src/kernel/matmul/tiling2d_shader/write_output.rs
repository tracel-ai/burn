use burn_cube::{
    cpa,
    ir::{Elem, Scope, Variable},
};

use super::{MatmulTiling2dShader, Tiling2dState};

#[allow(clippy::too_many_arguments)]
pub fn write_to_output(
    scope: &mut Scope,
    shader: &MatmulTiling2dShader,
    shader_state: &Tiling2dState,
) {
    let row = shader_state.row;
    let col = shader_state.col;

    let row_index = scope.create_local(Elem::UInt);
    let col_index = scope.create_local(Elem::UInt);

    if shader.bounds_check_required {
        let dim_m = shader_state.dim_m;
        let dim_n = shader_state.dim_n;

        let within_output = scope.create_local(Elem::Bool);
        let within_output_tmp = scope.create_local(Elem::Bool);

        cpa!(
            scope,
            range(0u32, shader.config.tile_size_m as u32, shader.config.unroll).for_each(
                |res_idx_m, scope| {
                    cpa!(
                        scope,
                        range(0u32, shader.config.tile_size_n as u32, shader.config.unroll)
                            .for_each(|res_idx_n, scope| {
                                cpa!(scope, row_index = row + res_idx_m);
                                cpa!(scope, col_index = col + res_idx_n);

                                cpa!(scope, within_output = row_index < dim_m);
                                cpa!(scope, within_output_tmp = col_index < dim_n);
                                cpa!(scope, within_output = within_output && within_output_tmp);

                                cpa!(scope, if(within_output).then(|scope|{
                                    write_inner(
                                        scope,
                                        shader,
                                        shader_state,
                                        res_idx_m,
                                        res_idx_n,
                                        row_index,
                                        col_index,
                                    );
                                }));
                            })
                    );
                }
            )
        );
    } else {
        cpa!(
            scope,
            range(0u32, shader.config.tile_size_m as u32, shader.config.unroll).for_each(
                |res_idx_m, scope| {
                    cpa!(
                        scope,
                        range(0u32, shader.config.tile_size_n as u32, shader.config.unroll)
                            .for_each(|res_idx_n, scope| {
                                cpa!(scope, row_index = row + res_idx_m);
                                cpa!(scope, col_index = col + res_idx_n);

                                write_inner(
                                    scope,
                                    shader,
                                    shader_state,
                                    res_idx_m,
                                    res_idx_n,
                                    row_index,
                                    col_index,
                                )
                            })
                    );
                }
            )
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn write_inner(
    scope: &mut Scope,
    shader: &MatmulTiling2dShader,
    shader_state: &Tiling2dState,
    res_idx_m: Variable,
    res_idx_n: Variable,
    row_index: Variable,
    col_index: Variable,
) {
    let offset_output = shader_state.offset_output;
    let out = shader_state.out;
    let out_stride_row = shader_state.out_stride_row;
    let out_stride_col = shader_state.out_stride_col;
    let results = shader_state.results;

    let elem = results.item().elem();
    let results_position = scope.create_local(Elem::UInt);
    let result = scope.create_local(elem);
    let output_position = scope.create_local(Elem::UInt);

    cpa!(
        scope,
        results_position = res_idx_m * shader.config.tile_size_n
    );
    cpa!(scope, results_position += res_idx_n);

    cpa!(scope, result = results[results_position]);

    cpa!(scope, row_index *= out_stride_row);
    cpa!(scope, col_index *= out_stride_col);
    cpa!(scope, output_position = row_index + col_index);
    cpa!(scope, output_position += offset_output);

    cpa!(scope, out[output_position] = result);
}
