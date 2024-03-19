use crate::gpu::{gpu, BinaryOperator, Elem, Item, Scope, Synchronization, Variable};

use super::Tiling2dConfig;

pub(crate) struct MatmulTiling2dShader {
    pub variables: BinaryOperator,
    pub config: Tiling2dConfig,
    pub bounds_check_required: bool,
    pub unroll: bool,
}

impl MatmulTiling2dShader {
    pub(crate) fn expand(self, scope: &mut Scope) {
        // Inputs
        let lhs = self.variables.lhs;
        let rhs = self.variables.rhs;
        let out = self.variables.out;

        // Config variables
        let block_size_m: Variable = self.config.block_size_m.into();
        let block_size_k: Variable = self.config.block_size_k.into();
        let block_size_n: Variable = self.config.block_size_n.into();
        let tile_size_m: Variable = self.config.tile_size_m.into();
        let tile_size_n: Variable = self.config.tile_size_n.into();
        let n_threads_per_row: Variable =
            (((self.config.block_size_n - 1) / self.config.tile_size_n) + 1).into();
        let results_size = (self.config.tile_size_m * self.config.tile_size_n) as u32;

        // Shader info
        let local_idx = Variable::LocalInvocationIndex;
        let batch = Variable::GlobalInvocationIdZ;

        // Shapes
        let rank = Variable::Rank;
        let ultimate_dim = scope.create_local(Elem::UInt);
        let penultimate_dim = scope.create_local(Elem::UInt);
        gpu!(scope, ultimate_dim = rank - 1u32);
        gpu!(scope, penultimate_dim = rank - 2u32);
        let dim_m = scope.create_local(Elem::UInt);
        let dim_k = scope.create_local(Elem::UInt);
        let dim_n = scope.create_local(Elem::UInt);
        gpu!(scope, dim_m = shape(lhs, penultimate_dim));
        gpu!(scope, dim_k = shape(lhs, ultimate_dim));
        gpu!(scope, dim_n = shape(rhs, ultimate_dim));

        // Strides
        let lhs_stride_row = scope.create_local(Elem::UInt);
        let lhs_stride_col = scope.create_local(Elem::UInt);
        let rhs_stride_row = scope.create_local(Elem::UInt);
        let rhs_stride_col = scope.create_local(Elem::UInt);
        let out_stride_row = scope.create_local(Elem::UInt);
        let out_stride_col = scope.create_local(Elem::UInt);
        gpu!(scope, lhs_stride_row = stride(lhs, penultimate_dim));
        gpu!(scope, lhs_stride_col = stride(lhs, ultimate_dim));
        gpu!(scope, rhs_stride_row = stride(rhs, penultimate_dim));
        gpu!(scope, rhs_stride_col = stride(rhs, ultimate_dim));
        gpu!(scope, out_stride_row = stride(out, penultimate_dim));
        gpu!(scope, out_stride_col = stride(out, ultimate_dim));

        // Workgroup offset
        let skip_row = scope.create_local(Elem::UInt);
        let workgroup_id_x = Variable::WorkgroupIdX;
        gpu!(scope, skip_row = workgroup_id_x);
        gpu!(scope, skip_row *= block_size_m);
        let skip_col = scope.create_local(Elem::UInt);
        let workgroup_id_y = Variable::WorkgroupIdY;
        gpu!(scope, skip_col = workgroup_id_y);
        gpu!(scope, skip_col *= block_size_n);

        // Position of the first element of the thread, relative to the block
        let thread_row = scope.create_local(Elem::UInt);
        gpu!(scope, thread_row = local_idx / n_threads_per_row);
        gpu!(scope, thread_row *= tile_size_m);
        let thread_col = scope.create_local(Elem::UInt);
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
        gpu!(scope, batch_dims = rank - 2u32);
        gpu!(
            scope,
            range(0u32, batch_dims).for_each(|b, scope| {
                let stride_lhs = scope.create_local(Elem::UInt);
                let stride_rhs = scope.create_local(Elem::UInt);
                let stride_output = scope.create_local(Elem::UInt);
                let shape_lhs = scope.create_local(Elem::UInt);
                let shape_rhs = scope.create_local(Elem::UInt);

                gpu!(scope, stride_lhs = stride(lhs, b));
                gpu!(scope, stride_rhs = stride(rhs, b));
                gpu!(scope, stride_output = stride(out, b));
                gpu!(scope, shape_lhs = shape(lhs, b));
                gpu!(scope, shape_rhs = shape(rhs, b));

                let tmp = scope.create_local(Elem::UInt);
                gpu!(scope, tmp = offset_output / stride_output);
                let tmp_lhs = scope.create_local(Elem::UInt);
                gpu!(scope, tmp_lhs = tmp % shape_lhs);
                gpu!(scope, tmp_lhs = tmp_lhs * stride_lhs);
                gpu!(scope, offset_lhs += tmp_lhs);

                let tmp_rhs = scope.create_local(Elem::UInt);
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
            self.config.block_size_m as u32 * self.config.block_size_k as u32 / 4u32,
        );
        let shared_rhs = scope.create_shared(
            Item::Vec4(elem),
            self.config.block_size_k as u32 * self.config.block_size_n as u32 / 4u32,
        );

        let n_loops = scope.create_local(Elem::UInt);
        if self.bounds_check_required {
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

        gpu!(
            scope,
            range(0u32, n_loops).for_each(|i, scope| {
                // Equivalent of looping from 0 to K with steps block_size_k
                let k = scope.create_local(Elem::UInt);
                gpu!(scope, k = i * block_size_k);

                if self.bounds_check_required {
                    // LHS
                    self.load_shared_memory_with_bound_check(
                        scope,
                        k,
                        dim_k,
                        thread_col,
                        thread_row,
                        lhs_stride_col,
                        lhs_stride_row,
                        dim_m,
                        row,
                        lhs,
                        offset_lhs,
                        shared_lhs,
                        true,
                    );

                    // RHS
                    self.load_shared_memory_with_bound_check(
                        scope,
                        k,
                        dim_k,
                        thread_row,
                        thread_col,
                        rhs_stride_row,
                        rhs_stride_col,
                        dim_n,
                        col,
                        rhs,
                        offset_rhs,
                        shared_rhs,
                        false,
                    );
                } else {
                    // LHS
                    self.load_shared_memory(
                        scope,
                        k,
                        thread_col,
                        thread_row,
                        lhs_stride_col,
                        lhs_stride_row,
                        lhs,
                        offset_lhs,
                        shared_lhs,
                        true,
                    );

                    // RHS
                    self.load_shared_memory(
                        scope,
                        k,
                        thread_row,
                        thread_col,
                        rhs_stride_row,
                        rhs_stride_col,
                        rhs,
                        offset_rhs,
                        shared_rhs,
                        false,
                    );
                }

                scope.register(Synchronization::WorkgroupBarrier);

                self.computation_loop(
                    scope, thread_col, thread_row, shared_lhs, shared_rhs, register_m, register_n,
                    results,
                );

                scope.register(Synchronization::WorkgroupBarrier);
            })
        );

        // Phase 3: Write to output
        self.write_to_output(
            scope,
            row,
            col,
            dim_m,
            dim_n,
            out_stride_row,
            out_stride_col,
            results,
            offset_output,
            out,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn load_shared_memory_with_bound_check(
        &self,
        scope: &mut Scope,
        k: Variable,
        dim_k: Variable,
        thread_idx_1: Variable,
        thread_idx_2: Variable,
        stride_1: Variable,
        stride_2: Variable,
        dim: Variable,
        pos_in_dim: Variable,
        input: Variable,
        input_offset: Variable,
        shared_memory: Variable,
        is_lhs: bool,
    ) {
        // How close is the thread to the end of the matrix.
        // If < 4 then it is an edge case

        let remain = scope.create_local(Elem::UInt);
        gpu!(scope, remain = dim - pos_in_dim);

        let block_size_k: Variable = self.config.block_size_k.into();
        let block_size_n: Variable = self.config.block_size_n.into();
        let elem = input.item().elem();

        gpu!(
            scope,
            range(0_u32, 4u32, self.unroll).for_each(|j, scope| {
                let current = scope.create_local(Elem::UInt);
                gpu!(scope, current = thread_idx_1 + j);

                let aligned_with_shared_memory = scope.create_local(Elem::Bool);
                gpu!(scope, aligned_with_shared_memory = current < block_size_k);

                // To avoid overwriting following row in shared memory
                gpu!(scope, if(aligned_with_shared_memory).then(|scope|{

                    // Position in shared memory
                    let sm_position = scope.create_local(Elem::UInt);
                    if is_lhs {
                        gpu!(scope, sm_position = thread_idx_2 / 4u32);
                        gpu!(scope, sm_position *= block_size_k);
                        gpu!(scope, sm_position += current);
                    } else {
                        gpu!(scope, sm_position = current * block_size_n);
                        gpu!(scope, sm_position += thread_idx_2);
                        gpu!(scope, sm_position = sm_position / 4u32);
                    }

                    // To pad with zeros if outside lhs
                    let within_input = scope.create_local(Elem::Bool);
                    let current_with_k = scope.create_local(Elem::UInt);
                    let remain_at_least_1 = scope.create_local(Elem::Bool);
                    let read_condition = scope.create_local(Elem::Bool);
                    gpu!(scope, current_with_k = current + k);
                    gpu!(scope, within_input = current_with_k < dim_k);
                    gpu!(scope, remain_at_least_1 = remain >= 1u32);
                    gpu!(scope, read_condition = within_input && remain_at_least_1);

                    gpu!(scope, if(read_condition).then(|scope| {
                        let position_0 = scope.create_local(Elem::UInt);
                        gpu!(scope, position_0 = k + current);
                        gpu!(scope, position_0 *= stride_1);
                        let tmp = scope.create_local(Elem::UInt);
                        gpu!(scope, tmp = thread_idx_2 * stride_2);
                        gpu!(scope, position_0 += tmp);
                        gpu!(scope, position_0 += input_offset);
                        let position_1 = scope.create_local(Elem::UInt);
                        let position_2 = scope.create_local(Elem::UInt);
                        let position_3 = scope.create_local(Elem::UInt);
                        gpu!(scope, position_1 = position_0 + stride_2);
                        gpu!(scope, position_2 = position_1 + stride_2);
                        gpu!(scope, position_3 = position_2 + stride_2);

                        let val_0 = scope.zero(elem);
                        let val_1 = scope.zero(elem);
                        let val_2 = scope.zero(elem);
                        let val_3 = scope.zero(elem);

                        let remain_n = scope.create_local(Elem::Bool);
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
                            }).else(|scope|{
                                gpu!(scope, remain_n = remain == 2u32);
                                gpu!(scope, if(remain_n).then(|scope|{
                                    gpu!(scope, val_0 = input[position_0]);
                                    gpu!(scope, val_1 = input[position_1]);
                                }).else(|scope|{
                                    gpu!(scope, remain_n = remain == 1u32);
                                    gpu!(scope, if(remain_n).then(|scope|{
                                        gpu!(scope, val_0 = input[position_0]);
                                    }));
                                }));
                            }));
                        }));

                        let val_vec4 = scope.create_local(shared_memory.item());
                        gpu!(scope, val_vec4 = vec4(val_0, val_1, val_2, val_3));
                        gpu!(scope, shared_memory[sm_position] = val_vec4);
                    }).else(|scope|{
                        let val_0 = scope.zero(elem);
                        let val_vec4 = scope.create_local(shared_memory.item());
                        gpu!(scope, val_vec4 = vec4(val_0, val_0, val_0, val_0));
                        gpu!(scope, shared_memory[sm_position] = val_vec4);
                    }));
                }));
            })
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn load_shared_memory(
        &self,
        scope: &mut Scope,
        k: Variable,
        thread_idx_1: Variable,
        thread_idx_2: Variable,
        stride_1: Variable,
        stride_2: Variable,
        input: Variable,
        input_offset: Variable,
        shared_memory: Variable,
        is_lhs: bool,
    ) {
        let block_size_k: Variable = self.config.block_size_k.into();
        let block_size_n: Variable = self.config.block_size_n.into();
        let elem = input.item().elem();

        gpu!(
            scope,
            range(0_u32, 4u32, self.unroll).for_each(|j, scope| {
                let current = scope.create_local(Elem::UInt);
                gpu!(scope, current = thread_idx_1 + j);

                let aligned_with_shared_memory = scope.create_local(Elem::Bool);
                gpu!(scope, aligned_with_shared_memory = current < block_size_k);

                // To avoid overwriting following row in shared memory
                gpu!(scope, if(aligned_with_shared_memory).then(|scope|{

                    let sm_position = scope.create_local(Elem::UInt);
                    if is_lhs {
                        gpu!(scope, sm_position = thread_idx_2 / 4u32);
                        gpu!(scope, sm_position *= block_size_k);
                        gpu!(scope, sm_position += current);
                    } else {
                        gpu!(scope, sm_position = current * block_size_n);
                        gpu!(scope, sm_position += thread_idx_2);
                        gpu!(scope, sm_position = sm_position / 4u32);
                    }

                    let position_0 = scope.create_local(Elem::UInt);
                    gpu!(scope, position_0 = k + current);
                    gpu!(scope, position_0 *= stride_1);
                    let tmp = scope.create_local(Elem::UInt);
                    gpu!(scope, tmp = thread_idx_2 * stride_2);
                    gpu!(scope, position_0 += tmp);
                    gpu!(scope, position_0 += input_offset);
                    let position_1 = scope.create_local(Elem::UInt);
                    let position_2 = scope.create_local(Elem::UInt);
                    let position_3 = scope.create_local(Elem::UInt);
                    gpu!(scope, position_1 = position_0 + stride_2);
                    gpu!(scope, position_2 = position_1 + stride_2);
                    gpu!(scope, position_3 = position_2 + stride_2);

                    let val_0 = scope.create_local(elem);
                    let val_1 = scope.create_local(elem);
                    let val_2 = scope.create_local(elem);
                    let val_3 = scope.create_local(elem);
                    gpu!(scope, val_0 = input[position_0]);
                    gpu!(scope, val_1 = input[position_1]);
                    gpu!(scope, val_2 = input[position_2]);
                    gpu!(scope, val_3 = input[position_3]);

                    let val_vec4 = scope.create_local(shared_memory.item());
                    gpu!(scope, val_vec4 = vec4(val_0, val_1, val_2, val_3));
                    gpu!(scope, shared_memory[sm_position] = val_vec4);
                }));
            })
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn computation_loop(
        &self,
        scope: &mut Scope,
        thread_col: Variable,
        thread_row: Variable,
        shared_lhs: Variable,
        shared_rhs: Variable,
        register_m: Variable,
        register_n: Variable,
        results: Variable,
    ) {
        let block_size_k: Variable = self.config.block_size_k.into();
        let block_size_n: Variable = self.config.block_size_n.into();
        let elem = results.item().elem();

        gpu!(
            scope,
            range(0u32, self.config.block_size_k as u32, self.unroll).for_each(
                |dot_index, scope| {
                    // Load a subcolumn of values from lhs
                    let lhs_sm_position = scope.create_local(Elem::UInt);
                    gpu!(scope, lhs_sm_position = thread_row / 4u32);
                    gpu!(scope, lhs_sm_position *= block_size_k);
                    gpu!(scope, lhs_sm_position += dot_index);
                    gpu!(scope, register_m = shared_lhs[lhs_sm_position]);

                    // Load a subrow of values from rhs
                    let rhs_sm_position = scope.create_local(Elem::UInt);
                    gpu!(scope, rhs_sm_position = dot_index * block_size_n);
                    gpu!(scope, rhs_sm_position += thread_col);
                    gpu!(scope, rhs_sm_position = rhs_sm_position / 4u32);
                    gpu!(scope, register_n = shared_rhs[rhs_sm_position]);

                    gpu!(
                        scope,
                        range(0u32, self.config.tile_size_m as u32, self.unroll).for_each(
                            |res_idx_m, scope| {
                                gpu!(
                                    scope,
                                    range(0u32, self.config.tile_size_n as u32, self.unroll)
                                        .for_each(|res_idx_n, scope| {
                                            let registered_m = scope.create_local(elem);
                                            let registered_n = scope.create_local(elem);
                                            gpu!(scope, registered_m = register_m[res_idx_m]);
                                            gpu!(scope, registered_n = register_n[res_idx_n]);

                                            let multiplied = scope.create_local(elem);
                                            gpu!(scope, multiplied = registered_m * registered_n);

                                            let results_position = scope.create_local(Elem::UInt);
                                            gpu!(
                                                scope,
                                                results_position =
                                                    res_idx_m * self.config.tile_size_n
                                            );
                                            gpu!(scope, results_position += res_idx_n);

                                            let results_before = scope.create_local(elem);
                                            gpu!(scope, results_before = results[results_position]);
                                            let results_after = scope.create_local(elem);
                                            gpu!(
                                                scope,
                                                results_after = results_before + multiplied
                                            );

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

    #[allow(clippy::too_many_arguments)]
    fn write_to_output(
        &self,
        scope: &mut Scope,
        row: Variable,
        col: Variable,
        dim_m: Variable,
        dim_n: Variable,
        out_stride_row: Variable,
        out_stride_col: Variable,
        results: Variable,
        offset_output: Variable,
        out: Variable,
    ) {
        gpu!(
            scope,
            range(0u32, self.config.tile_size_m as u32, self.unroll).for_each(
                |res_idx_m, scope| {
                    gpu!(
                        scope,
                        range(0u32, self.config.tile_size_n as u32, self.unroll).for_each(
                            |res_idx_n, scope| {
                                let row_index = scope.create_local(Elem::UInt);
                                let col_index = scope.create_local(Elem::UInt);
                                gpu!(scope, row_index = row + res_idx_m);
                                gpu!(scope, col_index = col + res_idx_n);

                                if self.bounds_check_required {
                                    let within_output = scope.create_local(Elem::Bool);
                                    let within_output_tmp = scope.create_local(Elem::Bool);
                                    gpu!(scope, within_output = row_index < dim_m);
                                    gpu!(scope, within_output_tmp = col_index < dim_n);
                                    gpu!(scope, within_output = within_output && within_output_tmp);
                                    gpu!(scope, if(within_output).then(|scope|{
                                        self.write_inner(
                                            scope,
                                            res_idx_m,
                                            res_idx_n,
                                            row_index,
                                            col_index,
                                            out_stride_row,
                                            out_stride_col,
                                            results,
                                            offset_output,
                                            out,
                                        );
                                    }));
                                } else {
                                    self.write_inner(
                                        scope,
                                        res_idx_m,
                                        res_idx_n,
                                        row_index,
                                        col_index,
                                        out_stride_row,
                                        out_stride_col,
                                        results,
                                        offset_output,
                                        out,
                                    )
                                }
                            }
                        )
                    );
                }
            )
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn write_inner(
        &self,
        scope: &mut Scope,
        res_idx_m: Variable,
        res_idx_n: Variable,
        row_index: Variable,
        col_index: Variable,
        out_stride_row: Variable,
        out_stride_col: Variable,
        results: Variable,
        offset_output: Variable,
        out: Variable,
    ) {
        let elem = results.item().elem();
        let results_position = scope.create_local(Elem::UInt);
        gpu!(
            scope,
            results_position = res_idx_m * self.config.tile_size_n
        );
        gpu!(scope, results_position += res_idx_n);

        let result = scope.create_local(elem);
        gpu!(scope, result = results[results_position]);

        let output_position = scope.create_local(Elem::UInt);
        gpu!(scope, row_index *= out_stride_row);
        gpu!(scope, col_index *= out_stride_col);
        gpu!(scope, output_position = row_index + col_index);
        gpu!(scope, output_position += offset_output);

        gpu!(scope, out[output_position] = result);
    }
}
