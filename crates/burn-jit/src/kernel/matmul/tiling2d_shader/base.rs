use crate::gpu::{gpu, BinaryOperator, Scope, Synchronization, Variable};

use crate::kernel::matmul::tiling2d_shader::{
    computation_loop, gather_shader_information, load_shared_memory, write_to_output,
};
use crate::kernel::matmul::Tiling2dConfig;

pub(crate) struct MatmulTiling2dShader {
    pub variables: BinaryOperator,
    pub config: Tiling2dConfig,
    pub bounds_check_required: bool,
    pub unroll: bool,
}

pub(crate) struct Tiling2dState {
    pub n_loops: Variable,
    pub k: Variable,
    pub lhs: Variable,
    pub rhs: Variable,
    pub out: Variable,
    pub offset_lhs: Variable,
    pub offset_rhs: Variable,
    pub offset_output: Variable,
    pub row: Variable,
    pub col: Variable,
    pub dim_m: Variable,
    pub dim_k: Variable,
    pub dim_n: Variable,
    pub thread_col: Variable,
    pub thread_row: Variable,
    pub shared_lhs: Variable,
    pub shared_rhs: Variable,
    pub register_m: Variable,
    pub register_n: Variable,
    pub results: Variable,
    pub lhs_stride_col: Variable,
    pub lhs_stride_row: Variable,
    pub rhs_stride_col: Variable,
    pub rhs_stride_row: Variable,
    pub out_stride_row: Variable,
    pub out_stride_col: Variable,
}

impl MatmulTiling2dShader {
    pub(crate) fn expand(self, scope: &mut Scope) {
        let shader_state = gather_shader_information(scope, &self);

        let block_size_k: Variable = self.config.block_size_k.into();
        gpu!(
            scope,
            range(0u32, shader_state.n_loops).for_each(|i, scope| {
                // From 0 to K with steps block_size_k
                let k = shader_state.k;
                gpu!(scope, k = i * block_size_k);

                load_shared_memory(scope, &self, &shader_state);

                scope.register(Synchronization::WorkgroupBarrier);

                computation_loop(scope, &self, &shader_state);

                scope.register(Synchronization::WorkgroupBarrier);
            })
        );

        write_to_output(scope, &self, &shader_state);
    }
}
