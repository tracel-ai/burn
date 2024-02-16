use crate::codegen::dialect::gpu::{
    algo::read::OffsetGlobalWithLayoutAlgo, macros::gpu, Elem, MatmulAlgo, Scope, Variable,
};

impl MatmulAlgo {
    pub fn expand(self, scope: &mut Scope) {
        match self {
            MatmulAlgo::MemCoalescing {
                variables,
                block_size,
            } => {
                let block_size: Variable = block_size.into();
                let local_idx = Variable::InvocationIndex;
                let tmp_index = scope.create_local(Elem::UInt);
                let row = scope.create_local(Elem::UInt);
                let col = scope.create_local(Elem::UInt);
                let lhs = variables.lhs;
                let rhs = variables.rhs;
                let out = variables.out;
                let n_rows = scope.create_local(Elem::UInt);
                let n_cols = scope.create_local(Elem::UInt);
                let k = scope.create_local(Elem::UInt);

                gpu!(scope, tmp_index = local_idx / block_size);
                gpu!(scope, row = block_size * Variable::WorkgroupIdX);
                gpu!(scope, row = row + tmp_index);

                gpu!(scope, tmp_index = local_idx % block_size);
                gpu!(scope, col = block_size * Variable::WorkgroupIdY);
                gpu!(scope, col = col + tmp_index);

                gpu!(scope, tmp_index = sub(Variable::Rank, 2u32));
                gpu!(scope, n_rows = shape(out, tmp_index));

                gpu!(scope, k = shape(rhs, tmp_index));

                gpu!(scope, tmp_index = sub(Variable::Rank, 1u32));
                gpu!(scope, n_cols = shape(out, tmp_index));

                let offset_lhs = scope.create_local(Elem::UInt);
                let offset_rhs = scope.create_local(Elem::UInt);
                let offset_output = scope.create_local(Elem::UInt);

                let n_batches: Variable = 0u32.into();
                gpu!(scope, offset_output = n_rows * n_cols);
                gpu!(scope, offset_output = offset_output * n_batches);

                OffsetGlobalWithLayoutAlgo {
                    global: lhs,
                    layout: out,
                    offset: offset_lhs,
                }
                .expand(scope);
                OffsetGlobalWithLayoutAlgo {
                    global: rhs,
                    layout: out,
                    offset: offset_rhs,
                }
                .expand(scope);

                let sum = scope.create_local(out.item());
                let zero: Variable = 0u32.into();
                gpu!(scope, sum = zero);

                gpu!(
                    scope,
                    range(0u32, k).for_each(|i, scope| {
                        let lhs_index = scope.create_local(Elem::UInt);
                        let rhs_index = scope.create_local(Elem::UInt);

                        let lhs_value = scope.create_local(lhs.item());
                        let rhs_value = scope.create_local(rhs.item());
                        let out_value = scope.create_local(rhs.item());

                        gpu!(scope, lhs_index = row * k);
                        gpu!(scope, lhs_index = lhs_index + i);
                        gpu!(scope, lhs_index = lhs_index + offset_lhs);

                        gpu!(scope, rhs_index = i * n_cols);
                        gpu!(scope, rhs_index = rhs_index + col);
                        gpu!(scope, rhs_index = rhs_index + offset_rhs);

                        gpu!(scope, lhs_value = lhs[lhs_index]);
                        gpu!(scope, rhs_value = rhs[rhs_index]);

                        gpu!(scope, out_value = lhs_value * rhs_value);
                        gpu!(scope, sum = sum + rhs_value);
                    })
                );

                let out_index = scope.create_local(Elem::UInt);

                gpu!(scope, out_index = row * n_cols);
                gpu!(scope, out_index = out_index + col);
                gpu!(scope, out_index = out_index + offset_output);
                gpu!(scope, out[out_index] = sum);
            }
        }
    }
}
