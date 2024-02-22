use crate::codegen::dialect::gpu::{
    macros::gpu, procedure::read::IndexOffsetGlobalWithLayout, BinaryOperator, Branch, Elem, Scope,
    Variable, Vectorization,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Matmul {
    MemCoalescing {
        variables: BinaryOperator,
        block_size: usize,
    },
}

impl Matmul {
    pub fn expand(self, scope: &mut Scope) {
        match self {
            Matmul::MemCoalescing {
                variables,
                block_size,
            } => {
                // Define out global variables.
                let local_idx = Variable::InvocationIndex;
                let batch = Variable::GlobalInvocationIdZ;
                let rank = Variable::Rank;
                let block_size: Variable = block_size.into();

                // Extract tensor variables.
                let lhs = variables.lhs;
                let rhs = variables.rhs;
                let out = variables.out;

                // Define where we have to work on the current matrix.
                let tmp_index = scope.create_local(Elem::UInt);
                let batch_dims = scope.create_local(Elem::UInt);
                let row = scope.create_local(Elem::UInt);
                let col = scope.create_local(Elem::UInt);

                // Row position.
                gpu!(scope, tmp_index = local_idx / block_size);
                gpu!(scope, row = block_size * Variable::WorkgroupIdX);
                gpu!(scope, row = row + tmp_index);

                // Col position.
                gpu!(scope, tmp_index = local_idx % block_size);
                gpu!(scope, col = block_size * Variable::WorkgroupIdY);
                gpu!(scope, col = col + tmp_index);

                // Batch position.
                gpu!(scope, batch_dims = rank - 2u32);

                // Define the matrix size.
                let n_rows = scope.create_local(Elem::UInt);
                let n_cols = scope.create_local(Elem::UInt);
                let k = scope.create_local(Elem::UInt);

                // Number of rows.
                gpu!(scope, n_rows = shape(out, batch_dims));

                // Number of cols.
                gpu!(scope, tmp_index = batch_dims + 1u32);
                gpu!(scope, n_cols = shape(out, tmp_index));

                // The dimension that is going to be squashed.
                gpu!(scope, k = shape(lhs, tmp_index));

                // Check if there is some work to be done.
                let should_stop = scope.create_local(Elem::Bool);
                gpu!(scope, should_stop = row >= n_rows);
                gpu!(scope, if (should_stop).then(|scope| {
                    scope.register(Branch::Return);
                }));

                gpu!(scope, should_stop = col >= n_cols);
                gpu!(scope, if (should_stop).then(|scope| {
                    scope.register(Branch::Return);
                }));

                // Calculate the batch offset.
                let offset_lhs = scope.create_local(Elem::UInt);
                let offset_rhs = scope.create_local(Elem::UInt);
                let offset_output = scope.create_local(Elem::UInt);

                // Batch offset for the output.
                gpu!(scope, offset_output = n_rows * n_cols);
                gpu!(scope, offset_output = offset_output * batch);

                // Batch offset for the lhs & rhs matrices.
                IndexOffsetGlobalWithLayout {
                    tensors: vec![lhs, rhs],
                    indexes: vec![offset_lhs, offset_rhs],
                    layout: out,
                    index_ref: offset_output,
                    dim_start: 0u32.into(),
                    dim_end: batch_dims,
                }
                .expand(scope);

                // Calculate the dot product (row X col).
                let sum = scope.create_local(out.item());

                // Initialize the sum to zero.
                let zero: Variable = 0f32.into();
                gpu!(scope, sum = zero);

                // Loop over the k dimension.
                gpu!(
                    scope,
                    range(0u32, k).for_each(|i, scope| {
                        let lhs_index = scope.create_local(Elem::UInt);
                        let rhs_index = scope.create_local(Elem::UInt);

                        let lhs_value = scope.create_local(lhs.item());
                        let rhs_value = scope.create_local(rhs.item());
                        let out_value = scope.create_local(out.item());

                        gpu!(scope, lhs_index = row * k);
                        gpu!(scope, lhs_index = lhs_index + i);
                        gpu!(scope, lhs_index = lhs_index + offset_lhs);

                        gpu!(scope, rhs_index = i * n_cols);
                        gpu!(scope, rhs_index = rhs_index + col);
                        gpu!(scope, rhs_index = rhs_index + offset_rhs);

                        gpu!(scope, lhs_value = lhs[lhs_index]);
                        gpu!(scope, rhs_value = rhs[rhs_index]);

                        gpu!(scope, out_value = lhs_value * rhs_value);
                        gpu!(scope, sum += out_value);
                    })
                );

                let out_index = scope.create_local(Elem::UInt);

                gpu!(scope, out_index = row * n_cols);
                gpu!(scope, out_index += col);
                gpu!(scope, out_index += offset_output);
                gpu!(scope, out[out_index] = sum);
            }
        }
    }

    pub(crate) fn vectorize(&self, vectorization: Vectorization) -> Self {
        match self {
            Matmul::MemCoalescing {
                variables: _,
                block_size: _,
            } => match vectorization {
                Vectorization::Scalar => self.clone(),
                _ => panic!("MemCoalescing can't be vectorized with {vectorization:?}."),
            },
        }
    }
}
