use crate::codegen::dialect::gpu::{gpu, Elem, Scope, Variable, Vectorization};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gather {
    pub tensor: Variable,
    pub indices: Variable,
    pub out: Variable,
    pub dim: usize,
}

impl Gather {
    pub fn expand(self, scope: &mut Scope) {
        match self.tensor {
            Variable::GlobalInputArray(_, _) => (),
            Variable::GlobalOutputArray(_, _) => (),
            _ => panic!("Tensor variable must be an global array."),
        };

        let tensor = self.tensor;
        let output = self.out;

        let stride = scope.create_local(Elem::UInt);
        let index = scope.create_local(Elem::UInt);

        // int to uint since we use it as index.
        gpu!(scope, index = cast(self.indices));
        gpu!(scope, stride = stride(tensor, self.dim));
        gpu!(scope, index = index * stride);

        if self.dim > 0 {
            let index_prev_dim =
                scope.index_offset_with_output_layout(tensor, 0u32.into(), self.dim.into());
            gpu!(scope, index += index_prev_dim);
        }

        let index_after_dim =
            scope.index_offset_with_output_layout(tensor, (self.dim + 1).into(), Variable::Rank);

        gpu!(scope, index += index_after_dim);
        gpu!(scope, output = tensor[index]);
    }

    pub(crate) fn vectorize(&self, vectorization: Vectorization) -> Self {
        match vectorization {
            Vectorization::Scalar => (),
            _ => panic!("Vectorization isn't supported yet"),
        };
        Self {
            tensor: self.tensor.vectorize(vectorization),
            indices: self.indices.vectorize(vectorization),
            out: self.out.vectorize(vectorization),
            dim: self.dim,
        }
    }
}
