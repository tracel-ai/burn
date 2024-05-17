use crate::{
    branch::range,
    codegen::dialect::{macros::cpa, Scope, Variable, Vectorization},
};
use serde::{Deserialize, Serialize};

/// Assign value to a variable based on a given condition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct ConditionalAssign {
    pub cond: Variable,
    pub lhs: Variable,
    pub rhs: Variable,
    pub out: Variable,
}

impl ConditionalAssign {
    #[allow(missing_docs)]
    pub fn expand(self, scope: &mut Scope) {
        let cond = self.cond;
        let lhs = self.lhs;
        let rhs = self.rhs;
        let out = self.out;

        let index_var =
            |scope: &mut Scope, var: Variable, index: usize| match var.item().vectorization {
                Vectorization::Scalar => var,
                Vectorization::Vectorized(_) => {
                    let out = scope.create_local(var.item().elem());
                    cpa!(scope, out = var[index]);
                    out
                }
            };

        let mut assign_index = |index: usize| {
            let cond = index_var(scope, cond, index);

            cpa!(scope, if (cond).then(|scope| {
                let lhs = index_var(scope, lhs, index);
                let index: Variable = index.into();
                cpa!(scope, out[index] = lhs);
            }).else(|scope| {
                let rhs = index_var(scope, rhs, index);
                let index: Variable = index.into();
                cpa!(scope, out[index] = rhs);
            }));
        };

        match out.item().vectorization {
            Vectorization::Scalar => {
                cpa!(scope, if (cond).then(|scope| {
                    cpa!(scope, out = lhs);
                }).else(|scope| {
                    cpa!(scope, out = rhs);
                }));
            }
            Vectorization::Vectorized(v) => {
                for i in range(0u32, v as u32, true) {
                    assign_index(i);
                }
            }
        };
    }

    pub(crate) fn vectorize(&self, vectorization: Vectorization) -> Self {
        Self {
            cond: self.cond.vectorize(vectorization),
            lhs: self.lhs.vectorize(vectorization),
            rhs: self.rhs.vectorize(vectorization),
            out: self.out.vectorize(vectorization),
        }
    }
}
