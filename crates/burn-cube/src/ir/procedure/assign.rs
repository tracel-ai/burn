use crate::ir::{macros::cpa, Scope, Variable, Vectorization};
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
            |scope: &mut Scope, var: Variable, index: usize| match var.item().vectorization == 1 {
                true => var,
                false => {
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

        let vectorization = out.item().vectorization;
        match vectorization == 1 {
            true => {
                cpa!(scope, if (cond).then(|scope| {
                    cpa!(scope, out = lhs);
                }).else(|scope| {
                    cpa!(scope, out = rhs);
                }));
            }
            false => {
                for i in 0..vectorization {
                    assign_index(i as usize);
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
