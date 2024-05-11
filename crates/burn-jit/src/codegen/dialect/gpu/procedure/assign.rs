use crate::codegen::dialect::gpu::{macros::gpu, Item, Scope, Variable, Vectorization};
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

        let index_var = |scope: &mut Scope, var: Variable, index: usize| match var.item() {
            Item::Scalar(_) => var,
            _ => {
                let out = scope.create_local(var.item().elem());
                gpu!(scope, out = var[index]);
                out
            }
        };

        let mut assign_index = |index: usize| {
            let cond = index_var(scope, cond, index);

            gpu!(scope, if (cond).then(|scope| {
                let lhs = index_var(scope, lhs, index);
                let index: Variable = index.into();
                gpu!(scope, out[index] = lhs);
            }).else(|scope| {
                let rhs = index_var(scope, rhs, index);
                let index: Variable = index.into();
                gpu!(scope, out[index] = rhs);
            }));
        };

        match out.item() {
            Item::Vec4(_) => {
                assign_index(0);
                assign_index(1);
                assign_index(2);
                assign_index(3);
            }
            Item::Vec3(_) => {
                assign_index(0);
                assign_index(1);
                assign_index(2);
            }
            Item::Vec2(_) => {
                assign_index(0);
                assign_index(1);
            }
            Item::Scalar(_) => {
                gpu!(scope, if (cond).then(|scope| {
                    gpu!(scope, out = lhs);
                }).else(|scope| {
                    gpu!(scope, out = rhs);
                }));
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
