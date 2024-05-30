use crate::ir::{macros::cpa, Elem, Item, Scope, Variable, Vectorization};
use serde::{Deserialize, Serialize};

/// Perform a check bound on the index (lhs) of value (rhs)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct CheckedIndex {
    pub lhs: Variable,
    pub rhs: Variable,
    pub out: Variable,
}

impl CheckedIndex {
    #[allow(missing_docs)]
    pub fn expand(self, scope: &mut Scope) {
        let lhs = self.lhs;
        let rhs = self.rhs;
        let out = self.out;
        let array_len = scope.create_local(Item::new(Elem::UInt));
        let inside_bound = scope.create_local(Item::new(Elem::Bool));

        cpa!(scope, array_len = len(lhs));
        cpa!(scope, inside_bound = rhs < array_len);

        cpa!(scope, if(inside_bound).then(|scope| {
            cpa!(scope, out = unchecked(lhs[rhs]));
        }).else(|scope| {
            cpa!(scope, out = cast(0));
        }));
    }

    pub(crate) fn vectorize(&self, vectorization: Vectorization) -> Self {
        Self {
            lhs: self.lhs.vectorize(vectorization),
            rhs: self.rhs.vectorize(vectorization),
            out: self.out.vectorize(vectorization),
        }
    }
}

/// Perform a check bound on the index (lhs) of output before assigning the value (rhs)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct CheckedIndexAssign {
    pub lhs: Variable,
    pub rhs: Variable,
    pub out: Variable,
}

impl CheckedIndexAssign {
    #[allow(missing_docs)]
    pub fn expand(self, scope: &mut Scope) {
        let lhs = self.lhs;
        let rhs = self.rhs;
        let out = self.out;
        let array_len = scope.create_local(Item::new(Elem::UInt));
        let inside_bound = scope.create_local(Item::new(Elem::Bool));

        cpa!(scope, array_len = len(out));
        cpa!(scope, inside_bound = lhs < array_len);

        cpa!(scope, if(inside_bound).then(|scope| {
            cpa!(scope, unchecked(out[lhs]) = rhs);
        }));
    }

    pub(crate) fn vectorize(&self, vectorization: Vectorization) -> Self {
        Self {
            lhs: self.lhs.vectorize(vectorization),
            rhs: self.rhs.vectorize(vectorization),
            out: self.out.vectorize(vectorization),
        }
    }
}
