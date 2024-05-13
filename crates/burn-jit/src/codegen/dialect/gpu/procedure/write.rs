use crate::codegen::dialect::gpu::{macros::gpu, Scope, Variable, Vectorization};
use serde::{Deserialize, Serialize};

/// Write to a global array.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct WriteGlobal {
    pub input: Variable,
    pub global: Variable,
    pub position: Variable,
}

impl WriteGlobal {
    #[allow(missing_docs)]
    pub fn expand(self, scope: &mut Scope) {
        let output = self.global;
        let input = self.input;
        let position = self.position;

        gpu!(scope, output[position] = input);
    }

    pub(crate) fn vectorize(&self, vectorization: Vectorization) -> Self {
        Self {
            input: self.input.vectorize(vectorization),
            global: self.global.vectorize(vectorization),
            position: self.position,
        }
    }
}
