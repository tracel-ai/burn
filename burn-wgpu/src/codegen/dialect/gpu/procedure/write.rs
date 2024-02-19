use crate::codegen::dialect::gpu::{macros::gpu, Scope, Variable};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteGlobal {
    pub input: Variable,
    pub global: Variable,
}

impl WriteGlobal {
    pub fn expand(self, scope: &mut Scope) {
        let output = self.global;
        let intput = self.input;
        let position = Variable::Id;

        gpu!(scope, output[position] = intput);
    }
}
