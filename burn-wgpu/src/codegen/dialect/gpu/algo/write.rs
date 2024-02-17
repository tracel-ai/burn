use crate::codegen::dialect::gpu::{macros::gpu, Scope, Variable, WriteGlobalAlgo};

impl WriteGlobalAlgo {
    pub fn expand(self, scope: &mut Scope) {
        let output = self.global;
        let intput = self.input;
        let position = Variable::Id;

        gpu!(scope, output[position] = intput);
    }
}
