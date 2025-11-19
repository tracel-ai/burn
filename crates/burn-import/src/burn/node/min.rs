use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::min::MinNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let lhs_arg = self.inputs.first().unwrap();
        let rhs_arg = self.inputs.get(1).unwrap();
        let output = arg_to_ident(self.outputs.first().unwrap());

        // TODO: Add support for broadcasting when tensors have different ranks
        // TODO: ONNX Min spec supports variadic inputs (2+ tensors), currently only handles 2
        // TODO: Add proper error handling for non-tensor inputs

        let lhs = match &lhs_arg.ty {
            ArgType::Tensor(_) => scope.tensor_use_owned(lhs_arg, node_position),
            _ => panic!("lhs must be a tensor"),
        };

        let rhs = match &rhs_arg.ty {
            ArgType::Tensor(_) => scope.tensor_use_owned(rhs_arg, node_position),
            _ => panic!("rhs must be a tensor"),
        };

        quote! {
            let #output = #lhs.min_pair(#rhs);
        }
    }
}
