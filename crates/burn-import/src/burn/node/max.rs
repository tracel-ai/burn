use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::max::MaxNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let lhs_arg = self.inputs.first().unwrap();
        let rhs_arg = self.inputs.get(1).unwrap();
        let output = arg_to_ident(self.outputs.first().unwrap());

        // TODO: Add support for broadcasting when tensors have different ranks
        // TODO: ONNX Max spec supports variadic inputs (2+ tensors), currently only handles 2
        // TODO: Add proper error handling for non-tensor inputs

        let lhs = scope.arg(lhs_arg);

        let rhs = scope.arg(rhs_arg);

        quote! {
            let #output = #lhs.max_pair(#rhs);
        }
    }
}
