use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::sum::SumNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let inputs = self.inputs.iter().map(|arg| scope.arg(arg));

        let output = arg_to_ident(self.outputs.first().unwrap());

        quote! {
            let #output = #(#inputs)+*;
        }
    }
}
