use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::sum::SumNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let inputs = self
            .inputs
            .iter()
            .map(|arg| scope.tensor_use_owned(arg, node_position));

        let output = arg_to_ident(self.outputs.first().unwrap());

        quote! {
            let #output = #(#inputs)+*;
        }
    }
}
