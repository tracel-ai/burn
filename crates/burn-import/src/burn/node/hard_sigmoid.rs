use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::hard_sigmoid::HardSigmoidNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(self.inputs.first().unwrap(), node_position);
        let output = arg_to_ident(self.outputs.first().unwrap());
        let alpha = self.config.alpha.to_tokens();
        let beta = self.config.beta.to_tokens();

        quote! {
            let #output = burn::tensor::activation::hard_sigmoid(#input, #alpha, #beta);
        }
    }

    // No need to register imports since we use the fully qualified path
}
