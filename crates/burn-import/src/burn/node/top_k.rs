use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::topk::TopKNode {
    fn inputs(&self) -> &[Argument] {
        // Filter inputs only dynamic and constant
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        // TopK has 2 outputs: values and indices
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());

        // TopK has 2 outputs
        let values_output = arg_to_ident(&self.outputs[0]);
        let indices_output = arg_to_ident(&self.outputs[1]);

        let axis = self.config.axis.to_tokens();

        // Extract static k from the enum wrapper
        let k = match &self.config.k {
            onnx_ir::topk::TopKInput::Static(k_value) => k_value.to_tokens(),
            onnx_ir::topk::TopKInput::Runtime(_) => {
                panic!("Runtime k value is not supported in burn-import")
            }
        };

        quote! {
            let (#values_output, #indices_output) = #input.topk_with_indices(#k, #axis);
        }
    }
}
