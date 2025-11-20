use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::gather_elements::GatherElementsNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> proc_macro2::TokenStream {
        let dim = self.config.axis.to_tokens();
        let input = scope.arg(self.inputs.first().unwrap());
        let index = scope.arg(&self.inputs[1]);
        let output = arg_to_ident(self.outputs.first().unwrap());

        quote! {
            let #output = #input.gather(#dim, #index);
        }
    }
}
