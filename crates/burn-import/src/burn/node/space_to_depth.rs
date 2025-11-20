use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::space_to_depth::SpaceToDepthNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let block_size = self.config.block_size;

        quote! {
            let #output = {
                let [b, c, h, w] = #input.shape().dims();
                #input
                    .reshape([b, c, h / #block_size, #block_size, w / #block_size, #block_size])
                    .permute([0, 3, 5, 1, 2, 4])
                    .reshape([b, c * #block_size * #block_size, h / #block_size, w / #block_size])
            };
        }
    }
}
