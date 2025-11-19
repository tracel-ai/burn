use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::tile::TileNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(self.inputs.first().unwrap(), node_position);
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Extract static repeats from the enum wrapper
        let repeats_vec = match &self.config.repeats {
            onnx_ir::tile::TileInput::Static(repeats) => repeats,
            onnx_ir::tile::TileInput::Runtime(_) => {
                panic!("Runtime repeats are not supported in burn-import")
            }
        };
        let repeats = repeats_vec.iter().map(|r| r.to_tokens());

        quote! {
            let #output = #input.repeat(&[#(#repeats),*]);
        }
    }
}
