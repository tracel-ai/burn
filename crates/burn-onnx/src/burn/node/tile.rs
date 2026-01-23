use super::prelude::*;

impl NodeCodegen for onnx_ir::tile::TileNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Extract static repeats from the enum wrapper
        let repeats_vec = match &self.config.repeats {
            onnx_ir::tile::TileInput::Static(repeats) => repeats,
            onnx_ir::tile::TileInput::Runtime(_) => {
                panic!("Runtime repeats are not supported in burn-onnx")
            }
        };
        let repeats = repeats_vec.iter().map(|r| r.to_tokens());

        quote! {
            let #output = #input.repeat(&[#(#repeats),*]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::tile::{TileConfig, TileInput, TileNode, TileNodeBuilder};

    fn create_tile_node(name: &str, repeats: Vec<usize>) -> TileNode {
        let config = TileConfig {
            repeats: TileInput::Static(repeats),
        };

        TileNodeBuilder::new(name)
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_tile_simple() {
        let node = create_tile_node("tile1", vec![2, 3]);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input.repeat(&[2, 3]);
            output
        }
        ");
    }

    #[test]
    fn test_tile_single_repeat() {
        let node = create_tile_node("tile1", vec![1, 2, 3]);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input.repeat(&[1, 2, 3]);
            output
        }
        ");
    }
}
