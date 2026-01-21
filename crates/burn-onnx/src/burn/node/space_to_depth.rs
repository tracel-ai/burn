use super::prelude::*;

impl NodeCodegen for onnx_ir::space_to_depth::SpaceToDepthNode {
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

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::space_to_depth::{SpaceToDepthConfig, SpaceToDepthNodeBuilder};

    #[test]
    fn test_space_to_depth() {
        let config = SpaceToDepthConfig::new(2);
        let node = SpaceToDepthNodeBuilder::new("s2d1")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = {
                let [b, c, h, w] = input.shape().dims();
                input
                    .reshape([b, c, h / 2usize, 2usize, w / 2usize, 2usize])
                    .permute([0, 3, 5, 1, 2, 4])
                    .reshape([b, c * 2usize * 2usize, h / 2usize, w / 2usize])
            };
            output
        }
        ");
    }
}
