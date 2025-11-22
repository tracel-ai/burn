use super::prelude::*;
use onnx_ir::depth_to_space::DepthToSpaceMode;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::depth_to_space::DepthToSpaceNode {
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

        let output_expr = match self.config.mode {
            DepthToSpaceMode::Dcr => {
                quote! {
                    let [b, c, h, w] = #input.shape().dims();
                    #input
                        .reshape([b, #block_size, #block_size, c / (#block_size * #block_size), h, w])
                        .permute([0, 3, 4, 1, 5, 2])
                        .reshape([b, c / (#block_size * #block_size), h * #block_size, w * #block_size])
                }
            }
            DepthToSpaceMode::Crd => {
                quote! {
                    let [b, c, h, w] = #input.shape().dims();
                    #input
                        .reshape([b, c / (#block_size * #block_size), #block_size, #block_size, h, w])
                        .permute([0, 1, 4, 2, 5, 3])
                        .reshape([b, c / (#block_size * #block_size), h * #block_size, w * #block_size])
                }
            }
        };
        quote! {
            let #output = {
                #output_expr
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::depth_to_space::{DepthToSpaceConfig, DepthToSpaceMode, DepthToSpaceNodeBuilder};

    #[test]
    fn test_depth_to_space_dcr() {
        let config = DepthToSpaceConfig {
            block_size: 2,
            mode: DepthToSpaceMode::Dcr,
        };
        let node = DepthToSpaceNodeBuilder::new("d2s1")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        let output = {
                let [b, c, h, w] = input.shape().dims();
                input
                    .reshape([b, 2usize, 2usize, c / (2usize * 2usize), h, w])
                    .permute([0, 3, 4, 1, 5, 2])
                    .reshape([b, c / (2usize * 2usize), h * 2usize, w * 2usize])
            };
        ");
    }

    #[test]
    fn test_depth_to_space_crd() {
        let config = DepthToSpaceConfig {
            block_size: 2,
            mode: DepthToSpaceMode::Crd,
        };
        let node = DepthToSpaceNodeBuilder::new("d2s2")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        let output = {
                let [b, c, h, w] = input.shape().dims();
                input
                    .reshape([b, c / (2usize * 2usize), 2usize, 2usize, h, w])
                    .permute([0, 1, 4, 2, 5, 3])
                    .reshape([b, c / (2usize * 2usize), h * 2usize, w * 2usize])
            };
        ");
    }
}
