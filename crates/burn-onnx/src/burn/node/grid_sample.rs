use super::prelude::*;
use onnx_ir::node::grid_sample::{GridSampleMode, GridSampleNode, GridSamplePaddingMode};

impl NodeCodegen for GridSampleNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(&self.inputs[0]);
        let grid = scope.arg(&self.inputs[1]);
        let output = arg_to_ident(&self.outputs[0]);

        // Map ONNX mode to Burn's InterpolateMode
        let mode = match self.config.mode {
            GridSampleMode::Bilinear => {
                quote! { burn::tensor::ops::InterpolateMode::Bilinear }
            }
            GridSampleMode::Nearest => {
                quote! { burn::tensor::ops::InterpolateMode::Nearest }
            }
            GridSampleMode::Bicubic => {
                quote! { burn::tensor::ops::InterpolateMode::Bicubic }
            }
        };

        // Map ONNX padding mode to Burn's GridSamplePaddingMode
        let padding_mode = match self.config.padding_mode {
            GridSamplePaddingMode::Zeros => {
                quote! { burn::tensor::ops::GridSamplePaddingMode::Zeros }
            }
            GridSamplePaddingMode::Border => {
                quote! { burn::tensor::ops::GridSamplePaddingMode::Border }
            }
            GridSamplePaddingMode::Reflection => {
                quote! { burn::tensor::ops::GridSamplePaddingMode::Reflection }
            }
        };

        let align_corners = self.config.align_corners;

        quote! {
            let #output = #input.grid_sample_2d(
                #grid,
                burn::tensor::ops::GridSampleOptions::new(#mode)
                    .with_padding_mode(#padding_mode)
                    .with_align_corners(#align_corners)
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::grid_sample::{GridSampleConfig, GridSampleNodeBuilder};

    use super::*;

    fn create_grid_sample_node(
        mode: GridSampleMode,
        padding_mode: GridSamplePaddingMode,
        align_corners: bool,
    ) -> GridSampleNode {
        let config = GridSampleConfig::new(mode, padding_mode, align_corners);

        GridSampleNodeBuilder::new("grid_sample1")
            .input_tensor("input", 4, DType::F32)
            .input_tensor("grid", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_grid_sample_bilinear_zeros() {
        let node = create_grid_sample_node(
            GridSampleMode::Bilinear,
            GridSamplePaddingMode::Zeros,
            false,
        );
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>, grid: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = input
                .grid_sample_2d(
                    grid,
                    burn::tensor::ops::GridSampleOptions::new(
                            burn::tensor::ops::InterpolateMode::Bilinear,
                        )
                        .with_padding_mode(burn::tensor::ops::GridSamplePaddingMode::Zeros)
                        .with_align_corners(false),
                );
            output
        }
        ");
    }

    #[test]
    fn test_grid_sample_bilinear_border_align_corners() {
        let node = create_grid_sample_node(
            GridSampleMode::Bilinear,
            GridSamplePaddingMode::Border,
            true,
        );
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>, grid: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = input
                .grid_sample_2d(
                    grid,
                    burn::tensor::ops::GridSampleOptions::new(
                            burn::tensor::ops::InterpolateMode::Bilinear,
                        )
                        .with_padding_mode(burn::tensor::ops::GridSamplePaddingMode::Border)
                        .with_align_corners(true),
                );
            output
        }
        ");
    }

    #[test]
    fn test_grid_sample_nearest() {
        let node =
            create_grid_sample_node(GridSampleMode::Nearest, GridSamplePaddingMode::Zeros, false);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>, grid: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = input
                .grid_sample_2d(
                    grid,
                    burn::tensor::ops::GridSampleOptions::new(
                            burn::tensor::ops::InterpolateMode::Nearest,
                        )
                        .with_padding_mode(burn::tensor::ops::GridSamplePaddingMode::Zeros)
                        .with_align_corners(false),
                );
            output
        }
        ");
    }

    #[test]
    fn test_grid_sample_bicubic() {
        let node =
            create_grid_sample_node(GridSampleMode::Bicubic, GridSamplePaddingMode::Zeros, false);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>, grid: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = input
                .grid_sample_2d(
                    grid,
                    burn::tensor::ops::GridSampleOptions::new(
                            burn::tensor::ops::InterpolateMode::Bicubic,
                        )
                        .with_padding_mode(burn::tensor::ops::GridSamplePaddingMode::Zeros)
                        .with_align_corners(false),
                );
            output
        }
        ");
    }

    #[test]
    fn test_grid_sample_reflection() {
        let node = create_grid_sample_node(
            GridSampleMode::Bilinear,
            GridSamplePaddingMode::Reflection,
            false,
        );
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>, grid: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = input
                .grid_sample_2d(
                    grid,
                    burn::tensor::ops::GridSampleOptions::new(
                            burn::tensor::ops::InterpolateMode::Bilinear,
                        )
                        .with_padding_mode(burn::tensor::ops::GridSamplePaddingMode::Reflection)
                        .with_align_corners(false),
                );
            output
        }
        ");
    }
}
