use super::prelude::*;
use onnx_ir::node::grid_sample::{GridSampleMode, GridSampleNode, GridSamplePaddingMode};

impl<PS: PrecisionSettings> NodeCodegen<PS> for GridSampleNode {
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

        // Panic for unsupported options that would produce incorrect results
        if self.config.padding_mode != GridSamplePaddingMode::Zeros {
            panic!(
                "GridSample: padding_mode {:?} is not supported by Burn",
                self.config.padding_mode
            );
        }

        if self.config.align_corners {
            panic!("GridSample: align_corners=true is not supported by Burn");
        }

        quote! {
            let #output = #input.grid_sample_2d(#grid, #mode);
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
    fn test_grid_sample_bilinear() {
        let node = create_grid_sample_node(
            GridSampleMode::Bilinear,
            GridSamplePaddingMode::Zeros,
            false,
        );
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>, grid: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = input
                .grid_sample_2d(grid, burn::tensor::ops::InterpolateMode::Bilinear);
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
            let output = input.grid_sample_2d(grid, burn::tensor::ops::InterpolateMode::Nearest);
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
            let output = input.grid_sample_2d(grid, burn::tensor::ops::InterpolateMode::Bicubic);
            output
        }
        ");
    }
}
