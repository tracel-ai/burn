use super::prelude::*;
use burn_store::TensorSnapshot;

impl NodeCodegen for onnx_ir::conv3d::Conv3dNode {
    fn inputs(&self) -> &[Argument] {
        // Filter inputs only dynamic and constant
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let name = Ident::new(&self.name, Span::call_site());
        let channels = self.config.channels.to_tokens();
        let kernel_size = self.config.kernel_size.to_tokens();
        let stride = self.config.stride.to_tokens();
        let dilation = self.config.dilation.to_tokens();
        let groups = self.config.groups.to_tokens();
        let padding = self.config.padding.to_tokens();
        let bias = self.config.bias;

        Some(Field::new(
            self.name.clone(),
            quote! {
                Conv3d<B>
            },
            quote! {
                let #name = Conv3dConfig::new(#channels, #kernel_size)
                    .with_stride(#stride)
                    .with_padding(#padding)
                    .with_dilation(#dilation)
                    .with_groups(#groups)
                    .with_bias(#bias)
                    .init(device);
            },
        ))
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let field = Ident::new(&self.name, Span::call_site());

        quote! {
            let #output = self.#field.forward(#input);
        }
    }
    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::nn::PaddingConfig3d");
        imports.register("burn::nn::conv::Conv3d");
        imports.register("burn::nn::conv::Conv3dConfig");
    }

    fn collect_snapshots(&self, field_name: &str) -> Vec<TensorSnapshot> {
        use crate::burn::node_traits::create_lazy_snapshot;
        let mut snapshots = vec![];

        // Weight tensor (input index 1)
        if let Some(weight_input) = self.inputs.get(1) {
            let weight_path = format!("{}.weight", field_name);
            if let Some(snapshot) = create_lazy_snapshot(weight_input, &weight_path, "Conv3d") {
                snapshots.push(snapshot);
            }
        }

        // Bias tensor if present (input index 2)
        if let Some(bias_input) = self.inputs.get(2) {
            let bias_path = format!("{}.bias", field_name);
            if let Some(snapshot) = create_lazy_snapshot(bias_input, &bias_path, "Conv3d") {
                snapshots.push(snapshot);
            }
        }

        snapshots
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::conv3d::{Conv3dConfig, Conv3dNode, Conv3dNodeBuilder};
    use onnx_ir::padding::PaddingConfig3d;

    fn create_conv3d_node(name: &str) -> Conv3dNode {
        let config = Conv3dConfig::new(
            [3, 64],
            [3, 3, 3],
            [1, 1, 1],
            [1, 1, 1],
            1,
            true,
            PaddingConfig3d::Explicit(1, 1, 1),
        );

        Conv3dNodeBuilder::new(name)
            .input_tensor("input", 5, DType::F32)
            .output_tensor("output", 5, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_conv3d_forward() {
        let node = create_conv3d_node("conv1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 5>) -> Tensor<B, 5> {
            let output = self.conv1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_conv3d_forward_with_clone() {
        let node = create_conv3d_node("conv1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 5>) -> Tensor<B, 5> {
            let output = self.conv1.forward(input.clone());
            output
        }
        ");
    }
}
