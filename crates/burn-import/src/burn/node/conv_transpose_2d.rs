use super::prelude::*;
use burn_store::TensorSnapshot;

impl NodeCodegen for onnx_ir::node::conv_transpose2d::ConvTranspose2dNode {
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
        let padding_out = self.config.padding_out.to_tokens();
        let bias = self.config.bias;

        Some(Field::new(
            self.name.clone(),
            quote! {
                ConvTranspose2d<B>
            },
            quote! {
                let #name = ConvTranspose2dConfig::new(#channels, #kernel_size)
                    .with_stride(#stride)
                    .with_padding(#padding)
                    .with_padding_out(#padding_out)
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
        imports.register("burn::nn::conv::ConvTranspose2d");
        imports.register("burn::nn::conv::ConvTranspose2dConfig");
    }

    fn collect_snapshots(&self, field_name: &str) -> Vec<TensorSnapshot> {
        use crate::burn::node_traits::create_lazy_snapshot;

        let mut snapshots = vec![];

        // Weight tensor (input index 1)
        // ONNX ConvTranspose weight: [in_channels, out_channels/groups, kH, kW]
        // Burn ConvTranspose2d weight: [channels_in, channels_out/groups, kH, kW]
        // These layouts match! No transformation needed.
        if let Some(weight_input) = self.inputs.get(1) {
            let weight_path = format!("{}.weight", field_name);
            if let Some(snapshot) =
                create_lazy_snapshot(weight_input, &weight_path, "ConvTranspose2d")
            {
                snapshots.push(snapshot);
            }
        }

        // Bias tensor (input index 2, optional)
        if self.config.bias
            && let Some(bias_input) = self.inputs.get(2)
        {
            let bias_path = format!("{}.bias", field_name);
            if let Some(snapshot) = create_lazy_snapshot(bias_input, &bias_path, "ConvTranspose2d")
            {
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
    use onnx_ir::node::conv_transpose2d::{
        ConvTranspose2dConfig, ConvTranspose2dNode, ConvTranspose2dNodeBuilder,
    };

    fn create_conv_transpose_2d_node(name: &str) -> ConvTranspose2dNode {
        let config =
            ConvTranspose2dConfig::new([3, 64], [3, 3], [1, 1], [1, 1], [1, 1], [0, 0], 1, true);

        ConvTranspose2dNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_conv_transpose_2d_forward() {
        let node = create_conv_transpose_2d_node("conv_transpose1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = self.conv_transpose1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_conv_transpose_2d_forward_with_clone() {
        let node = create_conv_transpose_2d_node("conv_transpose1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = self.conv_transpose1.forward(input.clone());
            output
        }
        ");
    }
}
