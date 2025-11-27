use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::random_like::RandomNormalLikeNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Build distribution with mean and scale (standard deviation)
        let mean = self.config.mean;
        let std_deviation = self.config.scale;
        let dist = quote! { Distribution::Normal(#mean, #std_deviation) };

        quote! {
            let #output = #input.random_like(#dist);
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::tensor::Distribution");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::random_like::{RandomNormalLikeConfig, RandomNormalLikeNodeBuilder};

    #[test]
    fn test_random_normal_like() {
        let config = RandomNormalLikeConfig {
            mean: 0.0,
            scale: 1.0,
        };
        let node = RandomNormalLikeNodeBuilder::new("randl1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input.random_like(Distribution::Normal(0f64, 1f64));
            output
        }
        ");
    }
}
