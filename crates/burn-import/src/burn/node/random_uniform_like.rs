use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::random_like::RandomUniformLikeNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Build distribution with low and high bounds
        let low = self.config.low;
        let high = self.config.high;
        let dist = quote! { Distribution::Uniform(#low, #high) };

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
    use onnx_ir::node::random_like::{RandomUniformLikeConfig, RandomUniformLikeNodeBuilder};

    #[test]
    fn test_random_uniform_like() {
        let config = RandomUniformLikeConfig {
            low: 0.0,
            high: 1.0,
        };
        let node = RandomUniformLikeNodeBuilder::new("randl1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = input.random_like(Distribution::Uniform(0f64, 1f64));");
    }
}
