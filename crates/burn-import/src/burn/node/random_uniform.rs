use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::random::RandomUniformNode {
    fn inputs(&self) -> &[Argument] {
        // RandomUniform has no inputs - it generates a tensor from scratch
        &[]
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, _scope: &mut super::super::scope::ScopeAtPosition<'_>) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Build shape expression
        let shape_values = self.config.shape.iter();
        let shape = quote! { Shape::new([#(#shape_values),*]) };

        // Build distribution with low and high bounds
        let low = self.config.low;
        let high = self.config.high;
        let dist = quote! { Distribution::Uniform(#low, #high) };

        quote! {
            let #output = Tensor::random(#shape, #dist, &*self.device);
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
    use onnx_ir::node::random::{RandomUniformConfig, RandomUniformNodeBuilder};

    #[test]
    fn test_random_uniform() {
        let config = RandomUniformConfig {
            low: 0.0,
            high: 1.0,
            shape: vec![3, 4],
        };
        let node = RandomUniformNodeBuilder::new("rand1")
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        let output = Tensor::random(
                Shape::new([3usize, 4usize]),
                Distribution::Uniform(0f64, 1f64),
                &*self.device,
            );
        ");
    }
}
