use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::random::RandomNormalNode {
    fn inputs(&self) -> &[Argument] {
        // RandomNormal has no inputs - it generates a tensor from scratch
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

        // Build distribution with mean and scale (standard deviation)
        let mean = self.config.mean;
        let std_deviation = self.config.scale;
        let dist = quote! { Distribution::Normal(#mean, #std_deviation) };

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
    use onnx_ir::node::random::{RandomNormalConfig, RandomNormalNodeBuilder};

    #[test]
    fn test_random_normal() {
        let config = RandomNormalConfig {
            mean: 0.0,
            scale: 1.0,
            shape: vec![2, 3, 4],
        };
        let node = RandomNormalNodeBuilder::new("rand1")
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> Tensor<B, 3> {
            let output = Tensor::random(
                Shape::new([2usize, 3usize, 4usize]),
                Distribution::Normal(0f64, 1f64),
                &*self.device,
            );
            output
        }
        ");
    }
}
