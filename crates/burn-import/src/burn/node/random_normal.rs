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
