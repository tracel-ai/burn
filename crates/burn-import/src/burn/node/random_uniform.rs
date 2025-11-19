use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::random::RandomUniformNode {
    fn inputs(&self) -> &[Argument] {
        // RandomUniform has no inputs - it generates a tensor from scratch
        &[]
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, _scope: &mut Scope, _node_position: usize) -> TokenStream {
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
