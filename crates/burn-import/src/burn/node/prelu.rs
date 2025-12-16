use super::prelude::*;
use burn_store::TensorSnapshot;

impl NodeCodegen for onnx_ir::prelu::PReluNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let name = Ident::new(&self.name, Span::call_site());
        let num_parameters = self.config.num_parameters;

        let init = if num_parameters > 1 {
            let n = num_parameters.to_tokens();
            quote! { let #name = PReluConfig::new().with_num_parameters(#n).init(device); }
        } else {
            quote! { let #name = PReluConfig::new().init(device); }
        };

        Some(Field::new(self.name.clone(), quote! { PRelu<B> }, init))
    }

    fn collect_snapshots(&self, field_name: &str) -> Vec<TensorSnapshot> {
        use crate::burn::node_traits::create_lazy_snapshot;

        let mut snapshots = vec![];

        // Alpha (slope) tensor at input index 1
        if let Some(alpha_input) = self.inputs.get(1) {
            let alpha_path = format!("{}.alpha", field_name);
            if let Some(mut snapshot) = create_lazy_snapshot(alpha_input, &alpha_path, "PRelu") {
                // Squeeze dimensions to 1D
                snapshot.shape = vec![self.config.num_parameters];
                snapshots.push(snapshot);
            }
        }

        snapshots
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
        imports.register("burn::nn::PRelu");
        imports.register("burn::nn::PReluConfig");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::prelu::{PReluConfig, PReluNodeBuilder};

    fn create_prelu_node(name: &str, num_parameters: usize) -> onnx_ir::prelu::PReluNode {
        PReluNodeBuilder::new(name)
            .input_tensor("input", 2, DType::F32)
            .input_tensor("slope", 1, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(PReluConfig::new(num_parameters))
            .build()
    }

    #[test]
    fn test_prelu_forward() {
        let node = create_prelu_node("prelu1", 64);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>, slope: Tensor<B, 1>) -> Tensor<B, 2> {
            let output = self.prelu1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_prelu_field_with_channel_slope() {
        let node = create_prelu_node("prelu1", 64);
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @"let prelu1 = PReluConfig::new().with_num_parameters(64).init(device);");
    }

    #[test]
    fn test_prelu_field_with_scalar_slope() {
        let node = create_prelu_node("prelu1", 1);
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @"let prelu1 = PReluConfig::new().init(device);");
    }
}
