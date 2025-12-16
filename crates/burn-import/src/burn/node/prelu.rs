use super::prelude::*;
use burn_store::TensorSnapshot;
use onnx_ir::ir::ArgType;

/// Calculate num_parameters from slope tensor's static shape.
fn num_parameters(node: &onnx_ir::prelu::PReluNode) -> usize {
    node.inputs
        .get(1)
        .and_then(|slope| {
            if let ArgType::Tensor(tensor) = &slope.ty {
                tensor.static_shape.as_ref()
            } else {
                None
            }
        })
        .map(|shape| shape.iter().product())
        .unwrap_or(1)
}

impl NodeCodegen for onnx_ir::prelu::PReluNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let name = Ident::new(&self.name, Span::call_site());
        let n = num_parameters(self).to_tokens();
        Some(Field::new(
            self.name.clone(),
            quote! { PRelu<B> },
            quote! { let #name = PReluConfig::new().with_num_parameters(#n).init(device); },
        ))
    }

    fn collect_snapshots(&self, field_name: &str) -> Vec<TensorSnapshot> {
        use crate::burn::node_traits::create_lazy_snapshot;

        let mut snapshots = vec![];

        // Alpha (slope) tensor at input index 1
        if let Some(alpha_input) = self.inputs.get(1) {
            let alpha_path = format!("{}.alpha", field_name);
            if let Some(mut snapshot) = create_lazy_snapshot(alpha_input, &alpha_path, "PRelu") {
                // Squeeze dimensions to 1D
                snapshot.shape = vec![num_parameters(self)];
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
    use onnx_ir::prelu::PReluNodeBuilder;

    #[test]
    fn test_prelu_forward() {
        let node = PReluNodeBuilder::new("prelu1")
            .input_tensor("input", 4, DType::F32)
            .input_tensor_shape("slope", vec![64, 1, 1], DType::F32)
            .output_tensor("output", 4, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>, slope: Tensor<B, 3>) -> Tensor<B, 4> {
            let output = self.prelu1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_prelu_field_with_channel_slope() {
        let node = PReluNodeBuilder::new("prelu1")
            .input_tensor("input", 4, DType::F32)
            .input_tensor_shape("slope", vec![64, 1, 1], DType::F32)
            .output_tensor("output", 4, DType::F32)
            .build();
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @"let prelu1 = PReluConfig::new().with_num_parameters(64).init(device);");
    }

    #[test]
    fn test_prelu_field_with_scalar_slope() {
        let node = PReluNodeBuilder::new("prelu1")
            .input_tensor("input", 4, DType::F32)
            .input_tensor_shape("slope", vec![1], DType::F32)
            .output_tensor("output", 4, DType::F32)
            .build();
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @"let prelu1 = PReluConfig::new().with_num_parameters(1).init(device);");
    }
}
