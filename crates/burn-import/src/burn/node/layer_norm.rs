use super::prelude::*;
use burn_store::TensorSnapshot;

impl NodeCodegen for onnx_ir::node::layer_norm::LayerNormalizationNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let name = Ident::new(&self.name, Span::call_site());
        let num_features = self.config.d_model.to_tokens();
        let epsilon = self.config.epsilon;

        Some(Field::new(
            self.name.clone(),
            quote! {
                LayerNorm<B>
            },
            quote! {
                let #name = LayerNormConfig::new(#num_features)
                    .with_epsilon(#epsilon)
                    .init(device);
            },
        ))
    }

    fn collect_snapshots(&self, field_name: &str) -> Vec<TensorSnapshot> {
        use crate::burn::node_traits::create_lazy_snapshot;

        let mut snapshots = vec![];

        // Gamma (scale) tensor at input index 1
        if let Some(gamma_input) = self.inputs.get(1) {
            let gamma_path = format!("{}.gamma", field_name);
            if let Some(snapshot) = create_lazy_snapshot(gamma_input, &gamma_path, "LayerNorm") {
                snapshots.push(snapshot);
            }
        }

        // Beta (bias) tensor at input index 2
        if let Some(beta_input) = self.inputs.get(2) {
            let beta_path = format!("{}.beta", field_name);
            if let Some(snapshot) = create_lazy_snapshot(beta_input, &beta_path, "LayerNorm") {
                snapshots.push(snapshot);
            }
        }

        snapshots
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let field = Ident::new(&self.name, Span::call_site());

        if self.config.full_precision {
            quote! {
                let #output = {
                    let dtype = #input.dtype();
                    self.#field.forward(#input.cast(burn::tensor::DType::F32)).cast(dtype)
                };
            }
        } else {
            quote! {
                let #output = self.#field.forward(#input);
            }
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::nn::LayerNorm");
        imports.register("burn::nn::LayerNormConfig");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::layer_norm::{
        LayerNormConfig, LayerNormalizationNode, LayerNormalizationNodeBuilder,
    };

    fn create_layer_norm_node(name: &str) -> LayerNormalizationNode {
        let config = LayerNormConfig::new(512, 1e-5, true);

        LayerNormalizationNodeBuilder::new(name)
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_layer_norm_forward() {
        let node = create_layer_norm_node("layer_norm1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = {
                let dtype = input.dtype();
                self.layer_norm1.forward(input.cast(burn::tensor::DType::F32)).cast(dtype)
            };
            output
        }
        ");
    }

    #[test]
    fn test_layer_norm_forward_with_clone() {
        let node = create_layer_norm_node("layer_norm1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = {
                let dtype = input.clone().dtype();
                self.layer_norm1
                    .forward(input.clone().cast(burn::tensor::DType::F32))
                    .cast(dtype)
            };
            output
        }
        ");
    }
}
