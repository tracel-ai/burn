use super::prelude::*;
use burn_store::TensorSnapshot;

impl NodeCodegen for onnx_ir::node::batch_norm::BatchNormalizationNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let name = Ident::new(&self.name, Span::call_site());
        let num_features = self.config.num_features.to_tokens();
        let epsilon = self.config.epsilon;
        let momentum = self.config.momentum;

        Some(Field::new(
            self.name.clone(),
            quote! {
                BatchNorm<B>
            },
            quote! {
                let #name = BatchNormConfig::new(#num_features)
                    .with_epsilon(#epsilon)
                    .with_momentum(#momentum)
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
        imports.register("burn::nn::BatchNorm");
        imports.register("burn::nn::BatchNormConfig");
    }

    fn collect_snapshots(&self, field_name: &str) -> Vec<TensorSnapshot> {
        use crate::burn::node_traits::create_lazy_snapshot;
        let mut snapshots = vec![];

        // Gamma tensor (input index 1)
        if let Some(gamma_input) = self.inputs.get(1) {
            let gamma_path = format!("{}.gamma", field_name);
            if let Some(snapshot) = create_lazy_snapshot(gamma_input, &gamma_path, "BatchNorm") {
                snapshots.push(snapshot);
            }
        }

        // Beta tensor (input index 2)
        if let Some(beta_input) = self.inputs.get(2) {
            let beta_path = format!("{}.beta", field_name);
            if let Some(snapshot) = create_lazy_snapshot(beta_input, &beta_path, "BatchNorm") {
                snapshots.push(snapshot);
            }
        }

        // Running mean tensor (input index 3)
        if let Some(running_mean_input) = self.inputs.get(3) {
            let running_mean_path = format!("{}.running_mean", field_name);
            if let Some(snapshot) =
                create_lazy_snapshot(running_mean_input, &running_mean_path, "BatchNorm")
            {
                snapshots.push(snapshot);
            }
        }

        // Running var tensor (input index 4)
        if let Some(running_var_input) = self.inputs.get(4) {
            let running_var_path = format!("{}.running_var", field_name);
            if let Some(snapshot) =
                create_lazy_snapshot(running_var_input, &running_var_path, "BatchNorm")
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
    use onnx_ir::node::batch_norm::{
        BatchNormConfig, BatchNormalizationNode, BatchNormalizationNodeBuilder,
    };

    fn create_batch_norm_node(name: &str) -> BatchNormalizationNode {
        let config = BatchNormConfig::new(64, 1e-5, 0.9);

        BatchNormalizationNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_batch_norm_forward() {
        let node = create_batch_norm_node("batch_norm1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = self.batch_norm1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_batch_norm_forward_with_clone() {
        let node = create_batch_norm_node("batch_norm1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = self.batch_norm1.forward(input.clone());
            output
        }
        ");
    }
}
