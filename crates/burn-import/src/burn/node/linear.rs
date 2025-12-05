use super::prelude::*;
use burn_store::TensorSnapshot;

impl NodeCodegen for onnx_ir::linear::LinearNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let name = Ident::new(&self.name, Span::call_site());
        let d_input = self.config.d_input.to_tokens();
        let d_output = self.config.d_output.to_tokens();
        let bias = self.config.bias;

        // ONNX Gemm stores weights as [d_output, d_input], which matches LinearLayout::Col.
        // MatMul-sourced Linear stores weights as [d_input, d_output], matching LinearLayout::Row.
        // Using the appropriate layout avoids data transposition during import.
        let init_code = if self.config.transpose_weight {
            quote! {
                let #name = LinearConfig::new(#d_input, #d_output)
                    .with_bias(#bias)
                    .with_layout(LinearLayout::Col)
                    .init(device);
            }
        } else {
            quote! {
                let #name = LinearConfig::new(#d_input, #d_output)
                    .with_bias(#bias)
                    .init(device);
            }
        };

        Some(Field::new(
            self.name.clone(),
            quote! { Linear<B> },
            init_code,
        ))
    }

    fn collect_snapshots(&self, field_name: &str) -> Vec<TensorSnapshot> {
        use crate::burn::node_traits::create_lazy_snapshot;

        let mut snapshots = vec![];

        // Weight tensor (input index 1)
        // No transposition needed - LinearLayout::Col handles ONNX [out, in] format
        if let Some(weight_input) = self.inputs.get(1) {
            let weight_path = format!("{}.weight", field_name);
            if let Some(snapshot) = create_lazy_snapshot(weight_input, &weight_path, "Linear") {
                snapshots.push(snapshot);
            }
        }

        // Bias tensor (input index 2, optional)
        if let Some(bias_input) = self.inputs.get(2) {
            let bias_path = format!("{}.bias", field_name);
            if let Some(snapshot) = create_lazy_snapshot(bias_input, &bias_path, "Linear") {
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
        imports.register("burn::nn::Linear");
        imports.register("burn::nn::LinearConfig");
        if self.config.transpose_weight {
            imports.register("burn::nn::LinearLayout");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::ir::{ArgType, Argument, TensorType};
    use onnx_ir::linear::{LinearConfig, LinearNode};

    #[test]
    fn test_linear_forward() {
        // transpose_weight=true simulates Gemm-sourced Linear
        let config = LinearConfig::new(128, 64, true, true);
        let input = Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let weight = Argument::new(
            "weight",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let bias = Argument::new(
            "bias",
            ArgType::Tensor(TensorType::new(DType::F32, 1, None)),
        );

        let node = LinearNode {
            name: "linear1".to_string(),
            inputs: vec![input, weight, bias],
            outputs: vec![Argument::new(
                "output",
                ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
            )],
            config,
        };
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 2>,
            weight: Tensor<B, 2>,
            bias: Tensor<B, 1>,
        ) -> Tensor<B, 2> {
            let output = self.linear1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_linear_forward_no_bias() {
        // transpose_weight=false simulates MatMul-sourced Linear
        let config = LinearConfig::new(128, 64, false, false);
        let input = Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let weight = Argument::new(
            "weight",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );

        let node = LinearNode {
            name: "linear2".to_string(),
            inputs: vec![input, weight],
            outputs: vec![Argument::new(
                "output",
                ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
            )],
            config,
        };
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>, weight: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = self.linear2.forward(input);
            output
        }
        ");
    }
}
