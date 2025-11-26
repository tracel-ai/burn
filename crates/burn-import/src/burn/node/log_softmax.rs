use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::log_softmax::LogSoftmaxNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let dim = self.config.axis.to_tokens();

        quote! {
            let #output = log_softmax(#input, #dim);
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::tensor::activation::log_softmax");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::log_softmax::{LogSoftmaxConfig, LogSoftmaxNode, LogSoftmaxNodeBuilder};

    fn create_log_softmax_node(name: &str, axis: usize) -> LogSoftmaxNode {
        let config = LogSoftmaxConfig::new(axis);

        LogSoftmaxNodeBuilder::new(name)
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_log_softmax_forward_last_axis() {
        let node = create_log_softmax_node("log_softmax1", 1);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = log_softmax(input, 1);
            output
        }
        ");
    }

    #[test]
    fn test_log_softmax_forward_axis_0() {
        let node = create_log_softmax_node("log_softmax1", 0);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = log_softmax(input, 0);
            output
        }
        ");
    }
}
