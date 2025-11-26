use super::prelude::*;
use crate::burn::TensorKind;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::nonzero::NonZeroNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Determine input tensor kind
        let input_arg = self.inputs.first().unwrap();
        let input_kind = match &input_arg.ty {
            ArgType::Tensor(t) => TensorKind::from(t.dtype),
            _ => panic!("Expected tensor input"),
        };

        // Generate the appropriate zero value based on input tensor type
        match input_kind {
            TensorKind::Float => {
                quote! {
                    let #output = #input.not_equal_elem(0.0).argwhere().transpose();
                }
            }
            TensorKind::Int => {
                quote! {
                    let #output = #input.not_equal_elem(0).argwhere().transpose();
                }
            }
            TensorKind::Bool => {
                // For bool tensors, we can use argwhere directly since false is the "zero" value
                // ONNX NonZero expects output shape [rank, num_nonzero] but argwhere returns [num_nonzero, rank]
                // So we need to transpose the result
                quote! {
                    let #output = #input.argwhere().transpose();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::nonzero::NonZeroNodeBuilder;

    #[test]
    fn test_nonzero_float() {
        let node = NonZeroNodeBuilder::new("nz1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::I64)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2, Int> {
            let output = input.not_equal_elem(0.0).argwhere().transpose();
            output
        }
        ");
    }

    #[test]
    fn test_nonzero_int() {
        let node = NonZeroNodeBuilder::new("nz2")
            .input_tensor("input", 2, DType::I32)
            .output_tensor("output", 2, DType::I64)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 2, Int> {
            let output = input.not_equal_elem(0).argwhere().transpose();
            output
        }
        ");
    }

    #[test]
    fn test_nonzero_bool() {
        let node = NonZeroNodeBuilder::new("nz3")
            .input_tensor("input", 2, DType::Bool)
            .output_tensor("output", 2, DType::I64)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2, Bool>) -> Tensor<B, 2, Int> {
            let output = input.argwhere().transpose();
            output
        }
        ");
    }
}
