use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::neg::NegNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        use onnx_ir::ir::ArgType;

        let input_arg = self.inputs.first().unwrap();
        let output = arg_to_ident(self.outputs.first().unwrap());

        let input = scope.arg(input_arg);

        let neg_expr = match &input_arg.ty {
            ArgType::Tensor(_) => quote! { #input.neg() },
            ArgType::Scalar(_) => quote! { -#input },
            _ => panic!("Neg only supports tensor or scalar inputs"),
        };

        quote! {
            let #output = #neg_expr;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::neg::NegNodeBuilder;

    #[test]
    fn test_neg_forward() {
        let node = NegNodeBuilder::new("neg1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = input.neg();");
    }
}
