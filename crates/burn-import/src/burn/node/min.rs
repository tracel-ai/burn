use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::min::MinNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let lhs_arg = self.inputs.first().unwrap();
        let rhs_arg = self.inputs.get(1).unwrap();
        let output = arg_to_ident(self.outputs.first().unwrap());

        // TODO: Add support for broadcasting when tensors have different ranks
        // TODO: ONNX Min spec supports variadic inputs (2+ tensors), currently only handles 2
        // TODO: Add proper error handling for non-tensor inputs

        let lhs = scope.arg(lhs_arg);

        let rhs = scope.arg(rhs_arg);

        quote! {
            let #output = #lhs.min_pair(#rhs);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::min::MinNodeBuilder;

    #[test]
    fn test_min() {
        let node = MinNodeBuilder::new("min1")
            .input_tensor("a", 2, DType::F32)
            .input_tensor("b", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = a.min_pair(b);");
    }
}
