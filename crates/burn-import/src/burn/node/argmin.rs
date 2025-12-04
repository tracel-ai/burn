use super::prelude::*;

impl NodeCodegen for onnx_ir::node::argmin::ArgMinNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        // NOTE: select_last_index=1 is not supported (will panic during conversion)
        let axis = self.config.axis.to_tokens();
        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();

        let input = scope.arg(input_arg);
        let output = arg_to_ident(output_arg);

        match &output_arg.ty {
            onnx_ir::ir::ArgType::Tensor(tensor) => {
                if self.config.keepdims {
                    // keepdims=True: Burn's argmin keeps dimensions by default
                    quote! {
                        let #output = #input.argmin(#axis);
                    }
                } else {
                    // keepdims=False: use argmin followed by squeeze to remove the kept dimension
                    let output_rank = tensor.rank;
                    quote! {
                        let argmin_result = #input.argmin(#axis);
                        let #output = argmin_result.squeeze_dim::<#output_rank>(#axis);
                    }
                }
            }
            onnx_ir::ir::ArgType::Scalar(_) => {
                // 1D tensor with keepdims=false -> scalar output
                // ArgMin always outputs Int64 indices
                quote! {
                    let argmin_result = #input.argmin(#axis);
                    let #output = argmin_result.into_scalar().elem::<i64>();
                }
            }
            _ => panic!("ArgMin output must be Tensor or Scalar"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::argmin::{ArgMinConfig, ArgMinNodeBuilder};

    #[test]
    fn test_argmin_keepdims() {
        let config = ArgMinConfig::new(1, true);
        let node = ArgMinNodeBuilder::new("argmin1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3, Int> {
            let output = input.argmin(1);
            output
        }
        ");
    }

    #[test]
    fn test_argmin_no_keepdims() {
        let config = ArgMinConfig::new(0, false);
        let node = ArgMinNodeBuilder::new("argmin2")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 1, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 1, Int> {
            let argmin_result = input.argmin(0);
            let output = argmin_result.squeeze_dim::<1usize>(0);
            output
        }
        ");
    }

    #[test]
    fn test_argmin_scalar_output() {
        let config = ArgMinConfig::new(0, false);
        let node = ArgMinNodeBuilder::new("argmin3")
            .input_tensor("input", 1, DType::F32)
            .output_scalar("output", DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 1>) -> i64 {
            let argmin_result = input.argmin(0);
            let output = argmin_result.into_scalar().elem::<i64>();
            output
        }
        ");
    }
}
