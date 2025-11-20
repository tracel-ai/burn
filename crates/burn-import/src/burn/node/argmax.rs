use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::argmax::ArgMaxNode {
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
                    // keepdims=True: Burn's argmax keeps dimensions by default
                    quote! {
                        let #output = #input.argmax(#axis);
                    }
                } else {
                    // keepdims=False: use argmax followed by squeeze to remove the kept dimension
                    let output_rank = tensor.rank;
                    quote! {
                        let argmax_result = #input.argmax(#axis);
                        let #output = argmax_result.squeeze_dim::<#output_rank>(#axis);
                    }
                }
            }
            onnx_ir::ir::ArgType::Scalar(_) => {
                // 1D tensor with keepdims=false -> scalar output
                // ArgMax always outputs Int64 indices
                quote! {
                    let argmax_result = #input.argmax(#axis);
                    let #output = argmax_result.into_scalar().elem::<i64>();
                }
            }
            _ => panic!("ArgMax output must be Tensor or Scalar"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::argmax::{ArgMaxConfig, ArgMaxNodeBuilder};

    #[test]
    fn test_argmax_keepdims() {
        let config = ArgMaxConfig::new(1, true);
        let node = ArgMaxNodeBuilder::new("argmax1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = input.argmax(1);");
    }

    #[test]
    fn test_argmax_no_keepdims() {
        let config = ArgMaxConfig::new(2, false);
        let node = ArgMaxNodeBuilder::new("argmax2")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 3, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        let argmax_result = input.argmax(2);
            let output = argmax_result.squeeze_dim::<3usize>(2);
        ");
    }

    #[test]
    fn test_argmax_scalar_output() {
        let config = ArgMaxConfig::new(0, false);
        let node = ArgMaxNodeBuilder::new("argmax3")
            .input_tensor("input", 1, DType::F32)
            .output_scalar("output", DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        let argmax_result = input.argmax(0);
            let output = argmax_result.into_scalar().elem::<i64>();
        ");
    }
}
