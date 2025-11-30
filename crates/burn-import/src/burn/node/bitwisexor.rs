use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::bitwisexor::BitwiseXorNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let lhs = self.inputs.first().unwrap();
        let rhs = self.inputs.get(1).unwrap();
        let output = arg_to_ident(self.outputs.first().unwrap());

        let lhs_value = scope.arg(lhs);

        let rhs_value = scope.arg(rhs);

        let function = match (&lhs.ty, &rhs.ty) {
            (ArgType::Tensor(lhs_tensor), ArgType::Tensor(rhs_tensor)) => {
                let lhs_rank = lhs_tensor.rank;
                let rhs_rank = rhs_tensor.rank;

                if lhs_rank == rhs_rank {
                    quote! { #lhs_value.bitwise_xor(#rhs_value) }
                } else if lhs_rank > rhs_rank {
                    let num_dims = lhs_rank - rhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs_value.bitwise_xor(#rhs_value.unsqueeze_dims(&[#(#dims),*])) }
                } else {
                    let num_dims = rhs_rank - lhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs_value.unsqueeze_dims(&[#(#dims),*]).bitwise_xor(#rhs_value) }
                }
            }
            (ArgType::Tensor(_), ArgType::Scalar(_)) => {
                quote! { #lhs_value.bitwise_xor_scalar((#rhs_value as i64).elem()) }
            }
            (ArgType::Scalar(_), ArgType::Tensor(_)) => {
                quote! { #rhs_value.bitwise_xor_scalar((#lhs_value as i64).elem()) }
            }
            (ArgType::Scalar(_), ArgType::Scalar(_)) => {
                quote! { #lhs_value ^ #rhs_value }
            }
            _ => panic!("BitwiseXor operation requires tensor or scalar inputs"),
        };

        quote! {
            let #output = #function;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::bitwisexor::BitwiseXorNodeBuilder;

    #[test]
    fn test_bitwisexor_tensor() {
        let node = BitwiseXorNodeBuilder::new("xor1")
            .input_tensor("lhs", 2, DType::I32)
            .input_tensor("rhs", 2, DType::I32)
            .output_tensor("output", 2, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            lhs: Tensor<B, 2, Int>,
            rhs: Tensor<B, 2, Int>,
        ) -> Tensor<B, 2, Int> {
            let output = lhs.bitwise_xor(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_bitwisexor_scalar() {
        let node = BitwiseXorNodeBuilder::new("xor2")
            .input_tensor("lhs", 2, DType::I32)
            .input_scalar("rhs", DType::I32)
            .output_tensor("output", 2, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<B, 2, Int>, rhs: i32) -> Tensor<B, 2, Int> {
            let output = lhs.bitwise_xor_scalar((rhs as i64).elem());
            output
        }
        ");
    }

    #[test]
    fn test_bitwisexor_scalar_tensor() {
        let node = BitwiseXorNodeBuilder::new("xor3")
            .input_scalar("lhs", DType::I32)
            .input_tensor("rhs", 2, DType::I32)
            .output_tensor("output", 2, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: i32, rhs: Tensor<B, 2, Int>) -> Tensor<B, 2, Int> {
            let output = rhs.bitwise_xor_scalar((lhs as i64).elem());
            output
        }
        ");
    }

    #[test]
    fn test_bitwisexor_scalar_scalar() {
        let node = BitwiseXorNodeBuilder::new("xor4")
            .input_scalar("lhs", DType::I32)
            .input_scalar("rhs", DType::I32)
            .output_scalar("output", DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: i32, rhs: i32) -> i32 {
            let output = lhs ^ rhs;
            output
        }
        ");
    }

    #[test]
    fn test_bitwisexor_broadcast_lhs_higher() {
        let node = BitwiseXorNodeBuilder::new("xor5")
            .input_tensor("lhs", 3, DType::I32)
            .input_tensor("rhs", 2, DType::I32)
            .output_tensor("output", 3, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            lhs: Tensor<B, 3, Int>,
            rhs: Tensor<B, 2, Int>,
        ) -> Tensor<B, 3, Int> {
            let output = lhs.bitwise_xor(rhs.unsqueeze_dims(&[0isize]));
            output
        }
        ");
    }

    #[test]
    fn test_bitwisexor_broadcast_rhs_higher() {
        let node = BitwiseXorNodeBuilder::new("xor6")
            .input_tensor("lhs", 2, DType::I32)
            .input_tensor("rhs", 3, DType::I32)
            .output_tensor("output", 3, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            lhs: Tensor<B, 2, Int>,
            rhs: Tensor<B, 3, Int>,
        ) -> Tensor<B, 3, Int> {
            let output = lhs.unsqueeze_dims(&[0isize]).bitwise_xor(rhs);
            output
        }
        ");
    }
}
