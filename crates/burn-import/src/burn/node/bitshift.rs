use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::bitshift::BitShiftNode {
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

        let lhs = scope.arg(lhs_arg);

        let rhs = scope.arg(rhs_arg);

        // Determine operation based on direction
        let operation = match self.config.direction {
            onnx_ir::bitshift::Direction::Left => match (&lhs_arg.ty, &rhs_arg.ty) {
                (ArgType::Tensor(_), ArgType::Tensor(_)) => {
                    quote! { #lhs.bitwise_left_shift(#rhs) }
                }
                (ArgType::Tensor(_), ArgType::Scalar(_)) => {
                    quote! { #lhs.bitwise_left_shift_scalar(#rhs.elem()) }
                }
                (ArgType::Scalar(_), ArgType::Tensor(_)) => {
                    // For scalar << tensor, broadcast scalar to tensor first
                    quote! {
                        {
                            let _scalar_tensor = Tensor::full(#rhs.shape(), #lhs, &#rhs.device());
                            _scalar_tensor.bitwise_left_shift(#rhs)
                        }
                    }
                }
                (ArgType::Scalar(_), ArgType::Scalar(_)) => {
                    quote! { #lhs << #rhs }
                }
                _ => panic!("BitShift only supports tensor and scalar inputs"),
            },
            onnx_ir::bitshift::Direction::Right => match (&lhs_arg.ty, &rhs_arg.ty) {
                (ArgType::Tensor(_), ArgType::Tensor(_)) => {
                    quote! { #lhs.bitwise_right_shift(#rhs) }
                }
                (ArgType::Tensor(_), ArgType::Scalar(_)) => {
                    quote! { #lhs.bitwise_right_shift_scalar(#rhs.elem()) }
                }
                (ArgType::Scalar(_), ArgType::Tensor(_)) => {
                    // For scalar >> tensor, broadcast scalar to tensor first
                    quote! {
                        {
                            let _scalar_tensor = Tensor::full(#rhs.shape(), #lhs, &#rhs.device());
                            _scalar_tensor.bitwise_right_shift(#rhs)
                        }
                    }
                }
                (ArgType::Scalar(_), ArgType::Scalar(_)) => {
                    quote! { #lhs >> #rhs }
                }
                _ => panic!("BitShift only supports tensor and scalar inputs"),
            },
        };

        quote! {
            let #output = #operation;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::bitshift::{BitShiftConfig, BitShiftNodeBuilder, Direction};

    #[test]
    fn test_bitshift_left_tensor() {
        let config = BitShiftConfig::new(Direction::Left);
        let node = BitShiftNodeBuilder::new("bitshift1")
            .input_tensor("lhs", 2, DType::I32)
            .input_tensor("rhs", 2, DType::I32)
            .output_tensor("output", 2, DType::I32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = lhs.bitwise_left_shift(rhs);");
    }

    #[test]
    fn test_bitshift_right_tensor() {
        let config = BitShiftConfig::new(Direction::Right);
        let node = BitShiftNodeBuilder::new("bitshift2")
            .input_tensor("lhs", 2, DType::I32)
            .input_tensor("rhs", 2, DType::I32)
            .output_tensor("output", 2, DType::I32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = lhs.bitwise_right_shift(rhs);");
    }

    #[test]
    fn test_bitshift_left_scalar() {
        let config = BitShiftConfig::new(Direction::Left);
        let node = BitShiftNodeBuilder::new("bitshift3")
            .input_tensor("lhs", 2, DType::I32)
            .input_scalar("rhs", DType::I32)
            .output_tensor("output", 2, DType::I32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = lhs.bitwise_left_shift_scalar(rhs.elem());");
    }

    #[test]
    fn test_bitshift_right_scalar() {
        let config = BitShiftConfig::new(Direction::Right);
        let node = BitShiftNodeBuilder::new("bitshift4")
            .input_tensor("lhs", 2, DType::I32)
            .input_scalar("rhs", DType::I32)
            .output_tensor("output", 2, DType::I32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = lhs.bitwise_right_shift_scalar(rhs.elem());");
    }

    #[test]
    fn test_bitshift_left_scalar_tensor() {
        let config = BitShiftConfig::new(Direction::Left);
        let node = BitShiftNodeBuilder::new("bitshift5")
            .input_scalar("lhs", DType::I32)
            .input_tensor("rhs", 2, DType::I32)
            .output_tensor("output", 2, DType::I32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        let output = {
                let _scalar_tensor = Tensor::full(rhs.shape(), lhs, &rhs.device());
                _scalar_tensor.bitwise_left_shift(rhs)
            };
        ");
    }

    #[test]
    fn test_bitshift_right_scalar_tensor() {
        let config = BitShiftConfig::new(Direction::Right);
        let node = BitShiftNodeBuilder::new("bitshift6")
            .input_scalar("lhs", DType::I32)
            .input_tensor("rhs", 2, DType::I32)
            .output_tensor("output", 2, DType::I32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        let output = {
                let _scalar_tensor = Tensor::full(rhs.shape(), lhs, &rhs.device());
                _scalar_tensor.bitwise_right_shift(rhs)
            };
        ");
    }

    #[test]
    fn test_bitshift_left_scalar_scalar() {
        let config = BitShiftConfig::new(Direction::Left);
        let node = BitShiftNodeBuilder::new("bitshift7")
            .input_scalar("lhs", DType::I32)
            .input_scalar("rhs", DType::I32)
            .output_scalar("output", DType::I32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = lhs << rhs;");
    }

    #[test]
    fn test_bitshift_right_scalar_scalar() {
        let config = BitShiftConfig::new(Direction::Right);
        let node = BitShiftNodeBuilder::new("bitshift8")
            .input_scalar("lhs", DType::I32)
            .input_scalar("rhs", DType::I32)
            .output_scalar("output", DType::I32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = lhs >> rhs;");
    }
}
