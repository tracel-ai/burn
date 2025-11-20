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
