use super::{NodeCodegen, arg_to_ident};
use crate::burn::{BurnImports, Scope};
use burn::record::PrecisionSettings;
use onnx_ir::{ArgType, Argument};
use proc_macro2::TokenStream;
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::bitshift::BitShiftNode {
    fn inputs(&self) -> Vec<&Argument> {
        self.inputs.iter().collect()
    }

    fn outputs(&self) -> Vec<&Argument> {
        self.outputs.iter().collect()
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let lhs_arg = self.inputs.first().unwrap();
        let rhs_arg = self.inputs.get(1).unwrap();
        let output = arg_to_ident(self.outputs.first().unwrap());

        let lhs = match &lhs_arg.ty {
            ArgType::Tensor(_) => scope.tensor_use_owned(lhs_arg, node_position),
            ArgType::Scalar(_) => {
                let name = arg_to_ident(lhs_arg);
                quote! { #name }
            }
            _ => panic!("BitShift lhs must be a tensor or scalar"),
        };

        let rhs = match &rhs_arg.ty {
            ArgType::Tensor(_) => scope.tensor_use_owned(rhs_arg, node_position),
            ArgType::Scalar(_) => {
                let name = arg_to_ident(rhs_arg);
                quote! { #name }
            }
            _ => panic!("BitShift rhs must be a tensor or scalar"),
        };

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

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::tensor::Tensor");
    }
}
