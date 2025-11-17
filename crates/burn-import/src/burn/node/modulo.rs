use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct ModNode {
    pub lhs: Type,
    pub rhs: Type,
    pub output: TensorType,
    pub fmod: bool, // false: use remainder (Python %), true: use fmod (C-style)
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ModNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let output = &self.output.name;

        match (&self.lhs, &self.rhs) {
            (Type::Tensor(lhs_tensor), Type::Tensor(rhs_tensor)) => {
                let lhs = scope.tensor_use_owned(lhs_tensor, node_position);
                let rhs = scope.tensor_use_owned(rhs_tensor, node_position);

                let lhs_rank = lhs_tensor.rank;
                let rhs_rank = rhs_tensor.rank;

                // Handle broadcasting if ranks differ
                if lhs_rank != rhs_rank {
                    let (smaller_tensor, larger_tensor, smaller_rank, larger_rank) =
                        if lhs_rank < rhs_rank {
                            (&lhs, &rhs, lhs_rank, rhs_rank)
                        } else {
                            (&rhs, &lhs, rhs_rank, lhs_rank)
                        };

                    // Calculate dimensions to unsqueeze
                    let rank_diff = larger_rank - smaller_rank;
                    let unsqueeze_dims = (0..rank_diff)
                        .map(|i| {
                            let i = i as isize;
                            quote! { #i }
                        })
                        .collect::<Vec<_>>();

                    let mod_op = if self.fmod {
                        quote! { fmod }
                    } else {
                        quote! { remainder }
                    };

                    if lhs_rank < rhs_rank {
                        quote! {
                            let #output = #smaller_tensor
                                .unsqueeze_dims(&[#(#unsqueeze_dims),*])
                                .#mod_op(#larger_tensor);
                        }
                    } else {
                        quote! {
                            let #output = #larger_tensor.#mod_op(#smaller_tensor.unsqueeze_dims(&[#(#unsqueeze_dims),*]));
                        }
                    }
                } else {
                    let mod_op = if self.fmod {
                        quote! { fmod }
                    } else {
                        quote! { remainder }
                    };
                    quote! {
                        let #output = #lhs.#mod_op(#rhs);
                    }
                }
            }
            (Type::Tensor(lhs_tensor), Type::Scalar(rhs_scalar)) => {
                let lhs = scope.tensor_use_owned(lhs_tensor, node_position);
                let rhs = rhs_scalar.name.clone();

                let mod_op = if self.fmod {
                    quote! { fmod_scalar }
                } else {
                    quote! { remainder_scalar }
                };

                quote! {
                    let #output = #lhs.#mod_op(#rhs);
                }
            }
            (Type::Scalar(_), Type::Tensor(_)) => {
                panic!("Mod operation with scalar dividend and tensor divisor is not supported")
            }
            _ => panic!("Mod operation requires at least one tensor input"),
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Mod(self)
    }
}

impl OnnxIntoNode for ModNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Mod(n) = node else {
            panic!("Expected Mod node");
        };
        let lhs = Type::from(n.inputs.first().unwrap());
        let rhs = Type::from(n.inputs.get(1).unwrap());
        let output = TensorType::from(n.outputs.first().unwrap());
        Self::new(lhs, rhs, output, n.config.fmod)
    }
}
