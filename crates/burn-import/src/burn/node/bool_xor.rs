use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{ScalarKind, Scope, TensorKind, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone)]
pub struct BoolXorNode {
    pub lhs: Type,
    pub rhs: Type,
    pub output: Type,
}

impl BoolXorNode {
    pub fn new(lhs: Type, rhs: Type, output: Type) -> Self {
        Self { lhs, rhs, output }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for BoolXorNode {
    fn input_types(&self) -> Vec<Type> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let lhs = match &self.lhs {
            Type::Tensor(tensor) => scope.tensor_use_owned(tensor, node_position),
            Type::Scalar(scalar) => {
                let name = scalar.name.clone();
                quote! { #name }
            }
            _ => panic!("lhs must be a tensor or scalar"),
        };

        let rhs = match &self.rhs {
            Type::Tensor(tensor) => scope.tensor_use_owned(tensor, node_position),
            Type::Scalar(scalar) => {
                let name = scalar.name.clone();
                quote! { #name }
            }
            _ => panic!("rhs must be a tensor or scalar"),
        };

        let output = &self.output.name();

        let function = match (&self.lhs, &self.rhs) {
            (Type::Tensor(lhs_tensor), Type::Tensor(rhs_tensor)) => {
                if lhs_tensor.kind != TensorKind::Bool || rhs_tensor.kind != TensorKind::Bool {
                    panic!("xor operation requires boolean tensors");
                }

                let lhs_rank = lhs_tensor.rank;
                let rhs_rank = rhs_tensor.rank;

                // Handle broadcasting for different ranks
                if lhs_rank == rhs_rank {
                    quote! { #lhs.not_equal(#rhs) }
                } else if lhs_rank > rhs_rank {
                    // Broadcast rhs to match lhs rank by adding leading dimensions
                    let num_dims = lhs_rank - rhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.not_equal(#rhs.unsqueeze_dims(&[#(#dims),*])) }
                } else {
                    // Broadcast lhs to match rhs rank by adding leading dimensions
                    let num_dims = rhs_rank - lhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.unsqueeze_dims(&[#(#dims),*]).not_equal(#rhs) }
                }
            }
            (Type::Scalar(lhs_scalar), Type::Scalar(rhs_scalar)) => {
                if lhs_scalar.kind != ScalarKind::Bool || rhs_scalar.kind != ScalarKind::Bool {
                    panic!("xor operation requires boolean scalars");
                }
                quote! { #lhs ^ #rhs }
            }
            _ => panic!("xor is supported for tensor and scalar bool only"),
        };

        quote! {
            let #output = #function;
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Xor(self)
    }
}

impl OnnxIntoNode for BoolXorNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Xor(n) = node else {
            panic!("Expected Xor node");
        };
        let lhs = Type::from(n.inputs.first().unwrap());
        let rhs = Type::from(n.inputs.get(1).unwrap());
        let output = Type::from(n.outputs.first().unwrap());
        Self::new(lhs, rhs, output)
    }
}
