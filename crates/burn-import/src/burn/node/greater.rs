use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone)]
pub struct GreaterNode {
    pub lhs: Type,
    pub rhs: Type,
    pub output: Type,
}

impl GreaterNode {
    pub fn new(lhs: Type, rhs: Type, output: Type) -> Self {
        Self { lhs, rhs, output }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for GreaterNode {
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
            Type::Shape(shape) => {
                let name = shape.name.clone();
                quote! { #name }
            }
            _ => panic!("lhs must be a tensor, scalar, or shape"),
        };

        let rhs = match &self.rhs {
            Type::Tensor(tensor) => scope.tensor_use_owned(tensor, node_position),
            Type::Scalar(scalar) => {
                let name = scalar.name.clone();
                quote! { #name }
            }
            Type::Shape(shape) => {
                let name = shape.name.clone();
                quote! { #name }
            }
            _ => panic!("rhs must be a tensor, scalar, or shape"),
        };

        let output = &self.output.name();

        let function = match (&self.lhs, &self.rhs) {
            (Type::Tensor(lhs_tensor), Type::Tensor(rhs_tensor)) => {
                let lhs_rank = lhs_tensor.rank;
                let rhs_rank = rhs_tensor.rank;

                if lhs_rank == rhs_rank {
                    quote! { #lhs.greater(#rhs) }
                } else if lhs_rank > rhs_rank {
                    let num_dims = lhs_rank - rhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.greater(#rhs.unsqueeze_dims(&[#(#dims),*])) }
                } else {
                    let num_dims = rhs_rank - lhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.unsqueeze_dims(&[#(#dims),*]).greater(#rhs) }
                }
            }
            (Type::Tensor(_), Type::Scalar(_)) => quote! { #lhs.greater_elem(#rhs) },
            (Type::Scalar(_), Type::Tensor(_)) => {
                // L > R == R < L
                quote! { #rhs.lower_elem(#lhs) }
            }
            (Type::Shape(_), Type::Tensor(_)) => quote! {
                Tensor::<B, 1, burn::tensor::Int>::from_data(&#lhs as &[_], &*self.device).greater(#rhs)
            },
            (Type::Tensor(_), Type::Shape(_)) => quote! {
                #lhs.greater(Tensor::<B, 1, burn::tensor::Int>::from_data(&#rhs as &[_], &*self.device))
            },
            (lhs, rhs) => panic!("greater is not supported for {lhs:?} > {rhs:?}"),
        };

        quote! {
            let #output = #function;
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Greater(self)
    }
}

impl OnnxIntoNode for GreaterNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Greater(n) = node else {
            panic!("Expected Greater node");
        };
        let lhs = Type::from(n.inputs.first().unwrap());
        let rhs = Type::from(n.inputs.get(1).unwrap());
        let output = Type::from(n.outputs.first().unwrap());
        Self::new(lhs, rhs, output)
    }
}
