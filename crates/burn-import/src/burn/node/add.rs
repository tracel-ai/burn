use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone)]
pub struct AddNode {
    pub lhs: Type,
    pub rhs: Type,
    pub output: Type,
}

impl AddNode {
    pub fn new(lhs: Type, rhs: Type, output: Type) -> Self {
        Self { lhs, rhs, output }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for AddNode {
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
                    quote! { #lhs.add(#rhs) }
                } else if lhs_rank > rhs_rank {
                    let num_dims = lhs_rank - rhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.add(#rhs.unsqueeze_dims(&[#(#dims),*])) }
                } else {
                    let num_dims = rhs_rank - lhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.unsqueeze_dims(&[#(#dims),*]).add(#rhs) }
                }
            }
            (Type::Tensor(_), Type::Scalar(_)) => quote! { #lhs.add_scalar(#rhs) },
            (Type::Scalar(_), Type::Tensor(_)) => quote! { #rhs.add_scalar(#lhs) },
            (Type::Scalar(_), Type::Scalar(_)) => quote! { #lhs + #rhs },
            (Type::Shape(_), Type::Shape(_)) => quote! {
                {
                    let mut result = #lhs;
                    for (result_item, rhs_item) in result.iter_mut().zip(#rhs.iter()) {
                        *result_item = result_item.saturating_add(*rhs_item);
                    }
                    result
                }
            },
            (Type::Shape(_), Type::Scalar(_)) => quote! {
                {
                    let mut result = #lhs;
                    for result_item in result.iter_mut() {
                        *result_item = result_item.saturating_add(#rhs as i64);
                    }
                    result
                }
            },
            (Type::Scalar(_), Type::Shape(_)) => quote! {
                {
                    let mut result = #rhs;
                    for result_item in result.iter_mut() {
                        *result_item = result_item.saturating_add(#lhs as i64);
                    }
                    result
                }
            },
            (Type::Shape(_), Type::Tensor(_)) => quote! {
                Tensor::<B, 1, burn::tensor::Int>::from_data(&#lhs as &[_], &*self.device).add(#rhs)
            },
            (Type::Tensor(_), Type::Shape(_)) => quote! {
                #lhs.add(Tensor::<B, 1, burn::tensor::Int>::from_data(&#rhs as &[_], &*self.device))
            },
            _ => panic!("Addition is supported for tensor, scalar, and shape types only"),
        };

        quote! {
            let #output = #function;
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Add(self)
    }
}

impl OnnxIntoNode for AddNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Add(n) = node else {
            panic!("Expected Add node");
        };
        let lhs = Type::from(n.inputs.first().unwrap());
        let rhs = Type::from(n.inputs.get(1).unwrap());
        let output = Type::from(n.outputs.first().unwrap());

        Self::new(lhs, rhs, output)
    }
}
