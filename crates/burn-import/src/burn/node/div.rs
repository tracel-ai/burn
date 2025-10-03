use super::{Node, NodeCodegen};
use crate::burn::{Scope, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone)]
pub struct DivNode {
    pub lhs: Type,
    pub rhs: Type,
    pub output: Type,
}

impl DivNode {
    pub fn new(lhs: Type, rhs: Type, output: Type) -> Self {
        Self { lhs, rhs, output }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for DivNode {
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
                    quote! { #lhs.div(#rhs) }
                } else if lhs_rank > rhs_rank {
                    let num_dims = lhs_rank - rhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.div(#rhs.unsqueeze_dims(&[#(#dims),*])) }
                } else {
                    let num_dims = rhs_rank - lhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.unsqueeze_dims(&[#(#dims),*]).div(#rhs) }
                }
            }
            (Type::Tensor(_), Type::Scalar(_)) => quote! { #lhs.div_scalar(#rhs) },
            (Type::Scalar(_), Type::Scalar(_)) => quote! { #lhs / #rhs },
            (Type::Shape(_), Type::Shape(_)) => quote! {
                {
                    let mut result = #lhs;
                    for (result_item, rhs_item) in result.iter_mut().zip(#rhs.iter()) {
                        *result_item = if *rhs_item != 0 { *result_item / *rhs_item } else { *result_item };
                    }
                    result
                }
            },
            (Type::Shape(_), Type::Scalar(_)) => quote! {
                {
                    let mut result = #lhs;
                    for result_item in result.iter_mut() {
                        *result_item = if #rhs as i64 != 0 { *result_item / (#rhs as i64) } else { *result_item };
                    }
                    result
                }
            },
            (Type::Scalar(_), Type::Shape(_)) => quote! {
                {
                    let mut result = #rhs;
                    for result_item in result.iter_mut() {
                        *result_item = if *result_item != 0 { (#lhs as i64) / *result_item } else { (#lhs as i64) };
                    }
                    result
                }
            },
            (Type::Shape(_), Type::Tensor(_)) => quote! {
                Tensor::<B, 1, burn::tensor::Int>::from_data(&#lhs as &[_], &*self.device).div(#rhs)
            },
            (Type::Tensor(_), Type::Shape(_)) => quote! {
                #lhs.div(Tensor::<B, 1, burn::tensor::Int>::from_data(&#rhs as &[_], &*self.device))
            },
            _ => panic!("Division is supported for tensor, scalar, and shape types only"),
        };

        quote! {
            let #output = #function;
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Div(self)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{TensorType, graph::BurnGraph, node::test::assert_tokens};

    #[test]
    fn test_codegen_div() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(DivNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 4)),
            Type::Tensor(TensorType::new_float("tensor2", 4)),
            Type::Tensor(TensorType::new_float("tensor3", 4)),
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
        );

        let expected = quote! {
            use burn::prelude::*;

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, tensor1: Tensor<B, 4>, tensor2: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor3 = tensor1.div(tensor2);

                    tensor3
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
