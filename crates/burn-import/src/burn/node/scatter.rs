use super::{Node, NodeCodegen};
use crate::burn::{TensorType, ToTokens, Type};

use burn::record::PrecisionSettings;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct ScatterNode {
    pub input: TensorType,
    pub indices: TensorType,
    pub updates: TensorType,
    pub output: TensorType,
    pub axis: usize,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ScatterNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<crate::burn::Type> {
        vec![
            Type::Tensor(self.input.clone()),
            Type::Tensor(self.indices.clone()),
        ]
    }

    fn forward(
        &self,
        scope: &mut crate::burn::Scope,
        node_position: usize,
    ) -> proc_macro2::TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let indices = scope.tensor_use_owned(&self.indices, node_position);
        let updates = scope.tensor_use_owned(&self.updates, node_position);
        let dim = self.axis.to_tokens();
        let output = &self.output.name;

        quote! {
            let #output = #input.scatter(#dim, #indices, #updates);
        }
    }

    fn into_node(self) -> super::Node<PS> {
        Node::Scatter(self)
    }
}

//#[cfg(test)]
//mod tests {
//
//    use burn::record::FullPrecisionSettings;
//
//    use super::*;
//    use crate::burn::{
//        graph::BurnGraph,
//        node::{gather::GatherNode, test::assert_tokens},
//        ScalarKind, ScalarType, ShapeType, TensorType,
//    };
//
//    #[test]
//    fn test_codegen_gather() {
//        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
//
//        graph.register(GatherNode::new(
//            Type::Tensor(TensorType::new_float("tensor1", 2)),
//            Type::Tensor(TensorType::new_int("tensor2", 1)),
//            TensorType::new_float("tensor3", 2),
//            0,
//        ));
//
//        graph.register_input_output(
//            vec!["tensor1".to_string(), "tensor2".to_string()],
//            vec!["tensor3".to_string()],
//        );
//
//        let expected = quote! {
//            use burn::tensor::Int;
//            use burn::{
//                module::Module,
//                tensor::{backend::Backend, Tensor},
//            };
//
//            #[derive(Module, Debug)]
//            pub struct Model<B: Backend> {
//                phantom: core::marker::PhantomData<B>,
//                device: burn::module::Ignored<B::Device>,
//            }
//
//            impl<B: Backend> Model <B> {
//                #[allow(unused_variables)]
//                pub fn new(device: &B::Device) -> Self {
//                    Self {
//                        phantom: core::marker::PhantomData,
//                        device: burn::module::Ignored(device.clone()),
//                    }
//                }
//
//                #[allow(clippy::let_and_return, clippy::approx_constant)]
//                pub fn forward(
//                    &self,
//                    tensor1: Tensor<B, 2>,
//                    tensor2: Tensor<B, 1, Int>
//                ) -> Tensor<B, 2> {
//                    let tensor3 = tensor1.select(0, tensor2);
//
//                    tensor3
//                }
//            }
//        };
//
//        assert_tokens(graph.codegen(), expected);
//    }
//
//    #[test]
//    fn test_codegen_gather_shape_input() {
//        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
//
//        graph.register(GatherNode::new(
//            Type::Shape(ShapeType::new("shape1", 3)),
//            Type::Tensor(TensorType::new_int("tensor1", 1)),
//            TensorType::new_float("tensor2", 2),
//            0,
//        ));
//
//        graph.register_input_output(
//            vec!["shape1".to_string(), "tensor1".to_string()],
//            vec!["tensor2".to_string()],
//        );
//
//        let expected = quote! {
//            use burn::tensor::Int;
//            use burn::{
//                module::Module,
//                tensor::{backend::Backend, Tensor},
//            };
//
//            #[derive(Module, Debug)]
//            pub struct Model<B: Backend> {
//                phantom: core::marker::PhantomData<B>,
//                device: burn::module::Ignored<B::Device>,
//            }
//
//            impl<B: Backend> Model <B> {
//                #[allow(unused_variables)]
//                pub fn new(device: &B::Device) -> Self {
//                    Self {
//                        phantom: core::marker::PhantomData,
//                        device: burn::module::Ignored(device.clone()),
//                    }
//                }
//
//                #[allow(clippy::let_and_return, clippy::approx_constant)]
//                pub fn forward(
//                    &self,
//                    shape1: [usize; 3],
//                    tensor1: Tensor<B, 1, Int>
//                ) -> Tensor<B, 2> {
//                    let tensor2 = Tensor::from_data(&shape1 as &[_], &*self.device).select(0, tensor1);
//
//                    tensor2
//                }
//            }
//        };
//
//        assert_tokens(graph.codegen(), expected);
//    }
//
//    #[test]
//    fn test_codegen_gather_scalar_idx() {
//        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
//
//        graph.register(GatherNode::new(
//            Type::Tensor(TensorType::new_float("tensor1", 2)),
//            Type::Scalar(ScalarType::new("scalar1", ScalarKind::Int64)),
//            TensorType::new_float("tensor2", 2),
//            0,
//        ));
//
//        graph.register_input_output(
//            vec!["tensor1".to_string(), "scalar1".to_string()],
//            vec!["tensor2".to_string()],
//        );
//
//        let expected = quote! {
//            use burn::{
//                module::Module,
//                tensor::{backend::Backend, Tensor},
//            };
//
//            #[derive(Module, Debug)]
//            pub struct Model<B: Backend> {
//                phantom: core::marker::PhantomData<B>,
//                device: burn::module::Ignored<B::Device>,
//            }
//
//            impl<B: Backend> Model <B> {
//                #[allow(unused_variables)]
//                pub fn new(device: &B::Device) -> Self {
//                    Self {
//                        phantom: core::marker::PhantomData,
//                        device: burn::module::Ignored(device.clone()),
//                    }
//                }
//
//                #[allow(clippy::let_and_return, clippy::approx_constant)]
//                pub fn forward(
//                    &self,
//                    tensor1: Tensor<B, 2>,
//                    scalar1: i64
//                ) -> Tensor<B, 2> {
//                    let tensor2 = tensor1.select(0, Tensor::from_data([scalar1], &*self.device)).squeeze(0);
//
//                    tensor2
//                }
//            }
//        };
//
//        assert_tokens(graph.codegen(), expected);
//    }
//}

