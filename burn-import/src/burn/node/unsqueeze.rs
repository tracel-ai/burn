use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct UnsqueezeNode {
    pub input: TensorType,
    pub output: TensorType,
    pub axes: Vec<i64>,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for UnsqueezeNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let shape_values = &self.axes.to_tokens();

        quote! {
            let #output = #input.unsqueeze(#shape_values);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Unsqueeze(self)
    }
}

// #[cfg(test)]
// mod tests {
//     use burn::record::FullPrecisionSettings;

//     use super::*;
//     use crate::burn::{
//         graph::BurnGraph,
//         node::{test::assert_tokens, unsqueeze::UnsqueezeNode},
//         TensorType,
//     };

//     #[test]
//     fn test_codegen_nodes() {
//         let mut graph = BurnGraph::<FullPrecisionSettings>::default();

//         graph.register(UnsqueezeNode::new(
//             TensorType::new_float("tensor1", 3),
//             TensorType::new_float("tensor2", 5),
//             [0, 4].into(),
//         ));

//         graph.register_input_output(vec!["tensor1".to_string()], vec!["tensor2".to_string()]);

//         let expected = quote! {
//             use burn::{
//                 module::Module,
//                 tensor::{backend::Backend, Tensor},
//             };

//             #[derive(Module, Debug)]
//             pub struct Model<B: Backend> {
//                 phantom: core::marker::PhantomData<B>,
//             }

//             impl<B: Backend> Model <B> {
//                 #[allow(unused_variables)]
//                 pub fn new_with(record: ModelRecord<B>) -> Self {
//                     Self {
//                         phantom: core::marker::PhantomData,
//                     }
//                 }
//                 #[allow(clippy::let_and_return, clippy::approx_constant)]
//                 pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
//                     let tensor2 = tensor1.unsqueeze([0,4]);

//                     tensor2
//                 }
//             }
//         };
//         assert_tokens(
//             graph,
//             quote! {
//                 let tensor1 = tensor1.unsqueeze([4, 4, 4, 4]);
//                 let tensor2 = tensor1;
//             },
//         );
//     }
// }
