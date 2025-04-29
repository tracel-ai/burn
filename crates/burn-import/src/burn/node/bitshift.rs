use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct BitShiftNode {
    pub inputs: Vec<TensorType>,
    pub output: TensorType,
    pub direction: String,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for BitShiftNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        self.inputs
            .iter()
            .map(|t| Type::Tensor(t.clone()))
            .collect()
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let inputs: Vec<TokenStream> = self
            .inputs
            .iter()
            .map(|t| scope.tensor_use_owned(t, node_position))
            .collect();
        let output = &self.output.name;
        let direction = &self.direction;
        let shift_op = match direction.to_lowercase().as_str() {
            "left" => "<<",
            "right" => ">>",
            _ => panic!("Invalid bit shift direction"),
        };

        let input_0 = &inputs[0];
        let input_1 = &inputs[1];

        quote! {
            let #output = #input_0 #shift_op #input_1;
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::BitShift(self)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{bitshift::BitShiftNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_bitshift() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(BitShiftNode::new(
            vec![
                TensorType::new_float("input1", 4),
                TensorType::new_float("input2", 4),
            ],
            TensorType::new_float("output", 4),
            "left".to_string(),
        ));

        graph.register_input_output(
            vec!["input1".to_string(), "input2".to_string()],
            vec!["output".to_string()],
        );

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model<B> {
                pub fn new(device: B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device),
                    }
                }

                pub fn forward(&self, input1: Tensor<B, 4>, input2: Tensor<B, 4>) -> Tensor<B, 4> {
                    let output = input1 << input2;
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
