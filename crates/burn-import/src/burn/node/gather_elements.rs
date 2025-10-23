use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{TensorType, ToTokens, Type};

use burn::record::PrecisionSettings;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct GatherElementsNode {
    pub input: TensorType,
    pub index: TensorType,
    pub output: TensorType,
    pub dim: usize,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for GatherElementsNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<crate::burn::Type> {
        vec![
            Type::Tensor(self.input.clone()),
            Type::Tensor(self.index.clone()),
        ]
    }

    fn forward(
        &self,
        scope: &mut crate::burn::Scope,
        node_position: usize,
    ) -> proc_macro2::TokenStream {
        let dim = self.dim.to_tokens();
        let input = scope.tensor_use_owned(&self.input, node_position);
        let index = scope.tensor_use_owned(&self.index, node_position);
        let output = &self.output.name;

        quote! {
            let #output = #input.gather(#dim, #index);
        }
    }

    fn into_node(self) -> super::Node<PS> {
        Node::GatherElements(self)
    }
}

impl OnnxIntoNode for GatherElementsNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let input = TensorType::from(node.inputs.first().unwrap());
        let index = TensorType::from(node.inputs.get(1).unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = node.config::<onnx_ir::node::gather_elements::GatherElementsConfig>();
        Self::new(input, index, output, config.axis)
    }
}

#[cfg(test)]
mod tests {

    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{gather_elements::GatherElementsNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_gather_elements() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(GatherElementsNode::new(
            TensorType::new_float("tensor1", 2),
            TensorType::new_int("tensor2", 2),
            TensorType::new_float("tensor3", 2),
            1,
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
            &[],
            &[],
        );

        let expected = quote! {
            use burn::prelude::*;

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }

                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(
                    &self,
                    tensor1: Tensor<B, 2>,
                    tensor2: Tensor<B, 2, Int>
                ) -> Tensor<B, 2> {
                    let tensor3 = tensor1.gather(1, tensor2);

                    tensor3
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
