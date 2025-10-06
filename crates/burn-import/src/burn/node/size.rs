use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{ScalarType, Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct SizeNode {
    pub input: TensorType,
    pub output: ScalarType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for SizeNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Scalar(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        quote! {
            let #output = #input.shape.num_elements();
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Size(self)
    }
}

impl OnnxIntoNode for SizeNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let input = match Type::from(node.inputs.first().unwrap()) {
            Type::Tensor(t) => t,
            _ => panic!("Size expects tensor input"),
        };
        let output = match Type::from(node.outputs.first().unwrap()) {
            Type::Scalar(s) => s,
            _ => panic!("Size expects scalar output"),
        };
        Self::new(input, output)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        ScalarKind, ScalarType, TensorType, graph::BurnGraph, node::test::assert_tokens,
    };

    #[test]
    fn test_codegen_size() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(SizeNode::new(
            TensorType::new_float("tensor1", 4),
            ScalarType::new("scalar1", ScalarKind::Int64),
        ));

        graph.register_input_output(vec!["tensor1".to_string()], vec!["scalar1".to_string()]);

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
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> i64 {
                    let scalar1 = tensor1.shape.num_elements();
                    scalar1
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
