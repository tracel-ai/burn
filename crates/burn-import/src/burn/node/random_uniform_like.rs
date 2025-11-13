use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct RandomUniformLikeNode {
    pub low: f64,
    pub high: f64,
    pub input: TensorType,
    pub output: TensorType,
}

impl RandomUniformLikeNode {
    // Set distribution parameters based on low and high
    fn get_distribution(&self) -> TokenStream {
        let low = self.low;
        let high = self.high;
        quote! { Distribution::Uniform(#low, #high) }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for RandomUniformLikeNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let output = &self.output.name;
        let input = scope.tensor_use_owned(&self.input, node_position);
        let dist = self.get_distribution();
        quote! {
            let #output = #input.random_like(#dist);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::RandomUniformLike(self)
    }

    fn register_imports(&self, imports: &mut crate::burn::BurnImports) {
        imports.register("burn::tensor::Distribution");
    }
}

impl OnnxIntoNode for RandomUniformLikeNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let input = TensorType::from(node.inputs().first().unwrap());
        let output = TensorType::from(node.outputs().first().unwrap());
        let config = match &node {
            onnx_ir::ir::Node::RandomUniformLike { config, .. } => config,
            _ => panic!("Expected RandomUniformLike node"),
        };
        Self::new(config.low, config.high, input, output)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::burn::{TensorKind, TensorType, graph::BurnGraph, node::test::assert_tokens};
    use burn::record::FullPrecisionSettings;

    #[test]
    fn test_random_normal_like_codegen() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(RandomUniformLikeNode::new(
            0.0f64,
            1.0f64,
            TensorType::new("input", 2, TensorKind::Float),
            TensorType::new("output", 2, TensorKind::Float),
        ));

        graph.register_input_output(
            vec!["input".to_string()],
            vec!["output".to_string()],
            &[],
            &[],
        );

        let expected = quote! {
            use burn::prelude::*;
            use burn::tensor::Distribution;

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
                pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
                    let output = input.random_like(Distribution::Uniform(0f64, 1f64));
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
