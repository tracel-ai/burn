use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use onnx_ir::node::topk::TopKConfig;
use proc_macro2::TokenStream;
use quote::{ToTokens, quote};

#[derive(Debug, Clone, new)]
pub struct TopKNode {
    pub input: TensorType,
    pub outputs: Vec<TensorType>,
    pub config: TopKConfig,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for TopKNode {
    fn output_types(&self) -> Vec<Type> {
        self.outputs
            .iter()
            .map(|t| Type::Tensor(t.clone()))
            .collect()
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let axis = self.config.axis.to_token_stream();

        // Extract static k from the enum wrapper
        let k_value = match &self.config.k {
            onnx_ir::node::topk::TopKInput::Static(k) => k,
            onnx_ir::node::topk::TopKInput::Runtime(_) => {
                panic!("Runtime k value is not supported in burn-import")
            }
        };
        let k = k_value.to_token_stream();

        let input = scope.tensor_use_owned(&self.input, node_position);
        let values_output = &self.outputs[0].name;
        let indices_output = &self.outputs[1].name;

        quote! {
            let (#values_output, #indices_output) = #input.topk_with_indices(#k, #axis);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::TopK(self)
    }
}

impl OnnxIntoNode for TopKNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::TopK(n) = &node else {
            panic!("Expected TopK node");
        };
        let inputs = &n.inputs;
        let outputs = &n.outputs;
        let config = &n.config;
        let input = TensorType::from(inputs.first().unwrap());
        let outputs = outputs.iter().map(TensorType::from).collect();
        Self::new(input, outputs, config.clone())
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{test::assert_tokens, top_k::TopKNode},
    };

    #[test]
    fn test_codegen_nodes() {
        use onnx_ir::node::topk::TopKInput;
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        let config = TopKConfig {
            axis: 1,
            k: TopKInput::Static(3),
        };

        graph.register(TopKNode::new(
            TensorType::new_float("input_tensor", 4),
            vec![
                TensorType::new_float("values_tensor", 4),
                TensorType::new_int("indices_tensor", 4),
            ],
            config,
        ));

        graph.register_input_output(
            vec!["input_tensor".to_string()],
            vec!["values_tensor".to_string(), "indices_tensor".to_string()],
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
                pub fn forward(&self, input_tensor: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4, Int>) {
                    let (values_tensor, indices_tensor) = input_tensor.topk_with_indices(3usize, 1usize);
                    (values_tensor, indices_tensor)
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
