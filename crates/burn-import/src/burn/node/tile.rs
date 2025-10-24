use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use onnx_ir::node::tile::TileConfig;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct TileNode {
    pub input: TensorType,
    pub output: TensorType,
    pub config: TileConfig,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for TileNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        // Extract static repeats from the enum wrapper
        let repeats_vec = match &self.config.repeats {
            onnx_ir::node::tile::TileInput::Static(repeats) => repeats,
            onnx_ir::node::tile::TileInput::Runtime(_) => {
                panic!("Runtime repeats are not supported in burn-import")
            }
        };
        let repeats = repeats_vec.iter().map(|r| r.to_tokens());

        quote! {
            let #output = #input.repeat(&[#(#repeats),*]);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Tile(self)
    }
}

impl OnnxIntoNode for TileNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = node.config::<onnx_ir::node::tile::TileConfig>().clone();
        Self::new(input, output, config)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{test::assert_tokens, tile::TileConfig, tile::TileNode},
    };

    #[test]
    fn test_codegen_tile() {
        use onnx_ir::node::tile::TileInput;
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        let config = TileConfig {
            repeats: TileInput::Static(vec![2, 3, 4]),
        };
        graph.register(TileNode::new(
            TensorType::new_float("input", 3),
            TensorType::new_float("output", 3),
            config,
        ));
        graph.register_input_output(
            vec!["input".to_string()],
            vec!["output".to_string()],
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

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
                    let output = input.repeat(&[2, 3, 4]);
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
