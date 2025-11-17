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
        let onnx_ir::Node::Tile(n) = &node else {
            panic!("Expected Tile node");
        };
        let inputs = &n.inputs;
        let outputs = &n.outputs;
        let config = &n.config;
        let input = TensorType::from(inputs.first().unwrap());
        let output = TensorType::from(outputs.first().unwrap());
        Self::new(input, output, config.clone())
    }
}
