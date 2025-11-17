use std::str::FromStr;

use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use onnx_ir::node::pad::PadConfig;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct PadNode {
    pub input: TensorType,
    pub output: TensorType,
    pub config: PadConfig,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for PadNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }
    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        // Extract static pads from the enum wrapper
        let pads_vec = match &self.config.pads {
            onnx_ir::node::pad::PadInput::Static(pads) => pads,
            onnx_ir::node::pad::PadInput::Runtime(_) => {
                panic!("Runtime pads are not supported in burn-import")
            }
        };
        let pads = pads_vec.iter().map(|p| p.to_tokens());

        // Extract static constant value from the enum wrapper
        let constant_value_f32 = match &self.config.constant_value {
            onnx_ir::node::pad::ConstantValueInput::Static(value) => value,
            onnx_ir::node::pad::ConstantValueInput::Runtime(_) => {
                panic!("Runtime constant value is not supported in burn-import")
            }
        };
        let constant_value_string = format!("{}_f32", constant_value_f32);
        let constant_value = TokenStream::from_str(&constant_value_string).unwrap();

        quote! {
            let #output = #input.pad((#(#pads),*), #constant_value);
        }
    }
    fn into_node(self) -> Node<PS> {
        Node::Pad(self)
    }
}

impl OnnxIntoNode for PadNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Pad(n) = node else {
            panic!("Expected Pad node");
        };
        let input = TensorType::from(n.inputs.first().unwrap());
        let output = TensorType::from(n.outputs.first().unwrap());
        Self::new(input, output, n.config.clone())
    }
}
