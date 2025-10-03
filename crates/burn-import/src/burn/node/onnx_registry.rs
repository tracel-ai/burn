/// Central registry for ONNX node converters
///
/// When adding a new ONNX node:
/// 1. Create the node file (e.g., my_op.rs) with from_onnx implementation
/// 2. Add it to this registry
/// 3. Never touch to_burn.rs
///
/// The macro generates the complete dispatcher automatically.

use burn::record::PrecisionSettings;
use super::*;
use onnx_ir::NodeType;

/// Trait for ONNX node conversion
pub trait OnnxConversion: Sized {
    fn from_onnx(node: onnx_ir::Node) -> Self;
}

// Generate the dispatcher using a declarative macro
macro_rules! onnx_registry {
    (
        $(
            $onnx_type:ident => $node_type:ty
        ),* $(,)?
    ) => {
        pub fn convert_onnx_node<PS: PrecisionSettings>(
            node: onnx_ir::Node,
        ) -> Node<PS> {
            match node.node_type {
                $(
                    NodeType::$onnx_type => {
                        <$node_type>::from_onnx(node).into_node()
                    }
                )*
                _ => panic!("Unsupported ONNX node type: {:?}", node.node_type),
            }
        }
    };
}

// Registry of all ONNX nodes
// ADD NEW NODES HERE (not in to_burn.rs!)
onnx_registry! {
    Add => add::AddNode,
    Sub => sub::SubNode,
    Mul => mul::MulNode,
    Div => div::DivNode,
    // ... more nodes to be added here
}
