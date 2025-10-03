use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{ScalarKind, Scope, TensorKind, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;

/// Wrapper node for Pow that dispatches to either Powi or Powf based on RHS type
#[derive(Debug, Clone)]
pub enum PowNode {
    Powi(super::powi::PowiNode),
    Powf(super::powf::PowfNode),
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for PowNode {
    fn input_types(&self) -> Vec<Type> {
        match self {
            PowNode::Powi(node) => NodeCodegen::<PS>::input_types(node),
            PowNode::Powf(node) => NodeCodegen::<PS>::input_types(node),
        }
    }

    fn output_types(&self) -> Vec<Type> {
        match self {
            PowNode::Powi(node) => NodeCodegen::<PS>::output_types(node),
            PowNode::Powf(node) => NodeCodegen::<PS>::output_types(node),
        }
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        match self {
            PowNode::Powi(node) => NodeCodegen::<PS>::forward(node, scope, node_position),
            PowNode::Powf(node) => NodeCodegen::<PS>::forward(node, scope, node_position),
        }
    }

    fn into_node(self) -> Node<PS> {
        match self {
            PowNode::Powi(node) => Node::Powi(node),
            PowNode::Powf(node) => Node::Powf(node),
        }
    }
}

impl OnnxIntoNode for PowNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let lhs = Type::from(node.inputs.first().unwrap());
        let rhs = Type::from(node.inputs.get(1).unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        // Dispatch based on RHS type
        match &rhs {
            Type::Tensor(x) => match x.kind {
                TensorKind::Int => PowNode::Powi(super::powi::PowiNode::new(lhs, rhs, output)),
                TensorKind::Float => PowNode::Powf(super::powf::PowfNode::new(lhs, rhs, output)),
                _ => panic!("pow function requires RHS to be int or float type"),
            },
            Type::Scalar(x) => match x.kind {
                ScalarKind::Int32 | ScalarKind::Int64 => {
                    PowNode::Powi(super::powi::PowiNode::new(lhs, rhs, output))
                }
                ScalarKind::Float32 | ScalarKind::Float64 => {
                    PowNode::Powf(super::powf::PowfNode::new(lhs, rhs, output))
                }
                _ => panic!("pow function requires RHS to be int or float type"),
            },
            _ => panic!("pow function only supports RHS scalar or tensor types"),
        }
    }
}
