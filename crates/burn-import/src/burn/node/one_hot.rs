use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorKind, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct OneHotNode {
    pub input: TensorType,
    pub output: TensorType,
    pub num_classes: usize,
    pub values: [f32; 2],
    pub axis: i64,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for OneHotNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        let num_classes = &self.num_classes;
        let on_value = &self.values[1];
        let off_value = &self.values[0];
        let axis = &self.axis;
        let input_type = &self.input.kind;
        let output_type = &self.output.kind; // use actual output type from ONNX model
        match (input_type, output_type) {
            (TensorKind::Int, TensorKind::Int) | (TensorKind::Float, TensorKind::Float) => {
                quote! {
                    let #output = #input.one_hot_fill(#num_classes, #on_value, #off_value, #axis);
                }
            }
            (TensorKind::Int, TensorKind::Float) => {
                quote! {
                    let #output = #input.one_hot_fill(#num_classes, #on_value, #off_value, #axis).float();
                }
            }
            (TensorKind::Float, TensorKind::Int) => {
                quote! {
                    let #output = #input.one_hot_fill(#num_classes, #on_value, #off_value, #axis).int();
                }
            }
            (TensorKind::Int, TensorKind::Bool) | (TensorKind::Float, TensorKind::Bool) => {
                quote! {
                    let #output = #input.one_hot_fill(#num_classes, #on_value, #off_value, #axis).bool();
                }
            }
            (TensorKind::Bool, _) => panic!("Input should be numeric"),
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::OneHot(self)
    }
}

impl OnnxIntoNode for OneHotNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::OneHot(n) = node else {
            panic!("Expected OneHot node");
        };
        let input = TensorType::from(n.inputs.first().unwrap());
        let output = TensorType::from(n.outputs.first().unwrap());

        // Extract num_classes from config.depth
        let num_classes = match n.config.depth {
            onnx_ir::node::one_hot::OneHotDepthInput::Static(d) => d,
            onnx_ir::node::one_hot::OneHotDepthInput::Runtime(_) => {
                panic!("OneHot with runtime depth is not supported in burn-import")
            }
        };

        // Extract values from config.values
        let values = match n.config.values {
            onnx_ir::node::one_hot::OneHotValuesInput::Static(v) => v,
            onnx_ir::node::one_hot::OneHotValuesInput::Runtime(_) => {
                panic!("OneHot with runtime values is not supported in burn-import")
            }
        };

        let axis = n.config.axis;
        Self::new(input, output, num_classes, values, axis)
    }
}
