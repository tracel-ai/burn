use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorKind, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

#[derive(Debug, Clone, new)]
pub struct OneHotNode {
    pub input: TensorType,
    pub output: TensorType,
    pub num_classes: usize,
    pub values: [f32; 2],
    pub values_type: TensorType,
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
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = node.config::<onnx_ir::node::one_hot::OneHotConfig>();

        // Extract num_classes from config.depth
        let num_classes = match config.depth {
            onnx_ir::node::one_hot::OneHotDepthInput::Static(d) => d,
            onnx_ir::node::one_hot::OneHotDepthInput::Runtime(_) => {
                panic!("OneHot with runtime depth is not supported in burn-import")
            }
        };

        // Extract values from config.values
        let values = match config.values {
            onnx_ir::node::one_hot::OneHotValuesInput::Static(v) => v,
            onnx_ir::node::one_hot::OneHotValuesInput::Runtime(_) => {
                panic!("OneHot with runtime values is not supported in burn-import")
            }
        };

        // Derive values_type from output to preserve the correct output element type
        // Even though onnx-ir stores values as [f32; 2], the output type is determined
        // by the ONNX model's output element type (e.g., Int64 -> Int)
        let mut values_type = output.clone();
        values_type.name = format_ident!("{}_values", output.name);

        let axis = config.axis;
        Self::new(input, output, num_classes, values, values_type, axis)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{one_hot::OneHotNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_nodes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(OneHotNode::new(
            TensorType::new_float("tensor1", 1),
            TensorType::new_float("tensor2", 2),
            3,
            [0., 1.],
            TensorType::new_float("tensor3", 1),
            -1,
        ));

        graph.register_input_output(
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
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
                pub fn forward(&self, tensor1: Tensor<B, 1>) -> Tensor<B, 2> {
                    let tensor2 = tensor1
                        .one_hot_fill(3usize, 1f32, 0f32, -1i64);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
