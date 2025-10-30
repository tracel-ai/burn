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
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = node.config::<onnx_ir::node::pad::PadConfig>().clone();
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
        node::{pad::PadNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_pad() {
        use onnx_ir::node::pad::{ConstantValueInput, PadInput, PadMode};
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        let config = PadConfig {
            pads: PadInput::Static(vec![1, 2, 3, 4]),
            constant_value: ConstantValueInput::Static(-1.0),
            mode: PadMode::Constant,
        };
        graph.register(PadNode::new(
            TensorType::new_float("input", 2),
            TensorType::new_float("output", 2),
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

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
                    let output = input.pad((1, 2, 3, 4), -1_f32);
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
