use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct ClipNode {
    pub input: TensorType,
    pub output: TensorType,
    pub min: Option<f64>, // Should be elem Type
    pub max: Option<f64>,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ClipNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        if let Some(min) = self.min {
            if let Some(max) = self.max {
                quote! {
                    let #output = #input.clamp(#min, #max);
                }
            } else {
                quote! {
                    let #output = #input.clamp_min(#min);
                }
            }
        } else if let Some(max) = self.max {
            quote! {
                let #output = #input.clamp_max(#max);
            }
        } else {
            panic!("Clip node must have at least one min or max value");
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Clip(self)
    }
}

impl OnnxIntoNode for ClipNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Clip(n) = node else {
            panic!("Expected Clip node");
        };
        let input = TensorType::from(n.inputs.first().unwrap());
        let output = TensorType::from(n.outputs.first().unwrap());

        // Extract static values from ClipInput enum
        let min = match &n.config.min {
            Some(onnx_ir::node::clip::ClipInput::Static(v)) => Some(*v),
            Some(onnx_ir::node::clip::ClipInput::Runtime(_)) => {
                panic!("Clip: runtime min values are not supported in burn-import")
            }
            None => None,
        };
        let max = match &n.config.max {
            Some(onnx_ir::node::clip::ClipInput::Static(v)) => Some(*v),
            Some(onnx_ir::node::clip::ClipInput::Runtime(_)) => {
                panic!("Clip: runtime max values are not supported in burn-import")
            }
            None => None,
        };

        Self::new(input, output, min, max)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{TensorType, graph::BurnGraph, node::test::assert_tokens};

    #[test]
    fn codegen_nodes_min_max() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ClipNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            Some(0.0),
            Some(1.0),
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

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.clamp(0f64, 1f64);

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn codegen_nodes_min() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ClipNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            Some(0.0),
            None,
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

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.clamp_min(0f64);

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn codegen_nodes_max() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ClipNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            None,
            Some(1.0),
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

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.clamp_max(1f64);

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
