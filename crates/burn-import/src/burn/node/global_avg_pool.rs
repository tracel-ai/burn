use proc_macro2::TokenStream;
use quote::quote;

use burn::record::PrecisionSettings;

use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{BurnImports, OtherType, Scope, TensorType, Type};

/// GlobalAvgPoolNode is a node that performs a global average pooling operation.
///
/// The node is implemented using the AdaptiveAvgPool1d or AdaptiveAvgPool2d module
/// depending on the input dimension. AdaptiveAvgPool with output size 1 or size (1,1)
/// is equivalent to global average pooling.
#[derive(Debug, Clone)]
pub struct GlobalAvgPoolNode {
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
}

impl GlobalAvgPoolNode {
    pub fn new<S: AsRef<str>>(name: S, input: TensorType, output: TensorType) -> Self {
        // Depending on the input dimension, we need to use a different type nn module
        let field_type = match input.rank {
            3 => quote! {
                AdaptiveAvgPool1d
            },
            4 => quote! {
                AdaptiveAvgPool2d
            },
            dim => panic!("Unsupported input dim ({dim}) for GlobalAvgPoolNode"),
        };

        Self {
            field: OtherType::new(name, field_type),
            input,
            output,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for GlobalAvgPoolNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }
    fn field_type(&self) -> Option<Type> {
        Some(Type::Other(self.field.clone()))
    }

    fn field_init(&self) -> Option<TokenStream> {
        let name = &self.field.name;

        let tokens = match self.input.rank {
            3 => {
                quote! {
                    let #name = AdaptiveAvgPool1dConfig::new(1)
                        .init();
                }
            }
            4 => {
                quote! {
                    let #name = AdaptiveAvgPool2dConfig::new([1,1])
                        .init();
                }
            }
            dim => panic!("Unsupported input dim ({dim}) for GlobalAvgPoolNode"),
        };

        Some(tokens)
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let field = &self.field.name;

        quote! {
            let #output = self.#field.forward(#input);
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        match self.input.rank {
            3 => {
                imports.register("burn::nn::pool::AdaptiveAvgPool1d");
                imports.register("burn::nn::pool::AdaptiveAvgPool1dConfig");
            }
            4 => {
                imports.register("burn::nn::pool::AdaptiveAvgPool2d");
                imports.register("burn::nn::pool::AdaptiveAvgPool2dConfig");
            }
            dim => panic!("Unsupported input dim ({dim}) for GlobalAvgPoolNode"),
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::GlobalAveragePool(self)
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        S::serialize_none(serializer)
    }
}

impl OnnxIntoNode for GlobalAvgPoolNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let (inputs, outputs, name) = match node {
            onnx_ir::Node::GlobalAveragePool {
                inputs,
                outputs,
                name,
                ..
            } => (inputs, outputs, name),
            _ => panic!("Expected GlobalAveragePool node"),
        };
        let input = TensorType::from(inputs.first().unwrap());
        let output = TensorType::from(outputs.first().unwrap());
        Self::new(name, input, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{global_avg_pool::GlobalAvgPoolNode, test::assert_tokens},
    };
    use burn::record::FullPrecisionSettings;

    #[test]
    fn test_codegen_2d() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(GlobalAvgPoolNode::new(
            "global_avg_pool1",
            TensorType::new_float("input", 4),
            TensorType::new_float("output", 4),
        ));

        graph.register_input_output(
            vec!["input".to_string()],
            vec!["output".to_string()],
            &[],
            &[],
        );

        let expected = quote! {
            use burn::prelude::*;
            use burn::nn::pool::AdaptiveAvgPool2d;
            use burn::nn::pool::AdaptiveAvgPool2dConfig;

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                global_avg_pool1: AdaptiveAvgPool2d,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let global_avg_pool1 = AdaptiveAvgPool2dConfig::new([1, 1])
                        .init();

                    Self {
                        global_avg_pool1,
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
                    let output = self.global_avg_pool1.forward(input);

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_1d() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(GlobalAvgPoolNode::new(
            "global_avg_pool1",
            TensorType::new_float("input", 3),
            TensorType::new_float("output", 3),
        ));

        graph.register_input_output(
            vec!["input".to_string()],
            vec!["output".to_string()],
            &[],
            &[],
        );

        let expected = quote! {
            use burn::prelude::*;
            use burn::nn::pool::AdaptiveAvgPool1d;
            use burn::nn::pool::AdaptiveAvgPool1dConfig;

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                global_avg_pool1: AdaptiveAvgPool1d,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let global_avg_pool1 = AdaptiveAvgPool1dConfig::new(1)
                        .init();

                    Self {
                        global_avg_pool1,
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
                    let output = self.global_avg_pool1.forward(input);

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
