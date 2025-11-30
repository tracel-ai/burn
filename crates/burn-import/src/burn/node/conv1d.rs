use super::{Node, NodeCodegen, OnnxIntoNode, SerializationBackend, extract_node_data};
use crate::burn::{BurnImports, OtherType, Scope, TensorType, ToTokens, Type};
use burn::{
    module::{ConstantRecord, Param, ParamId},
    nn::conv::Conv1dRecord,
    record::{PrecisionSettings, Record},
    tensor::{Tensor, TensorData},
};
use onnx_ir::node::conv1d::Conv1dConfig;
use proc_macro2::TokenStream;
use quote::quote;
use serde::Serialize;

#[derive(Clone, Debug)]
pub struct Conv1dNode {
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
    pub data_weights: TensorData,
    pub data_bias: Option<TensorData>,
    pub config: Conv1dConfig,
}

impl Conv1dNode {
    pub fn new<S: AsRef<str>>(
        name: S,
        input: TensorType,
        output: TensorType,
        data_weights: TensorData,
        data_bias: Option<TensorData>,
        config: Conv1dConfig,
    ) -> Self {
        Self {
            field: OtherType::new(
                name,
                quote! {
                    Conv1d<B>
                },
            ),
            input,
            output,
            data_weights,
            data_bias,
            config,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for Conv1dNode {
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
        let channels_in = self.config.channels_in.to_tokens();
        let channels_out = self.config.channels_out.to_tokens();
        let kernel_size = self.config.kernel_size.to_tokens();
        let stride = self.config.stride.to_tokens();
        let dilation = self.config.dilation.to_tokens();
        let groups = self.config.groups.to_tokens();
        let padding = self.config.padding.to_tokens();
        let bias = self.config.bias;

        let tokens = quote! {
            let #name = Conv1dConfig::new(#channels_in, #channels_out, #kernel_size)
                .with_stride(#stride)
                .with_padding(#padding)
                .with_dilation(#dilation)
                .with_groups(#groups)
                .with_bias(#bias)
                .init(device);
        };

        Some(tokens)
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let device = Default::default();
        let record = Conv1dRecord::<SerializationBackend> {
            weight: Param::initialized(
                ParamId::new(),
                Tensor::from_data(
                    self.data_weights.clone().convert::<PS::FloatElem>(),
                    &device,
                ),
            ),
            bias: self.data_bias.as_ref().map(|bias| {
                Param::initialized(
                    ParamId::new(),
                    Tensor::from_data(bias.clone().convert::<PS::FloatElem>(), &device),
                )
            }),
            stride: ConstantRecord::new(),
            kernel_size: ConstantRecord::new(),
            dilation: ConstantRecord::new(),
            groups: ConstantRecord::new(),
            padding: ConstantRecord::new(),
        };

        let item = Record::into_item::<PS>(record);
        item.serialize(serializer)
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
        imports.register("burn::nn::PaddingConfig1d");
        imports.register("burn::nn::conv::Conv1d");
        imports.register("burn::nn::conv::Conv1dConfig");
    }

    fn into_node(self) -> Node<PS> {
        Node::Conv1d(self)
    }
}

impl OnnxIntoNode for Conv1dNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = node.config::<onnx_ir::node::conv1d::Conv1dConfig>();
        let has_bias = node.inputs.len() == 3;
        let weight = extract_node_data::<f32>(&node, 1).unwrap();
        let bias = if has_bias {
            extract_node_data::<f32>(&node, 2)
        } else {
            None
        };
        let name = &node.name;
        Self::new(name, input, output, weight, bias, config.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{conv1d::Conv1dNode, test::assert_tokens},
    };
    use burn::record::FullPrecisionSettings;
    use onnx_ir::node::padding::PaddingConfig1d;

    #[test]
    fn test_codegen() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(Conv1dNode::new(
            "conv1d",
            TensorType::new_float("input", 4),
            TensorType::new_float("output", 4),
            TensorData::from([2f32]),
            None,
            Conv1dConfig::new(3, 3, 3, 1, PaddingConfig1d::Valid, 1, 1, true),
        ));

        graph.register_input_output(
            vec!["input".to_string()],
            vec!["output".to_string()],
            &[],
            &[],
        );

        let expected = quote! {
            use burn::prelude::*;
            use burn::nn::PaddingConfig1d;
            use burn::nn::conv::Conv1d;
            use burn::nn::conv::Conv1dConfig;

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                conv1d: Conv1d<B>,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let conv1d = Conv1dConfig::new(3, 3, 3)
                        .with_stride(1)
                        .with_padding(PaddingConfig1d::Valid)
                        .with_dilation(1)
                        .with_groups(1)
                        .with_bias(true)
                        .init(device);

                    Self {
                        conv1d,
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
                    let output = self.conv1d.forward(input);

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
