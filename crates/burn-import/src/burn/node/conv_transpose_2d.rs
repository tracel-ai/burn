use super::{Node, NodeCodegen, OnnxIntoNode, SerializationBackend, extract_node_data};
use crate::burn::{BurnImports, OtherType, Scope, TensorType, ToTokens, Type};
use burn::{
    module::{ConstantRecord, Param, ParamId},
    nn::conv::ConvTranspose2dRecord,
    record::{PrecisionSettings, Record},
    tensor::{Tensor, TensorData},
};
use onnx_ir::node::conv_transpose2d::ConvTranspose2dConfig;
use proc_macro2::TokenStream;
use quote::quote;
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct ConvTranspose2dNode {
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
    pub data_weights: TensorData,
    pub data_bias: Option<TensorData>,
    pub config: ConvTranspose2dConfig,
}

impl ConvTranspose2dNode {
    pub fn new<S: AsRef<str>>(
        name: S,
        input: TensorType,
        output: TensorType,
        data_weights: TensorData,
        data_bias: Option<TensorData>,
        config: ConvTranspose2dConfig,
    ) -> Self {
        Self {
            field: OtherType::new(
                name,
                quote! {
                    ConvTranspose2d<B>
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

impl<PS: PrecisionSettings> NodeCodegen<PS> for ConvTranspose2dNode {
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
        let channels = self.config.channels.to_tokens();
        let kernel_size = self.config.kernel_size.to_tokens();
        let stride = self.config.stride.to_tokens();
        let dilation = self.config.dilation.to_tokens();
        let groups = self.config.groups.to_tokens();
        let padding = self.config.padding.to_tokens();
        let padding_out = self.config.padding_out.to_tokens();
        let bias = self.config.bias;

        let tokens = quote! {
            let #name = ConvTranspose2dConfig::new(#channels, #kernel_size)
                .with_stride(#stride)
                .with_padding(#padding)
                .with_padding_out(#padding_out)
                .with_dilation(#dilation)
                .with_groups(#groups)
                .with_bias(#bias)
                .init(device);
        };

        Some(tokens)
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let device = Default::default();
        let record = ConvTranspose2dRecord::<SerializationBackend> {
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
            stride: [ConstantRecord::new(); 2],
            kernel_size: [ConstantRecord::new(); 2],
            dilation: [ConstantRecord::new(); 2],
            groups: ConstantRecord::new(),
            padding: [ConstantRecord::new(); 2],
            padding_out: [ConstantRecord::new(); 2],
            channels: [ConstantRecord::new(); 2],
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
        imports.register("burn::nn::conv::ConvTranspose2d");
        imports.register("burn::nn::conv::ConvTranspose2dConfig");
    }

    fn into_node(self) -> Node<PS> {
        Node::ConvTranspose2d(self)
    }
}

impl OnnxIntoNode for ConvTranspose2dNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = onnx_ir::node::conv_transpose2d::conv_transpose2d_config(&node);
        let has_bias = node.inputs.len() == 3;
        let weight = extract_node_data::<f32>(&node, 1).unwrap();
        let bias = if has_bias {
            extract_node_data::<f32>(&node, 2)
        } else {
            None
        };
        let name = &node.name;
        Self::new(name, input, output, weight, bias, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{conv_transpose_2d::ConvTranspose2dNode, test::assert_tokens},
    };
    use burn::record::FullPrecisionSettings;

    #[test]
    fn test_codegen() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ConvTranspose2dNode::new(
            "conv_transpose_2d",
            TensorType::new_float("input", 4),
            TensorType::new_float("output", 4),
            TensorData::from([2f32]),
            None,
            ConvTranspose2dConfig::new([3, 3], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0], 1, true),
        ));

        graph.register_input_output(vec!["input".to_string()], vec!["output".to_string()]);

        let expected = quote! {
            use burn::prelude::*;
            use burn::nn::conv::ConvTranspose2d;
            use burn::nn::conv::ConvTranspose2dConfig;

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                conv_transpose_2d: ConvTranspose2d<B>,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let conv_transpose_2d = ConvTranspose2dConfig::new([3, 3], [1, 1])
                        .with_stride([0, 0])
                        .with_padding([0, 0])
                        .with_padding_out([0, 0])
                        .with_dilation([0, 0])
                        .with_groups(1)
                        .with_bias(true)
                        .init(device);

                    Self {
                        conv_transpose_2d,
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
                    let output = self.conv_transpose_2d.forward(input);

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
