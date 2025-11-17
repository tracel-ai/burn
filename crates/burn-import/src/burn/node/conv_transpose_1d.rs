use super::{Node, NodeCodegen, OnnxIntoNode, SerializationBackend, extract_node_data};
use crate::burn::{BurnImports, OtherType, Scope, TensorType, ToTokens, Type};
use burn::{
    module::{ConstantRecord, Param, ParamId},
    nn::conv::{ConvTranspose1dConfig, ConvTranspose1dRecord},
    record::{PrecisionSettings, Record},
    tensor::{Tensor, TensorData},
};
use proc_macro2::TokenStream;
use quote::quote;
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct ConvTranspose1dNode {
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
    pub data_weights: TensorData,
    pub data_bias: Option<TensorData>,
    pub config: ConvTranspose1dConfig,
}

impl ConvTranspose1dNode {
    pub fn new<S: AsRef<str>>(
        name: S,
        input: TensorType,
        output: TensorType,
        data_weights: TensorData,
        data_bias: Option<TensorData>,
        config: ConvTranspose1dConfig,
    ) -> Self {
        Self {
            field: OtherType::new(
                name,
                quote! {
                    ConvTranspose1d<B>
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

impl<PS: PrecisionSettings> NodeCodegen<PS> for ConvTranspose1dNode {
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
            let #name = ConvTranspose1dConfig::new(#channels, #kernel_size)
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
        let record = ConvTranspose1dRecord::<SerializationBackend> {
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
            padding_out: ConstantRecord::new(),
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
        imports.register("burn::nn::conv::ConvTranspose1d");
        imports.register("burn::nn::conv::ConvTranspose1dConfig");
    }

    fn into_node(self) -> Node<PS> {
        Node::ConvTranspose1d(self)
    }
}

impl OnnxIntoNode for ConvTranspose1dNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::ConvTranspose1d(n) = &node else {
            panic!("Expected ConvTranspose1d node");
        };
        let input = TensorType::from(n.inputs.first().unwrap());
        let output = TensorType::from(n.outputs.first().unwrap());
        let config = burn::nn::conv::ConvTranspose1dConfig::new(
            [n.config.channels_in, n.config.channels_out],
            n.config.kernel_size,
        )
        .with_stride(n.config.stride)
        .with_padding(n.config.padding)
        .with_dilation(n.config.dilation)
        .with_padding_out(n.config.padding_out)
        .with_groups(n.config.groups);
        let has_bias = n.inputs.len() == 3;
        let weight = extract_node_data(&n.inputs, 1).unwrap();
        let bias = if has_bias {
            extract_node_data(&n.inputs, 2)
        } else {
            None
        };
        Self::new(&n.name, input, output, weight, bias, config)
    }
}
