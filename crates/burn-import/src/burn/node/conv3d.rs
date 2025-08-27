use super::{Node, NodeCodegen, SerializationBackend};
use crate::burn::{BurnImports, OtherType, Scope, TensorType, ToTokens, Type};
use burn::{
    module::{ConstantRecord, Param, ParamId},
    nn::conv::Conv3dRecord,
    record::{PrecisionSettings, Record},
    tensor::{Tensor, TensorData},
};
use onnx_ir::node::conv3d::Conv3dConfig;
use proc_macro2::TokenStream;
use quote::quote;
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct Conv3dNode {
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
    pub data_weights: TensorData,
    pub data_bias: Option<TensorData>,
    pub config: Conv3dConfig,
}

impl Conv3dNode {
    pub fn new<S: AsRef<str>>(
        name: S,
        input: TensorType,
        output: TensorType,
        data_weights: TensorData,
        data_bias: Option<TensorData>,
        config: Conv3dConfig,
    ) -> Self {
        Self {
            field: OtherType::new(
                name,
                quote! {
                    Conv3d<B>
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

impl<PS: PrecisionSettings> NodeCodegen<PS> for Conv3dNode {
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
        let bias = self.config.bias;

        let tokens = quote! {
            let #name = Conv3dConfig::new(#channels, #kernel_size)
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
        let record = Conv3dRecord::<SerializationBackend> {
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
            stride: [ConstantRecord::new(); 3],
            kernel_size: [ConstantRecord::new(); 3],
            dilation: [ConstantRecord::new(); 3],
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
        imports.register("burn::nn::PaddingConfig3d");
        imports.register("burn::nn::conv::Conv3d");
        imports.register("burn::nn::conv::Conv3dConfig");
    }

    fn into_node(self) -> Node<PS> {
        Node::Conv3d(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{conv3d::Conv3dNode, test::assert_tokens},
    };
    use burn::record::FullPrecisionSettings;
    use onnx_ir::node::padding::PaddingConfig3d;

    #[test]
    fn test_codegen() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(Conv3dNode::new(
            "conv3d",
            TensorType::new_float("input", 5),
            TensorType::new_float("output", 5),
            TensorData::from([2f32]),
            None,
            Conv3dConfig::new(
                [3, 3],
                [3, 3, 3],
                [1, 1, 1],
                [1, 1, 1],
                1,
                true,
                PaddingConfig3d::Valid,
            ),
        ));

        graph.register_input_output(vec!["input".to_string()], vec!["output".to_string()]);

        let expected = quote! {
            use burn::prelude::*;
            use burn::nn::PaddingConfig3d;
            use burn::nn::conv::Conv3d;
            use burn::nn::conv::Conv3dConfig;

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                conv3d: Conv3d<B>,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let conv3d = Conv3dConfig::new([3, 3], [3, 3, 3])
                        .with_stride([1, 1, 1])
                        .with_padding(PaddingConfig3d::Valid)
                        .with_dilation([1, 1, 1])
                        .with_groups(1)
                        .with_bias(true)
                        .init(device);

                    Self {
                        conv3d,
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, input: Tensor<B, 5>) -> Tensor<B, 5> {
                    let output = self.conv3d.forward(input);

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
