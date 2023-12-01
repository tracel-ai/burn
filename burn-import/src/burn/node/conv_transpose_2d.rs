use super::{Node, NodeCodegen, SerializationBackend};
use crate::burn::{BurnImports, OtherType, Scope, TensorType, ToTokens, Type};
use burn::{
    module::{ConstantRecord, Param, ParamId},
    nn::conv::{ConvTranspose2dConfig, ConvTranspose2dRecord},
    record::{PrecisionSettings, Record},
    tensor::{DataSerialize, Tensor},
};
use proc_macro2::TokenStream;
use quote::quote;
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct ConvTranspose2dNode<PS: PrecisionSettings> {
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
    pub data_weights: DataSerialize<PS::FloatElem>,
    pub data_bias: Option<DataSerialize<PS::FloatElem>>,
    pub config: ConvTranspose2dConfig,
}

impl<PS: PrecisionSettings> ConvTranspose2dNode<PS> {
    pub fn new<S: AsRef<str>>(
        name: S,
        input: TensorType,
        output: TensorType,
        data_weights: DataSerialize<PS::FloatElem>,
        data_bias: Option<DataSerialize<PS::FloatElem>>,
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

impl<PS: PrecisionSettings> NodeCodegen<PS> for ConvTranspose2dNode<PS> {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }
    fn field_type(&self) -> Option<Type> {
        Some(Type::Other(self.field.clone()))
    }

    fn field_init(&self, with_record: bool) -> Option<TokenStream> {
        let name = &self.field.name;
        let channels = self.config.channels.to_tokens();
        let kernel_size = self.config.kernel_size.to_tokens();
        let stride = self.config.stride.to_tokens();
        let dilation = self.config.dilation.to_tokens();
        let groups = self.config.groups.to_tokens();
        let padding = self.config.padding.to_tokens();
        let bias = self.config.bias;

        let init_line = match with_record {
            true => quote! {
                init_with(record.#name);
            },
            false => quote! {
                init();
            },
        };

        let tokens = quote! {
            let #name = ConvTranspose2dConfig::new(#channels, #kernel_size)
                .with_stride(#stride)
                .with_padding(#padding)
                .with_dilation(#dilation)
                .with_groups(#groups)
                .with_bias(#bias)
                .#init_line
        };

        Some(tokens)
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let record = ConvTranspose2dRecord::<SerializationBackend> {
            weight: Param::new(
                ParamId::new(),
                Tensor::from_data(self.data_weights.clone().convert()),
            ),
            bias: self
                .data_bias
                .as_ref()
                .map(|bias| Param::new(ParamId::new(), Tensor::from_data(bias.clone().convert()))),
            stride: [ConstantRecord::new(); 2],
            kernel_size: [ConstantRecord::new(); 2],
            dilation: [ConstantRecord::new(); 2],
            groups: ConstantRecord::new(),
            padding: [ConstantRecord::new(); 2],
            padding_out: [ConstantRecord::new(); 2],
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{
        graph::BurnGraph,
        node::{conv_transpose_2d::ConvTranspose2dNode, test::assert_tokens},
        TensorType,
    };
    use burn::{nn::conv::ConvTranspose2dConfig, record::FullPrecisionSettings, tensor::Data};

    #[test]
    fn test_codegen() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ConvTranspose2dNode::new(
            "conv_transpose_2d",
            TensorType::new_float("input", 4),
            TensorType::new_float("output", 4),
            Data::from([2.]).serialize(),
            None,
            ConvTranspose2dConfig::new([3, 3], [3, 3]).with_padding([0, 0]),
        ));

        graph.register_input_output(vec!["input".to_string()], vec!["output".to_string()]);

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };
            use burn::nn::conv::ConvTranspose2d;
            use burn::nn::conv::ConvTranspose2dConfig;

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                conv_transpose_2d: ConvTranspose2d<B>,
                phantom: core::marker::PhantomData<B>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new_with(record: ModelRecord<B>) -> Self {
                    let conv_transpose_2d = ConvTranspose2dConfig::new([3, 3], [3, 3])
                        .with_stride([1, 1])
                        .with_padding([0, 0])
                        .with_dilation([1, 1])
                        .with_groups(1)
                        .with_bias(true)
                        .init_with(record.conv_transpose_2d);

                    Self {
                        conv_transpose_2d,
                        phantom: core::marker::PhantomData,
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
