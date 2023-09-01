use super::{Node, NodeCodegen, SerializationBackend};
use crate::burn::{BurnImports, OtherType, Scope, TensorType, ToTokens, Type};
use burn::{
    module::{Param, ParamId},
    nn::{LinearConfig, LinearRecord},
    record::{PrecisionSettings, Record},
    tensor::{DataSerialize, Tensor},
};
use proc_macro2::TokenStream;
use quote::quote;
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct LinearNode<PS: PrecisionSettings> {
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
    pub data_weights: DataSerialize<PS::FloatElem>,
    pub data_bias: Option<DataSerialize<PS::FloatElem>>,
    pub config: LinearConfig,
}

impl<PS: PrecisionSettings> LinearNode<PS> {
    pub fn new<S: AsRef<str>>(
        name: S,
        input: TensorType,
        output: TensorType,
        data_weights: DataSerialize<PS::FloatElem>,
        data_bias: Option<DataSerialize<PS::FloatElem>>,
        config: LinearConfig,
    ) -> Self {
        Self {
            field: OtherType::new(
                name,
                quote! {
                    Linear<B>
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

impl<PS: PrecisionSettings> NodeCodegen<PS> for LinearNode<PS> {
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
        let d_input = self.config.d_input.to_tokens();
        let d_output = self.config.d_output.to_tokens();
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
            let #name = LinearConfig::new(#d_input, #d_output)
                .with_bias(#bias)
                .#init_line
        };

        Some(tokens)
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let record = LinearRecord::<SerializationBackend> {
            weight: Param::new(
                ParamId::new(),
                Tensor::from_data(self.data_weights.clone().convert()),
            ),
            bias: self
                .data_bias
                .as_ref()
                .map(|bias| Param::new(ParamId::new(), Tensor::from_data(bias.clone().convert()))),
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
        imports.register("burn::nn::Linear");
        imports.register("burn::nn::LinearConfig");
    }

    fn into_node(self) -> Node<PS> {
        Node::Linear(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{graph::BurnGraph, node::test::assert_tokens, TensorType};
    use burn::{record::FullPrecisionSettings, tensor::Data};

    #[test]
    fn test_codegen() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(LinearNode::new(
            "linear",
            TensorType::new_float("input", 4),
            TensorType::new_float("output", 4),
            Data::from([2.]).serialize(),
            None,
            LinearConfig::new(128, 128),
        ));

        graph.register_input_output(vec!["input".to_string()], vec!["output".to_string()]);

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };
            use burn::nn::Linear;
            use burn::nn::LinearConfig;

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                linear: Linear<B>,
                phantom: core::marker::PhantomData<B>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new_with(record: ModelRecord<B>) -> Self {
                    let linear = LinearConfig::new(128, 128)
                        .with_bias(true)
                        .init_with(record.linear);

                    Self {
                        linear,
                        phantom: core::marker::PhantomData,
                    }
                }
                #[allow(clippy::let_and_return)]
                pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
                    let output = self.linear.forward(input);

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
