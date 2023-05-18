use super::{Node, NodeCodegen, SerializationBackend};
use crate::burn::{BurnImports, OtherType, Scope, TensorType, ToTokens, Type};
use burn::{
    module::{Module, Param, ParamId},
    nn::{Linear, LinearConfig},
    record::{PrecisionSettings, Record},
    tensor::{Data, DataSerialize, Tensor},
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
impl<PS: PrecisionSettings> Serialize for LinearNode<PS> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let module: Linear<SerializationBackend> = self.config.init();
        let mut record = module.into_record();

        record.weight = Param::new(
            ParamId::new(),
            Tensor::from_data(Data::from(self.data_weights.clone().convert())),
        );

        if let Some(bias) = &self.data_bias {
            record.bias = Some(Param::new(
                ParamId::new(),
                Tensor::from_data(Data::from(bias.clone().convert())),
            ));
        }

        let item = Record::into_item::<PS>(record);
        item.serialize(serializer)
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for LinearNode<PS> {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(&self.input)]
    }
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(&self.output)]
    }

    fn field_type(&self) -> Option<Type> {
        Some(Type::Other(&self.field))
    }

    fn new_body(&self) -> TokenStream {
        let name = &self.field.name;
        let d_input = self.config.d_input.to_tokens();
        let d_output = self.config.d_output.to_tokens();
        let bias = self.config.bias;

        quote! {
            let #name = LinearConfig::new(#d_input, #d_output)
                .with_bias(#bias)
                .init_with(record.#name);
        }
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.use_owned_tensor(&self.input.name, node_position);
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
    use crate::burn::{graph::Graph, node::test::assert_tokens, TensorType};
    use burn::{record::FullPrecisionSettings, tensor::Data};

    #[test]
    fn test_codegen() {
        let mut graph = Graph::<FullPrecisionSettings>::default();

        graph.register(LinearNode::new(
            "linear",
            TensorType::new("input", 4),
            TensorType::new("output", 4),
            Data::from([2.]).serialize(),
            None,
            LinearConfig::new(128, 128),
        ));

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
            }

            impl<B: Backend> Model <B> {
                pub fn new_with(record: ModelRecord<B>) -> Self {
                    let linear = LinearConfig::new(128, 128)
                        .with_bias(true)
                        .init_with(record.linear);

                    Self {
                        linear,
                    }
                }

                pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
                    let output = self.linear.forward(input);

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
