use super::{Node, NodeCodegen, SerializationBackend};
use crate::burn::{BurnImports, OtherType, Scope, TensorType, ToTokens, Type};
use burn::{
    module::{ConstantRecord, Param, ParamId},
    nn::{BatchNormConfig, BatchNormRecord},
    record::{PrecisionSettings, Record},
    tensor::{DataSerialize, Tensor},
};
use proc_macro2::TokenStream;
use quote::quote;
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct BatchNormNode<PS: PrecisionSettings> {
    pub dim: usize,
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
    pub gamma: DataSerialize<PS::FloatElem>,
    pub beta: DataSerialize<PS::FloatElem>,
    pub running_mean: DataSerialize<PS::FloatElem>,
    pub running_var: DataSerialize<PS::FloatElem>,
    pub config: BatchNormConfig,
}

impl<PS: PrecisionSettings> BatchNormNode<PS> {
    #[allow(clippy::too_many_arguments)]
    pub fn new<S: AsRef<str>>(
        dim: usize,
        name: S,
        input: TensorType,
        output: TensorType,
        gamma: DataSerialize<PS::FloatElem>,
        beta: DataSerialize<PS::FloatElem>,
        running_mean: DataSerialize<PS::FloatElem>,
        running_var: DataSerialize<PS::FloatElem>,
        config: BatchNormConfig,
    ) -> Self {
        let dim_tokens = dim.to_tokens();

        Self {
            dim,
            field: OtherType::new(
                name,
                quote! {
                    BatchNorm<B, #dim_tokens>
                },
            ),
            input,
            output,
            gamma,
            beta,
            running_mean,
            running_var,
            config,
        }
    }
}

macro_rules! batch_norm_serialize {
    ($self:expr, $serializer:expr) => {{
        match $self.dim {
            0 => batch_norm_serialize!($self, $serializer, 0),
            1 => batch_norm_serialize!($self, $serializer, 1),
            2 => batch_norm_serialize!($self, $serializer, 2),
            3 => batch_norm_serialize!($self, $serializer, 3),
            4 => batch_norm_serialize!($self, $serializer, 4),
            _ => panic!("Unsupported dim {}", $self.dim),
        }
    }};

    ($self:expr, $serializer:expr, $dim:expr) => {{
        let record: BatchNormRecord<SerializationBackend, $dim> = batch_norm_serialize!(record $self);
        let item = Record::into_item::<PS>(record);

        item.serialize($serializer)
    }};

    (record $self:expr) => {{
        BatchNormRecord {
            gamma: Param::new(
                ParamId::new(),
                Tensor::from_data($self.gamma.clone().convert()),
            ),
            beta: Param::new(
                ParamId::new(),
                Tensor::from_data($self.beta.clone().convert()),
            ),
            running_mean: Param::new(
                ParamId::new(),
                Tensor::from_data($self.running_mean.clone().convert()),
            ),
            running_var: Param::new(
                ParamId::new(),
                Tensor::from_data($self.running_var.clone().convert()),
            ),
            epsilon: ConstantRecord::new(),
            momentum: ConstantRecord::new(),
        }
    }};
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for BatchNormNode<PS> {
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
        let num_features = self.config.num_features.to_tokens();
        let epsilon = self.config.epsilon;
        let momentum = self.config.momentum;

        let init_line = match with_record {
            true => quote! {
                init_with(record.#name);
            },
            false => quote! {
                init();
            },
        };

        let tokens = quote! {
            let #name = BatchNormConfig::new(#num_features)
                .with_epsilon(#epsilon)
                .with_momentum(#momentum)
                .#init_line
        };

        Some(tokens)
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        batch_norm_serialize!(self, serializer)
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
        imports.register("burn::nn::BatchNorm");
        imports.register("burn::nn::BatchNormConfig");
    }

    fn into_node(self) -> Node<PS> {
        Node::BatchNorm(self)
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

        graph.register(BatchNormNode::new(
            2, // Batch norm 2d
            "norm",
            TensorType::new_float("input", 4),
            TensorType::new_float("output", 4),
            Data::from([2.]).serialize(),
            Data::from([2.]).serialize(),
            Data::from([2.]).serialize(),
            Data::from([2.]).serialize(),
            BatchNormConfig::new(128),
        ));

        graph.register_input_output(vec!["input".to_string()], vec!["output".to_string()]);

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };
            use burn::nn::BatchNorm;
            use burn::nn::BatchNormConfig;

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                norm: BatchNorm<B, 2>,
                phantom: core::marker::PhantomData<B>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new_with(record: ModelRecord<B>) -> Self {
                    let norm = BatchNormConfig::new(128)
                        .with_epsilon(0.00001f64)
                        .with_momentum(0.1f64)
                        .init_with(record.norm);

                    Self {
                        norm,
                        phantom: core::marker::PhantomData,
                    }
                }
                #[allow(clippy::let_and_return)]
                pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
                    let output = self.norm.forward(input);

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
