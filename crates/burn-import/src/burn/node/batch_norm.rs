use super::{Node, NodeCodegen, OnnxIntoNode, SerializationBackend, extract_node_data};
use crate::burn::{BurnImports, OtherType, Scope, TensorType, ToTokens, Type};
use burn::{
    module::{ConstantRecord, Param, ParamId},
    nn::BatchNormRecord,
    record::{PrecisionSettings, Record},
    tensor::{Tensor, TensorData},
};
use onnx_ir::node::batch_norm::BatchNormConfig;
use proc_macro2::TokenStream;
use quote::quote;
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct BatchNormNode {
    pub dim: usize,
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
    pub gamma: TensorData,
    pub beta: TensorData,
    pub running_mean: TensorData,
    pub running_var: TensorData,
    pub config: BatchNormConfig,
}

impl BatchNormNode {
    #[allow(clippy::too_many_arguments)]
    pub fn new<S: AsRef<str>>(
        dim: usize,
        name: S,
        input: TensorType,
        output: TensorType,
        gamma: TensorData,
        beta: TensorData,
        running_mean: TensorData,
        running_var: TensorData,
        config: BatchNormConfig,
    ) -> Self {
        Self {
            dim,
            field: OtherType::new(
                name,
                quote! {
                    BatchNorm<B>
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
        let record: BatchNormRecord<SerializationBackend> =
            batch_norm_serialize!(record $self);
        let item = Record::into_item::<PS>(record);

        item.serialize($serializer)
    }};

    (record $self:expr) => {{
        let device = Default::default();
        BatchNormRecord {
            gamma: Param::initialized(
                ParamId::new(),
                Tensor::from_data($self.gamma.clone().convert::<PS::FloatElem>(), &device),
            ),
            beta: Param::initialized(
                ParamId::new(),
                Tensor::from_data($self.beta.clone().convert::<PS::FloatElem>(), &device),
            ),
            running_mean: Param::initialized(
                ParamId::new(),
                Tensor::from_data($self.running_mean.clone().convert::<PS::FloatElem>(), &device),
            ),
            running_var: Param::initialized(
                ParamId::new(),
                Tensor::from_data($self.running_var.clone().convert::<PS::FloatElem>(), &device),
            ),
            epsilon: ConstantRecord::new(),
            momentum: ConstantRecord::new(),
        }
    }};
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for BatchNormNode {
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
        let num_features = self.config.num_features.to_tokens();
        let epsilon = self.config.epsilon;
        let momentum = self.config.momentum;

        let tokens = quote! {
            let #name = BatchNormConfig::new(#num_features)
                .with_epsilon(#epsilon)
                .with_momentum(#momentum)
                .init(device);
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
        Node::BatchNormalization(self)
    }
}

impl OnnxIntoNode for BatchNormNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let config = node.config::<onnx_ir::node::batch_norm::BatchNormConfig>();
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let dim = input.rank - 2;

        // Extract data using f32 as the element type
        let gamma = extract_node_data::<f32>(&node, 1).expect("Gamma is required");
        let beta = extract_node_data::<f32>(&node, 2).expect("Beta is required");
        let running_mean = extract_node_data::<f32>(&node, 3).expect("Running mean is required");
        let running_var = extract_node_data::<f32>(&node, 4).expect("Running var is required");

        let name = &node.name;
        Self::new(
            dim,
            name,
            input,
            output,
            gamma,
            beta,
            running_mean,
            running_var,
            config.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{TensorType, graph::BurnGraph, node::test::assert_tokens};
    use burn::record::FullPrecisionSettings;

    #[test]
    fn test_codegen() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(BatchNormNode::new(
            2, // Batch norm 2d
            "norm",
            TensorType::new_float("input", 4),
            TensorType::new_float("output", 4),
            TensorData::from([2f32]),
            TensorData::from([2f32]),
            TensorData::from([2f32]),
            TensorData::from([2f32]),
            BatchNormConfig::new(128, 0.00001, 0.1),
        ));

        graph.register_input_output(vec!["input".to_string()], vec!["output".to_string()]);

        let expected = quote! {
            use burn::prelude::*;
            use burn::nn::BatchNorm;
            use burn::nn::BatchNormConfig;

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                norm: BatchNorm<B>,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let norm = BatchNormConfig::new(128)
                        .with_epsilon(0.00001f64)
                        .with_momentum(0.1f64)
                        .init(device);

                    Self {
                        norm,
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
                    let output = self.norm.forward(input);

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
