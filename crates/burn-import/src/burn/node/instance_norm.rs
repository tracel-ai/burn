use super::{Node, NodeCodegen, OnnxIntoNode, SerializationBackend, extract_node_data};
use crate::burn::{BurnImports, OtherType, Scope, TensorType, ToTokens, Type};
use burn::{
    module::{ConstantRecord, Param, ParamId},
    nn::InstanceNormRecord,
    record::{PrecisionSettings, Record},
    tensor::{Tensor, TensorData},
};
use onnx_ir::node::instance_norm::InstanceNormConfig;
use proc_macro2::TokenStream;
use quote::quote;
use serde::{Serialize, Serializer};

#[derive(Debug, Clone)]
pub struct InstanceNormNode {
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
    pub gamma: TensorData, // Scale
    pub beta: TensorData,  // Bias (B)
    pub config: InstanceNormConfig,
}

impl InstanceNormNode {
    pub fn new<S: AsRef<str>>(
        name: S,
        input: TensorType,
        output: TensorType,
        gamma: TensorData,
        beta: TensorData,
        config: InstanceNormConfig,
    ) -> Self {
        Self {
            field: OtherType::new(
                name,
                quote! {
                    InstanceNorm<B>
                },
            ),
            input,
            output,
            gamma,
            beta,
            config,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for InstanceNormNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn field_type(&self) -> Option<Type> {
        Some(Type::Other(self.field.clone()))
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let field = &self.field.name;

        quote! {
            let #output = self.#field.forward(#input);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::InstanceNormalization(self)
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::nn::InstanceNorm");
        imports.register("burn::nn::InstanceNormConfig");
    }

    fn field_init(&self) -> Option<TokenStream> {
        let name = &self.field.name;
        let num_features = self.config.num_features.to_tokens();
        let epsilon = self.config.epsilon;

        let tokens = quote! {
            let #name = InstanceNormConfig::new(#num_features)
                .with_epsilon(#epsilon)
                .with_affine(true)
                .init(device);
        };

        Some(tokens)
    }

    fn field_serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let device = Default::default();
        let record = InstanceNormRecord::<SerializationBackend> {
            gamma: Some(Param::initialized(
                ParamId::new(),
                Tensor::from_data(self.gamma.clone().convert::<PS::FloatElem>(), &device),
            )),
            beta: Some(Param::initialized(
                ParamId::new(),
                Tensor::from_data(self.beta.clone().convert::<PS::FloatElem>(), &device),
            )),
            affine: ConstantRecord::new(),
            num_channels: ConstantRecord::new(),
            epsilon: ConstantRecord::new(),
        };

        let item = Record::into_item::<PS>(record);
        item.serialize(serializer)
    }
}

impl OnnxIntoNode for InstanceNormNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let (inputs, outputs, config, name) = match &node {
            onnx_ir::ir::Node::InstanceNormalization {
                inputs,
                outputs,
                config,
                name,
                ..
            } => (inputs, outputs, config, name),
            _ => panic!("Expected InstanceNormalization node"),
        };
        let input = TensorType::from(inputs.first().unwrap());
        let output = TensorType::from(outputs.first().unwrap());

        // Scale tensor (aka gamma)
        let gamma = extract_node_data(inputs, 1).expect("Gamma is required");
        // Bias (B) tensor
        let beta = extract_node_data(inputs, 2).expect("Beta is required");

        Self::new(name, input, output, gamma, beta, config.clone())
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

        graph.register(InstanceNormNode::new(
            "norm",
            TensorType::new_float("input", 4),
            TensorType::new_float("output", 4),
            TensorData::from([2f32]),
            TensorData::from([2f32]),
            InstanceNormConfig::new(128, 1e-5),
        ));

        graph.register_input_output(
            vec!["input".to_string()],
            vec!["output".to_string()],
            &[],
            &[],
        );

        let expected = quote! {
            use burn::prelude::*;
            use burn::nn::InstanceNorm;
            use burn::nn::InstanceNormConfig;

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                norm: InstanceNorm<B>,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let norm = InstanceNormConfig::new(128)
                       .with_epsilon(0.00001f64)
                       .with_affine(true)
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
