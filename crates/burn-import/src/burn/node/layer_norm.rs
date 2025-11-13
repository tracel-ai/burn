use super::{Node, NodeCodegen, OnnxIntoNode, SerializationBackend, extract_node_data};
use crate::burn::{BurnImports, OtherType, Scope, TensorType, ToTokens, Type};
use burn::{
    module::{ConstantRecord, Param, ParamId},
    nn::LayerNormRecord,
    record::{PrecisionSettings, Record},
    tensor::{Tensor, TensorData},
};
use onnx_ir::node::layer_norm::LayerNormConfig;
use proc_macro2::TokenStream;
use quote::quote;
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct LayerNormNode {
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
    pub gamma: TensorData,        // Scale
    pub beta: Option<TensorData>, // Bias (B)
    pub config: LayerNormConfig,
    pub full_precision: bool,
}

impl LayerNormNode {
    pub fn new<S: AsRef<str>>(
        name: S,
        input: TensorType,
        output: TensorType,
        gamma: TensorData,
        beta: Option<TensorData>,
        config: LayerNormConfig,
        full_precision: bool,
    ) -> Self {
        Self {
            field: OtherType::new(
                name,
                quote! {
                    LayerNorm<B>
                },
            ),
            input,
            output,
            gamma,
            beta,
            config,
            full_precision,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for LayerNormNode {
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
        let num_features = self.config.d_model.to_tokens();
        let epsilon = self.config.epsilon;

        let tokens = quote! {
            let #name = LayerNormConfig::new(#num_features)
                .with_epsilon(#epsilon)
                .init(device);
        };

        Some(tokens)
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let device = Default::default();
        let record = LayerNormRecord::<SerializationBackend> {
            gamma: Param::initialized(
                ParamId::new(),
                Tensor::from_data(self.gamma.clone().convert::<PS::FloatElem>(), &device),
            ),
            beta: Param::initialized(
                ParamId::new(),
                if let Some(beta) = self.beta.clone() {
                    Tensor::from_data(beta.convert::<PS::FloatElem>(), &device)
                } else {
                    Tensor::zeros([self.config.d_model], &device)
                },
            ),
            epsilon: ConstantRecord::new(),
        };

        let item = Record::into_item::<PS>(record);
        item.serialize(serializer)
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let field = &self.field.name;

        if self.full_precision {
            quote! {
                let #output = {
                    let dtype = #input.dtype();
                    self.#field.forward(#input.cast(burn::tensor::DType::F32)).cast(dtype)
                };
            }
        } else {
            quote! {
                let #output = self.#field.forward(#input);
            }
        }
    }
    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::nn::LayerNorm");
        imports.register("burn::nn::LayerNormConfig");
    }

    fn into_node(self) -> Node<PS> {
        Node::LayerNormalization(self)
    }
}

impl OnnxIntoNode for LayerNormNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let config = match &node {
            onnx_ir::ir::Node::LayerNormalization { config, .. } => config,
            _ => panic!("Expected LayerNormalization node"),
        };
        let input = TensorType::from(node.inputs().first().unwrap());
        let output = TensorType::from(node.outputs().first().unwrap());

        // Scale tensor (aka gamma)
        let gamma = extract_node_data::<f32>(&node, 1).expect("Gamma is required");
        // Bias (B) optional tensor
        let beta = extract_node_data::<f32>(&node, 2);

        let name = &node.name();
        Self::new(
            name,
            input,
            output,
            gamma,
            beta,
            config.clone(),
            config.full_precision,
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

        graph.register(LayerNormNode::new(
            "norm",
            TensorType::new_float("input", 4),
            TensorType::new_float("output", 4),
            TensorData::from([2f32]),
            Some(TensorData::from([2f32])),
            LayerNormConfig::new(128),
            false,
        ));

        graph.register_input_output(
            vec!["input".to_string()],
            vec!["output".to_string()],
            &[],
            &[],
        );

        let expected = quote! {
            use burn::prelude::*;
            use burn::nn::LayerNorm;
            use burn::nn::LayerNormConfig;

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                norm: LayerNorm<B>,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let norm = LayerNormConfig::new(128)
                        .with_epsilon(0.00001f64)
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

    #[test]
    fn test_codegen_full_precision() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(LayerNormNode::new(
            "norm",
            TensorType::new_float("input", 4),
            TensorType::new_float("output", 4),
            TensorData::from([2f32]),
            Some(TensorData::from([2f32])),
            LayerNormConfig::new(128),
            true,
        ));

        graph.register_input_output(
            vec!["input".to_string()],
            vec!["output".to_string()],
            &[],
            &[],
        );

        let expected = quote! {
            use burn::prelude::*;
            use burn::nn::LayerNorm;
            use burn::nn::LayerNormConfig;

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                norm: LayerNorm<B>,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let norm = LayerNormConfig::new(128)
                        .with_epsilon(0.00001f64)
                        .init(device);

                    Self {
                        norm,
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
                    let output = {
                        let dtype = input.dtype();
                        self.norm
                            .forward(input.cast(burn::tensor::DType::F32))
                            .cast(dtype)
                    };

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
