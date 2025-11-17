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
        let onnx_ir::Node::LayerNormalization(n) = &node else {
            panic!("Expected LayerNormalization node");
        };
        let inputs = &n.inputs;
        let outputs = &n.outputs;
        let config = &n.config;
        let name = &n.name;
        let input = TensorType::from(inputs.first().unwrap());
        let output = TensorType::from(outputs.first().unwrap());

        // Scale tensor (aka gamma)
        let gamma = extract_node_data(inputs, 1).expect("Gamma is required");
        // Bias (B) optional tensor
        let beta = extract_node_data(inputs, 2);

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
