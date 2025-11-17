use super::{Node, NodeCodegen, OnnxIntoNode, SerializationBackend, extract_node_data};
use crate::burn::{BurnImports, OtherType, Scope, TensorType, ToTokens, Type};
use burn::{
    module::{ConstantRecord, Param, ParamId},
    nn::GroupNormRecord,
    record::{PrecisionSettings, Record},
    tensor::{Tensor, TensorData},
};
use onnx_ir::node::group_norm::GroupNormConfig;
use proc_macro2::TokenStream;
use quote::quote;
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct GroupNormNode {
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
    pub gamma: TensorData, // Scale
    pub beta: TensorData,  // Bias (B)
    pub config: GroupNormConfig,
    pub full_precision: bool,
}

impl GroupNormNode {
    pub fn new<S: AsRef<str>>(
        name: S,
        input: TensorType,
        output: TensorType,
        gamma: TensorData,
        beta: TensorData,
        config: GroupNormConfig,
        full_precision: bool,
    ) -> Self {
        Self {
            field: OtherType::new(
                name,
                quote! {
                    GroupNorm<B>
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

impl<PS: PrecisionSettings> NodeCodegen<PS> for GroupNormNode {
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
        let num_groups = self.config.num_groups.to_tokens();
        let epsilon = self.config.epsilon;

        let tokens = quote! {
            let #name = GroupNormConfig::new(#num_groups, #num_features)
                .with_epsilon(#epsilon)
                .init(device);
        };

        Some(tokens)
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let device = Default::default();
        let record = GroupNormRecord::<SerializationBackend> {
            gamma: Some(Param::initialized(
                ParamId::new(),
                Tensor::from_data(self.gamma.clone().convert::<PS::FloatElem>(), &device),
            )),
            beta: Some(Param::initialized(
                ParamId::new(),
                Tensor::from_data(self.beta.clone().convert::<PS::FloatElem>(), &device),
            )),
            affine: ConstantRecord::new(),
            num_groups: ConstantRecord::new(),
            num_channels: ConstantRecord::new(),
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
        imports.register("burn::nn::GroupNorm");
        imports.register("burn::nn::GroupNormConfig");
    }

    fn into_node(self) -> Node<PS> {
        Node::GroupNormalization(self)
    }
}

impl OnnxIntoNode for GroupNormNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::GroupNormalization(n) = &node else {
            panic!("Expected GroupNormalization node");
        };
        let inputs = &n.inputs;
        let outputs = &n.outputs;
        let config = &n.config;
        let name = &n.name;
        let input = TensorType::from(inputs.first().unwrap());
        let output = TensorType::from(outputs.first().unwrap());

        // Scale tensor (aka gamma)
        let gamma = extract_node_data(inputs, 1).expect("Gamma is required");
        // Bias (B) tensor
        let beta = extract_node_data(inputs, 2).expect("Beta is required");

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
