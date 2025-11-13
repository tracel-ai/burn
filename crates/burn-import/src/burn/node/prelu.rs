use super::{Node, NodeCodegen, OnnxIntoNode, SerializationBackend, extract_node_data};
use crate::burn::{BurnImports, OtherType, Scope, TensorType, Type};
use burn::{
    module::{ConstantRecord, Param, ParamId},
    nn::{PReluConfig, PReluRecord},
    record::{PrecisionSettings, Record},
    tensor::{Tensor, TensorData},
};
use proc_macro2::TokenStream;
use quote::quote;
use serde::Serialize;

#[derive(Clone, Debug)]
pub struct PReluNode {
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
    pub alpha: TensorData,
    pub config: PReluConfig,
}

impl PReluNode {
    pub fn new<S: AsRef<str>>(
        name: S,
        input: TensorType,
        output: TensorType,
        alpha: TensorData,
        config: PReluConfig,
    ) -> Self {
        Self {
            field: OtherType::new(
                name,
                quote! {
                    PRelu<B>
                },
            ),
            input,
            output,
            alpha,
            config,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for PReluNode {
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
        let alpha = &self.config.alpha;
        let num_parameters = self.config.num_parameters;
        let tokens = quote! {
            let #name = PReluConfig::new()
                .with_num_parameters(#num_parameters)
                .with_alpha(#alpha)
                .init(device);
        };

        Some(tokens)
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let device = Default::default();
        let record = PReluRecord::<SerializationBackend> {
            alpha: Param::initialized(
                ParamId::new(),
                Tensor::from_data(self.alpha.clone().convert::<PS::FloatElem>(), &device),
            ),
            alpha_value: ConstantRecord,
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
        imports.register("burn::nn::PRelu");
        imports.register("burn::nn::PReluConfig");
    }

    fn into_node(self) -> Node<PS> {
        Node::PRelu(self)
    }
}

impl OnnxIntoNode for PReluNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let input = TensorType::from(node.inputs().first().unwrap());
        let output = TensorType::from(node.outputs().first().unwrap());
        let mut weight = extract_node_data::<f32>(&node, 1).expect("PRelu weight is required");
        let name = &node.name();

        // Determine weight shape and flatten if necessary
        let weight_shape = if weight.shape.len() > 1 {
            let trailing_dims_product: usize = weight.shape[1..].iter().product();
            if trailing_dims_product == 1 {
                // Flatten to rank 1 as Burn expects
                weight.shape = vec![weight.shape[0]];
                weight.shape[0]
            } else {
                panic!(
                    "PRelu weight shape {:?} is invalid. Expected shape [C] or [C, 1, ...] where trailing dimensions are 1",
                    weight.shape
                );
            }
        } else if weight.shape.is_empty() {
            // Scalar weight
            1
        } else {
            // Already rank 1
            weight.shape[0]
        };

        let alpha_value = if weight_shape == 1 {
            weight.clone().to_vec::<f32>().unwrap()[0] as f64
        } else {
            0.01 // Default value if vectorized
        };

        let config = PReluConfig::new()
            .with_num_parameters(weight_shape)
            .with_alpha(alpha_value);

        Self::new(name, input, output, weight, config)
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

        graph.register(PReluNode::new(
            "prelu",
            TensorType::new_float("input", 4),
            TensorType::new_float("output", 4),
            TensorData::from([2f32]),
            PReluConfig::new(),
        ));

        graph.register_input_output(
            vec!["input".to_string()],
            vec!["output".to_string()],
            &[],
            &[],
        );

        let expected = quote! {
        use burn::prelude::*;
        use burn::nn::PRelu;
        use burn::nn::PReluConfig;
        #[derive(Module, Debug)]
        pub struct Model<B: Backend> {
            prelu: PRelu<B>,
            phantom: core::marker::PhantomData<B>,
            device: burn::module::Ignored<B::Device>,
        }
        impl<B: Backend> Model<B> {
            #[allow(unused_variables)]
            pub fn new(device: &B::Device) -> Self {
                let prelu = PReluConfig::new()
                    .with_num_parameters(1usize)
                    .with_alpha(0.25f64)
                    .init(device);
                Self {
                    prelu,
                    phantom: core::marker::PhantomData,
                   device: burn::module::Ignored(device.clone()),
                }
            }
            #[allow(clippy::let_and_return, clippy::approx_constant)]
            pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
                let output = self.prelu.forward(input);
                output
            }
        }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
