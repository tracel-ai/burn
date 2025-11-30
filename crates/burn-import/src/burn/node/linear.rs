use super::{Node, NodeCodegen, OnnxIntoNode, SerializationBackend};
use crate::burn::{BurnImports, OtherType, Scope, TensorType, ToTokens, Type};
use burn::{
    module::{Param, ParamId},
    nn::LinearRecord,
    record::{PrecisionSettings, Record},
    tensor::{Tensor, TensorData},
};
use onnx_ir::node::linear::LinearConfig;
use proc_macro2::TokenStream;
use quote::quote;
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct LinearNode {
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
    pub data_weights: TensorData,
    pub data_bias: Option<TensorData>,
    pub config: LinearConfig,
}

impl LinearNode {
    pub fn new<S: AsRef<str>>(
        name: S,
        input: TensorType,
        output: TensorType,
        data_weights: TensorData,
        data_bias: Option<TensorData>,
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

impl<PS: PrecisionSettings> NodeCodegen<PS> for LinearNode {
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
        let d_input = self.config.d_input.to_tokens();
        let d_output = self.config.d_output.to_tokens();
        let bias = self.config.bias;
        let tokens = quote! {
            let #name = LinearConfig::new(#d_input, #d_output)
                .with_bias(#bias)
                .init(device);
        };

        Some(tokens)
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let device = Default::default();
        let record = LinearRecord::<SerializationBackend> {
            weight: Param::initialized(
                ParamId::new(),
                Tensor::from_data(
                    self.data_weights.clone().convert::<PS::FloatElem>(),
                    &device,
                ),
            ),
            bias: self.data_bias.as_ref().map(|bias| {
                Param::initialized(
                    ParamId::new(),
                    Tensor::from_data(bias.clone().convert::<PS::FloatElem>(), &device),
                )
            }),
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

impl OnnxIntoNode for LinearNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        use burn::tensor::TensorData;
        use onnx_ir::ir::ArgType;

        let name = &node.name;
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = node.config::<onnx_ir::node::linear::LinearConfig>();

        // Helper function to extract and serialize data - converts to the appropriate dtype
        fn extract_data_serialize(input_index: usize, node: &onnx_ir::Node) -> Option<TensorData> {
            use onnx_ir::ir::DType;

            if node.inputs.is_empty() {
                return None;
            }

            let input = node.inputs.get(input_index)?;
            let value = input.value()?;
            let ty = input.ty.clone();

            match ty {
                ArgType::Tensor(tensor) => {
                    // Convert to the tensor's actual dtype
                    match tensor.dtype {
                        DType::F64 => Some(value.clone().convert::<f64>()),
                        DType::F32 => Some(value.clone().convert::<f32>()),
                        DType::F16 => Some(value.clone().convert::<half::f16>()),
                        DType::BF16 => Some(value.clone().convert::<half::bf16>()),
                        DType::I64 => Some(value.clone().convert::<i64>()),
                        DType::I32 => Some(value.clone().convert::<i32>()),
                        DType::I16 => Some(value.clone().convert::<i16>()),
                        DType::I8 => Some(value.clone().convert::<i8>()),
                        DType::U64 => Some(value.clone().convert::<u64>()),
                        DType::U32 => Some(value.clone().convert::<u32>()),
                        DType::U16 => Some(value.clone().convert::<u16>()),
                        DType::U8 => Some(value.clone().convert::<u8>()),
                        DType::Bool => Some(value.clone().convert::<bool>()),
                        _ => None, // Unsupported types (QFloat, Flex32)
                    }
                }
                ArgType::Scalar(dtype) => {
                    // For scalars, convert based on the scalar's dtype
                    match dtype {
                        DType::F64 => Some(value.clone().convert::<f64>()),
                        DType::F32 => Some(value.clone().convert::<f32>()),
                        DType::I64 => Some(value.clone().convert::<i64>()),
                        DType::I32 => Some(value.clone().convert::<i32>()),
                        _ => None,
                    }
                }
                ArgType::Shape(_) => {
                    // Shapes are typically i64
                    Some(value.clone().convert::<i64>())
                }
            }
        }

        let weight = extract_data_serialize(1, &node).expect("Weight is required");
        let bias = extract_data_serialize(2, &node);

        LinearNode::new(name, input, output, weight, bias, config.clone())
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

        graph.register(LinearNode::new(
            "linear",
            TensorType::new_float("input", 4),
            TensorType::new_float("output", 4),
            TensorData::from([2f32]),
            None,
            LinearConfig::new(128, 128),
        ));

        graph.register_input_output(
            vec!["input".to_string()],
            vec!["output".to_string()],
            &[],
            &[],
        );

        let expected = quote! {
            use burn::prelude::*;
            use burn::nn::Linear;
            use burn::nn::LinearConfig;

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                linear: Linear<B>,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let linear = LinearConfig::new(128, 128)
                        .with_bias(true)
                        .init(device);

                    Self {
                        linear,
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
                    let output = self.linear.forward(input);

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
