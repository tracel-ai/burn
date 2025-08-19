use super::{Node, NodeCodegen, SerializationBackend};
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
        let tokens = quote! {
            let #name = PReluConfig::new()
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

        graph.register_input_output(vec!["input".to_string()], vec!["output".to_string()]);

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
                let prelu = PReluConfig::new().init(device);
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
