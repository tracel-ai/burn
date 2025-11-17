use onnx_ir::node::max_pool2d::MaxPool2dConfig;
use proc_macro2::TokenStream;
use quote::quote;

use burn::record::PrecisionSettings;

use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{BurnImports, OtherType, Scope, TensorType, ToTokens, Type};

#[derive(Debug, Clone)]
pub struct MaxPool2dNode {
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
    pub config: MaxPool2dConfig,
}

impl MaxPool2dNode {
    pub fn new<S: AsRef<str>>(
        name: S,
        input: TensorType,
        output: TensorType,
        config: MaxPool2dConfig,
    ) -> Self {
        Self {
            field: OtherType::new(
                name,
                quote! {
                    MaxPool2d
                },
            ),
            input,
            output,
            config,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for MaxPool2dNode {
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
        let kernel_size = self.config.kernel_size.to_tokens();
        let strides = self.config.strides.to_tokens();
        let padding = self.config.padding.to_tokens();
        let dilation = self.config.dilation.to_tokens();
        let tokens = quote! {
            let #name = MaxPool2dConfig::new(#kernel_size)
                .with_strides(#strides)
                .with_padding(#padding)
                .with_dilation(#dilation)
                .init();
        };

        Some(tokens)
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
        imports.register("burn::nn::PaddingConfig2d");
        imports.register("burn::nn::pool::MaxPool2d");
        imports.register("burn::nn::pool::MaxPool2dConfig");
    }

    fn into_node(self) -> Node<PS> {
        Node::MaxPool2d(self)
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        S::serialize_none(serializer)
    }
}

impl OnnxIntoNode for MaxPool2dNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::MaxPool2d(n) = node else {
            panic!("Expected MaxPool2d node");
        };
        let input = TensorType::from(n.inputs.first().unwrap());
        let output = TensorType::from(n.outputs.first().unwrap());
        Self::new(&n.name, input, output, n.config.clone())
    }
}
