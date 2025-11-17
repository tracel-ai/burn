use onnx_ir::node::avg_pool2d::AvgPool2dConfig;
use proc_macro2::TokenStream;
use quote::quote;

use burn::record::PrecisionSettings;

use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{BurnImports, OtherType, Scope, TensorType, ToTokens, Type};

#[derive(Debug, Clone)]
pub struct AvgPool2dNode {
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
    pub config: AvgPool2dConfig,
}

impl AvgPool2dNode {
    pub fn new<S: AsRef<str>>(
        name: S,
        input: TensorType,
        output: TensorType,
        config: AvgPool2dConfig,
    ) -> Self {
        Self {
            field: OtherType::new(
                name,
                quote! {
                    AvgPool2d
                },
            ),
            input,
            output,
            config,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for AvgPool2dNode {
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
        let count_include_pad = self.config.count_include_pad;

        let tokens = quote! {
            let #name = AvgPool2dConfig::new(#kernel_size)
                .with_strides(#strides)
                .with_padding(#padding)
                .with_count_include_pad(#count_include_pad)
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
        imports.register("burn::nn::pool::AvgPool2d");
        imports.register("burn::nn::pool::AvgPool2dConfig");
    }

    fn into_node(self) -> Node<PS> {
        Node::AveragePool2d(self)
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        S::serialize_none(serializer)
    }
}

impl OnnxIntoNode for AvgPool2dNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::AveragePool2d(n) = node else {
            panic!("Expected AveragePool2d node");
        };
        let input = TensorType::from(n.inputs.first().unwrap());
        let output = TensorType::from(n.outputs.first().unwrap());

        // Burn doesn't support dilations in AvgPool2d yet
        if n.config.dilation != [1, 1] {
            panic!(
                "AvgPool2d: dilation ({:?}) is not supported in Burn. Only dilation=[1, 1] is supported.",
                n.config.dilation
            );
        }

        Self::new(&n.name, input, output, n.config.clone())
    }
}
