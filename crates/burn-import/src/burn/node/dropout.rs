use onnx_ir::node::dropout::DropoutConfig;
use proc_macro2::TokenStream;
use quote::quote;

use burn::record::PrecisionSettings;

use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{BurnImports, OtherType, Scope, TensorType, ToTokens, Type};

#[derive(Debug, Clone)]
pub struct DropoutNode {
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
    pub config: DropoutConfig,
}

impl DropoutNode {
    pub fn new<S: AsRef<str>>(
        name: S,
        input: TensorType,
        output: TensorType,
        config: DropoutConfig,
    ) -> Self {
        Self {
            field: OtherType::new(
                name,
                quote! {
                    Dropout
                },
            ),
            input,
            output,
            config,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for DropoutNode {
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
        let prob = match &self.config.prob {
            onnx_ir::node::dropout::DropoutInput::Static(val) => val.to_tokens(),
            onnx_ir::node::dropout::DropoutInput::Runtime(_) => {
                panic!("Runtime input is not implemented for Dropout")
            }
        };
        let tokens = quote! {
            let #name = DropoutConfig::new(#prob).init();
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
        imports.register("burn::nn::Dropout");
        imports.register("burn::nn::DropoutConfig");
    }

    fn into_node(self) -> Node<PS> {
        Node::Dropout(self)
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        S::serialize_none(serializer)
    }
}

impl OnnxIntoNode for DropoutNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Dropout(n) = &node else {
            panic!("Expected Dropout node");
        };
        let input = TensorType::from(n.inputs.first().unwrap());
        let output = TensorType::from(n.outputs.first().unwrap());
        Self::new(&n.name, input, output, n.config.clone())
    }
}
