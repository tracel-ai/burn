use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, Type};
use burn::record::PrecisionSettings;
use onnx_ir::node::is_inf::IsInfConfig;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct IsInfNode {
    pub input: Type,
    pub output: Type,
    pub config: IsInfConfig,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for IsInfNode {
    fn input_types(&self) -> Vec<Type> {
        vec![self.input.clone()]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = match &self.input {
            Type::Tensor(tensor) => scope.tensor_use_owned(tensor, node_position),
            Type::Scalar(scalar) => {
                let name = &scalar.name;
                quote! { #name }
            }
            _ => panic!("Input must be a tensor or scalar"),
        };
        let output = &self.output.name();

        let function = match &self.output {
            Type::Scalar(_) => match (self.config.detect_negative, self.config.detect_positive) {
                (true, true) => quote! { #input.is_infinite() },
                (false, true) => quote! { #input.is_infinite() && #input.is_sign_positive() },
                (true, false) => quote! { #input.is_infinite() && #input.is_sign_negative() },
                (false, false) => quote! { false },
            },
            Type::Tensor(_) => match (self.config.detect_negative, self.config.detect_positive) {
                (true, true) => quote! { #input.is_inf() },
                (false, true) => {
                    quote! { #input.clone().is_inf().bool_and(#input.greater_elem(0.0)) }
                }
                (true, false) => {
                    quote! { #input.clone().is_inf().bool_and(#input.lower_elem(0.0)) }
                }
                (false, false) => quote! { #input.zeros_like().bool() },
            },
            _ => panic!("IsInf only supports scalar or tensor outputs"),
        };

        quote! {
            let #output = #function;
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::IsInf(self)
    }
}

impl OnnxIntoNode for IsInfNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::IsInf(n) = node else {
            panic!("Expected IsInf node");
        };
        let input = Type::from(n.inputs.first().unwrap());
        let output = Type::from(n.outputs.first().unwrap());
        Self::new(input, output, n.config.clone())
    }
}
