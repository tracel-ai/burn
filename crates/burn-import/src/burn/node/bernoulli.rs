use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorKind, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct BernoulliNode {
    pub input: TensorType,
    pub output: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for BernoulliNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn register_imports(&self, imports: &mut crate::burn::BurnImports) {
        imports.register("burn::tensor::Distribution");
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let output = &self.output.name;
        let input = scope.tensor_use_owned(&self.input, node_position);
        let dist = quote! { Distribution::Default };

        let input_random = quote! { #input.random_like(#dist).lower(#input) };
        let output_random = match self.output.kind {
            TensorKind::Bool => input_random,
            TensorKind::Int => quote! { #input_random.int() },
            TensorKind::Float => quote! { #input_random.float() },
        };

        quote! { let #output = #output_random; }
    }

    fn into_node(self) -> Node<PS> {
        Node::Bernoulli(self)
    }
}

impl OnnxIntoNode for BernoulliNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Bernoulli(n) = node else {
            panic!("Expected Bernoulli node");
        };
        let input = crate::burn::TensorType::from(n.inputs.first().unwrap());
        let output = crate::burn::TensorType::from(n.outputs.first().unwrap());
        Self::new(input, output)
    }
}
