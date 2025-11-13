use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use onnx_ir::node::eye_like::EyeLikeConfig;
use proc_macro2::TokenStream;
use quote::{ToTokens, quote};

#[derive(Debug, Clone, new)]
pub struct EyeLikeNode {
    pub input: TensorType,
    pub output: TensorType,
    pub config: EyeLikeConfig,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for EyeLikeNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let k_offset = self.config.k.to_token_stream();

        // Convert mask to appropriate type based on output tensor kind
        let conversion = match self.output.kind {
            crate::burn::TensorKind::Int => quote! { .int() },
            crate::burn::TensorKind::Float => quote! { .float() },
            crate::burn::TensorKind::Bool => quote! {},
        };

        // Use diag_mask to create the diagonal matrix, then invert it
        // diag_mask returns false on diagonal, true off-diagonal
        // EyeLike needs true on diagonal, false off-diagonal
        quote! {
            let #output = Tensor::diag_mask(#input.shape(), #k_offset, &*self.device).bool_not()#conversion;
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::EyeLike(self)
    }
}

impl OnnxIntoNode for EyeLikeNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let input = TensorType::from(node.inputs().first().unwrap());
        let output = TensorType::from(node.outputs().first().unwrap());
        let config = match &node {
            onnx_ir::ir::Node::EyeLike { config, .. } => config,
            _ => panic!("Expected EyeLike node"),
        };
        Self::new(input, output, config.clone())
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{eye_like::EyeLikeNode, test::assert_tokens},
    };
    use onnx_ir::node::eye_like::EyeLikeConfig;

    #[test]
    fn test_codegen_nodes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        let config = EyeLikeConfig { dtype: None, k: 0 };

        graph.register(EyeLikeNode::new(
            TensorType::new_float("tensor1", 2),
            TensorType::new_float("tensor2", 2),
            config,
        ));

        graph.register_input_output(
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
            &[],
            &[],
        );

        let expected = quote! {
            use burn::prelude::*;

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, tensor1: Tensor<B, 2>) -> Tensor<B, 2> {
                    let tensor2 = Tensor::diag_mask(tensor1.shape(), 0i64, &*self.device)
                        .bool_not()
                        .float();
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
