use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[allow(clippy::too_many_arguments)]
#[derive(Debug, Clone, new)]
pub struct GemmNode {
    pub a: TensorType,
    pub b: TensorType,
    pub c: Option<TensorType>,
    pub output: TensorType,
    pub alpha: f32,
    pub beta: f32,
    pub trans_a: i64,
    pub trans_b: i64,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for GemmNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        let mut inputs = vec![Type::Tensor(self.a.clone()), Type::Tensor(self.b.clone())];

        if let Some(ref c) = self.c {
            inputs.push(Type::Tensor(c.clone()));
        }

        inputs
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let a = scope.tensor_use_owned(&self.a, node_position);
        let b = scope.tensor_use_owned(&self.b, node_position);

        let output = &self.output.name;
        let alpha = self.alpha;
        let beta = self.beta;
        let trans_a = self.trans_a;
        let trans_b = self.trans_b;

        let a = if trans_a != 0 {
            quote! {#a.transpose()}
        } else {
            quote! {#a}
        };

        let b = if trans_b != 0 {
            quote! {#b.transpose()}
        } else {
            quote! {#b}
        };

        let product = quote! {#a.matmul(#b)};
        let scaled_product = quote! {#product * #alpha};

        if let Some(ref c) = self.c {
            let c = scope.tensor_use_owned(c, node_position);

            quote! {
                let #output = (#scaled_product) + (#c * #beta);
            }
        } else {
            quote! {
                let #output = #scaled_product;
            }
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Gemm(self)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        graph::BurnGraph,
        node::{gemm::GemmNode, test::assert_tokens},
        TensorType,
    };

    #[test]
    fn test_codegen_nodes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(GemmNode::new(
            TensorType::new_float("tensor1", 2),
            TensorType::new_float("tensor2", 2),
            None,
            TensorType::new_float("tensor3", 2),
            1.0,
            1.0,
            0,
            0,
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
        );

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

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
                    "hello"
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
