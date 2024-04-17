use super::{Node, NodeCodegen};
use crate::burn::{BurnImports, TensorType, Type};

use burn::record::PrecisionSettings;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct WhereNode {
    /// Bool tensor. When True (nonzero), yield X, otherwise yield Y.
    pub condition: TensorType,
    /// Values selected at indices where condition is True.
    pub x: TensorType,
    /// Values selected at indices where condition is False.
    pub y: TensorType,
    pub output: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for WhereNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<crate::burn::Type> {
        vec![
            Type::Tensor(self.condition.clone()),
            Type::Tensor(self.x.clone()),
            Type::Tensor(self.y.clone()),
        ]
    }

    fn forward(
        &self,
        scope: &mut crate::burn::Scope,
        node_position: usize,
    ) -> proc_macro2::TokenStream {
        let mask = scope.tensor_use_owned(&self.condition, node_position);
        let x = scope.tensor_use_owned(&self.x, node_position);
        let y = scope.tensor_use_owned(&self.y, node_position);
        let output = &self.output.name;

        quote! {
            let #output = #y.mask_where(#mask, #x);
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::tensor::Bool");
    }

    fn into_node(self) -> super::Node<PS> {
        Node::Where(self)
    }
}

#[cfg(test)]
mod tests {

    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        graph::BurnGraph,
        node::{mask_where::WhereNode, test::assert_tokens},
        TensorType,
    };

    #[test]
    fn test_codegen_where() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(WhereNode::new(
            TensorType::new_bool("tensor1", 2),
            TensorType::new_float("tensor2", 2),
            TensorType::new_float("tensor3", 2),
            TensorType::new_float("tensor4", 2),
        ));

        graph.register_input_output(
            vec![
                "tensor1".to_string(),
                "tensor2".to_string(),
                "tensor3".to_string(),
            ],
            vec!["tensor4".to_string()],
        );

        let expected = quote! {
            use burn::tensor::Bool;
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                    }
                }

                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(
                    &self,
                    tensor1: Tensor<B, 2, Bool>,
                    tensor2: Tensor<B, 2>,
                    tensor3: Tensor<B, 2>
                ) -> Tensor<B, 2> {
                    let tensor4 = tensor3.mask_where(tensor1, tensor2);

                    tensor4
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
