use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct ReduceMeanNode {
    pub input: TensorType,
    pub output: TensorType,
    pub axes: Option<Vec<i64>>,
    pub keepdims: bool,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ReduceMeanNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        match &self.axes {
            Some(axes) if !axes.is_empty() => {
                // Convert axes to usize and sort in descending order to avoid index shifting when squeezing
                let mut dims: Vec<usize> = axes.iter().map(|&axis| axis as usize).collect();
                dims.sort_by(|a, b| b.cmp(a));

                let dims_tokens: Vec<_> = dims.iter().map(|d| d.to_tokens()).collect();
                let dims_isize: Vec<isize> = dims.iter().map(|&d| d as isize).collect();

                // Apply mean_dim for each axis
                let mut result = quote! { #input };
                for dim in &dims_tokens {
                    result = quote! { #result.mean_dim(#dim) };
                }

                if self.keepdims {
                    quote! {
                        let #output = #result;
                    }
                } else {
                    // Apply squeeze_dims to remove reduced dimensions
                    quote! {
                        let #output = #result.squeeze_dims(&[#(#dims_isize),*]);
                    }
                }
            }
            _ => {
                // Reduce all dimensions - results in scalar
                quote! {
                    let #output = #input.mean();
                }
            }
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::ReduceMean(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::graph::BurnGraph;
    use crate::burn::node::test::assert_tokens;
    use burn::record::FullPrecisionSettings;
    use quote::quote;

    fn one_node_graph(
        node: ReduceMeanNode,
        expected: TokenStream,
        inputs: Vec<String>,
        outputs: Vec<String>,
    ) {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(node);
        graph.register_input_output(inputs, outputs);

        let actual = graph.codegen();
        assert_tokens(actual, expected);
    }

    #[test]
    fn test_reduce_mean_single_axis_keepdims() {
        one_node_graph(
            ReduceMeanNode::new(
                TensorType::new_float("tensor1", 4),
                TensorType::new_float("tensor2", 4),
                Some(vec![1]),
                true,
            ),
            quote! {
                use burn::tensor::Tensor;
                use burn::{
                    module::Module,
                    tensor::backend::Backend,
                };

                #[derive(Module, Debug)]
                pub struct Model<B: Backend> {
                    phantom: core::marker::PhantomData<B>,
                    device: burn::module::Ignored<B::Device>,
                }

                impl<B: Backend> Model <B> {
                    #[allow(unused_variables)]
                    pub fn new(device: &B::Device) -> Self {
                        Self {
                            phantom: core::marker::PhantomData,
                            device: burn::module::Ignored(device.clone()),
                        }
                    }

                    #[allow(clippy::let_and_return, clippy::approx_constant)]
                    pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                        let tensor2 = tensor1.mean_dim(1);

                        tensor2
                    }
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }

    #[test]
    fn test_reduce_mean_single_axis_no_keepdims() {
        one_node_graph(
            ReduceMeanNode::new(
                TensorType::new_float("tensor1", 4),
                TensorType::new_float("tensor2", 3),
                Some(vec![1]),
                false,
            ),
            quote! {
                use burn::tensor::Tensor;
                use burn::{
                    module::Module,
                    tensor::backend::Backend,
                };

                #[derive(Module, Debug)]
                pub struct Model<B: Backend> {
                    phantom: core::marker::PhantomData<B>,
                    device: burn::module::Ignored<B::Device>,
                }

                impl<B: Backend> Model <B> {
                    #[allow(unused_variables)]
                    pub fn new(device: &B::Device) -> Self {
                        Self {
                            phantom: core::marker::PhantomData,
                            device: burn::module::Ignored(device.clone()),
                        }
                    }

                    #[allow(clippy::let_and_return, clippy::approx_constant)]
                    pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 3> {
                        let tensor2 = tensor1.mean_dim(1).squeeze_dims(&[1isize]);

                        tensor2
                    }
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }

    #[test]
    fn test_reduce_mean_multi_axes_keepdims() {
        one_node_graph(
            ReduceMeanNode::new(
                TensorType::new_float("tensor1", 4),
                TensorType::new_float("tensor2", 4),
                Some(vec![1, 2]),
                true,
            ),
            quote! {
                use burn::tensor::Tensor;
                use burn::{
                    module::Module,
                    tensor::backend::Backend,
                };

                #[derive(Module, Debug)]
                pub struct Model<B: Backend> {
                    phantom: core::marker::PhantomData<B>,
                    device: burn::module::Ignored<B::Device>,
                }

                impl<B: Backend> Model <B> {
                    #[allow(unused_variables)]
                    pub fn new(device: &B::Device) -> Self {
                        Self {
                            phantom: core::marker::PhantomData,
                            device: burn::module::Ignored(device.clone()),
                        }
                    }

                    #[allow(clippy::let_and_return, clippy::approx_constant)]
                    pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                        let tensor2 = tensor1.mean_dim(2).mean_dim(1);

                        tensor2
                    }
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }

    #[test]
    fn test_reduce_mean_multi_axes_no_keepdims() {
        one_node_graph(
            ReduceMeanNode::new(
                TensorType::new_float("tensor1", 4),
                TensorType::new_float("tensor2", 2),
                Some(vec![1, 2]),
                false,
            ),
            quote! {
                use burn::tensor::Tensor;
                use burn::{
                    module::Module,
                    tensor::backend::Backend,
                };

                #[derive(Module, Debug)]
                pub struct Model<B: Backend> {
                    phantom: core::marker::PhantomData<B>,
                    device: burn::module::Ignored<B::Device>,
                }

                impl<B: Backend> Model <B> {
                    #[allow(unused_variables)]
                    pub fn new(device: &B::Device) -> Self {
                        Self {
                            phantom: core::marker::PhantomData,
                            device: burn::module::Ignored(device.clone()),
                        }
                    }

                    #[allow(clippy::let_and_return, clippy::approx_constant)]
                    pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 2> {
                        let tensor2 = tensor1.mean_dim(2).mean_dim(1).squeeze_dims(&[2isize, 1isize]);

                        tensor2
                    }
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }

    #[test]
    fn test_reduce_mean_all_dims() {
        one_node_graph(
            ReduceMeanNode::new(
                TensorType::new_float("tensor1", 4),
                TensorType::new_float("tensor2", 1),
                None,
                false,
            ),
            quote! {
                use burn::tensor::Tensor;
                use burn::{
                    module::Module,
                    tensor::backend::Backend,
                };

                #[derive(Module, Debug)]
                pub struct Model<B: Backend> {
                    phantom: core::marker::PhantomData<B>,
                    device: burn::module::Ignored<B::Device>,
                }

                impl<B: Backend> Model <B> {
                    #[allow(unused_variables)]
                    pub fn new(device: &B::Device) -> Self {
                        Self {
                            phantom: core::marker::PhantomData,
                            device: burn::module::Ignored(device.clone()),
                        }
                    }

                    #[allow(clippy::let_and_return, clippy::approx_constant)]
                    pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 1> {
                        let tensor2 = tensor1.mean();

                        tensor2
                    }
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }
}
