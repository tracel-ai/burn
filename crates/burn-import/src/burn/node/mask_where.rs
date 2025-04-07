use super::{Node, NodeCodegen};
use crate::burn::{BurnImports, ScalarType, ToTokens, Type};

use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct WhereNode {
    /// Bool tensor. When True (nonzero), yield X, otherwise yield Y.
    pub condition: Type,
    /// Values selected at indices where condition is True.
    pub x: Type,
    /// Values selected at indices where condition is False.
    pub y: Type,
    pub output: Type,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for WhereNode {
    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn input_types(&self) -> Vec<crate::burn::Type> {
        vec![self.condition.clone(), self.x.clone(), self.y.clone()]
    }

    fn forward(&self, scope: &mut crate::burn::Scope, node_position: usize) -> TokenStream {
        match &self.output {
            Type::Tensor(out) => {
                let cond = Self::input_as_tensor(&self.condition, out.rank, scope, node_position);
                let y = Self::input_as_tensor(&self.y, out.rank, scope, node_position);
                let out_id = &out.name;

                if let Type::Scalar(x) = &self.x {
                    let x = &x.name;
                    quote! {
                        let #out_id = #y.mask_fill(#cond, #x);
                    }
                } else {
                    let x = Self::input_as_tensor(&self.x, out.rank, scope, node_position);
                    quote! {
                        let #out_id = #y.mask_where(#cond, #x);
                    }
                }
            }
            Type::Scalar(out) => {
                // Scalar out means all inputs are scalars as well:
                let cond = self.condition.as_scalar();
                let x = self.x.as_scalar();
                let y = self.y.as_scalar();
                Self::forward_scalar(out, cond, x, y)
            }
            other => panic!("Where cannot handle {other:?}"),
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        if matches!(&self.output, Type::Tensor(_)) {
            imports.register("burn::tensor::Bool");
        }
    }

    fn into_node(self) -> super::Node<PS> {
        Node::Where(self)
    }
}

impl WhereNode {
    fn forward_scalar(
        out: &ScalarType,
        cond: &ScalarType,
        x: &ScalarType,
        y: &ScalarType,
    ) -> TokenStream {
        let out_name = &out.name;
        let out_type = out.ty();
        let cond_name = &cond.name;
        let x_name = &x.name;
        let y_name = &y.name;

        quote! {
            let #out_name : #out_type = if #cond_name {
                #x_name
            }
            else {
                #y_name
            };
        }
    }

    fn input_as_tensor(
        input: &Type,
        broadcast_rank: usize,
        scope: &mut crate::burn::Scope,
        node_position: usize,
    ) -> TokenStream {
        let (tensor, rank) = match input {
            Type::Tensor(t) => (scope.tensor_use_owned(t, node_position), t.rank),
            Type::Scalar(s) => (s.to_full_tensor(&vec![1; broadcast_rank]), broadcast_rank),
            Type::Shape(s) => (s.to_tensor(), 1),
            _ => panic!("Where op: {input:?} input not implemented"),
        };
        if rank < broadcast_rank {
            let broadcast_rank_tokens = broadcast_rank.to_tokens();
            quote! { #tensor.unsqueeze::<#broadcast_rank_tokens>()}
        } else {
            tensor
        }
    }
}

#[cfg(test)]
mod tests {

    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        ScalarKind, TensorType,
        graph::BurnGraph,
        node::{mask_where::WhereNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_where() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(WhereNode::new(
            Type::Tensor(TensorType::new_bool("tensor1", 2)),
            Type::Tensor(TensorType::new_float("tensor2", 2)),
            Type::Tensor(TensorType::new_float("tensor3", 2)),
            Type::Tensor(TensorType::new_float("tensor4", 2)),
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

    #[test]
    fn test_codegen_where_broadcasted() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(WhereNode::new(
            Type::Tensor(TensorType::new_bool("tensor1", 4)),
            Type::Tensor(TensorType::new_float("tensor2", 2)),
            Type::Tensor(TensorType::new_float("tensor3", 3)),
            Type::Tensor(TensorType::new_float("tensor4", 4)),
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
                pub fn forward(
                    &self,
                    tensor1: Tensor<B, 4, Bool>,
                    tensor2: Tensor<B, 2>,
                    tensor3: Tensor<B, 3>
                ) -> Tensor<B, 4> {
                    let tensor4 = tensor3
                        .unsqueeze::<4>()
                        .mask_where(tensor1, tensor2.unsqueeze::<4>());

                    tensor4
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_where_scalar_x() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(WhereNode::new(
            Type::Tensor(TensorType::new_bool("tensor1", 2)),
            Type::Scalar(ScalarType::new("scalar2", ScalarKind::Float64)),
            Type::Tensor(TensorType::new_float("tensor3", 2)),
            Type::Tensor(TensorType::new_float("tensor4", 2)),
        ));

        graph.register_input_output(
            vec![
                "tensor1".to_string(),
                "scalar2".to_string(),
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
                pub fn forward(
                    &self,
                    tensor1: Tensor<B, 2, Bool>,
                    scalar2: f64,
                    tensor3: Tensor<B, 2>
                ) -> Tensor<B, 2> {
                    let tensor4 = tensor3.mask_fill(tensor1, scalar2);

                    tensor4
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_where_scalar_y() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(WhereNode::new(
            Type::Tensor(TensorType::new_bool("tensor1", 2)),
            Type::Tensor(TensorType::new_float("tensor2", 2)),
            Type::Scalar(ScalarType::new("scalar3", ScalarKind::Float64)),
            Type::Tensor(TensorType::new_float("tensor4", 2)),
        ));

        graph.register_input_output(
            vec![
                "tensor1".to_string(),
                "tensor2".to_string(),
                "scalar3".to_string(),
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
                pub fn forward(
                    &self,
                    tensor1: Tensor<B, 2, Bool>,
                    tensor2: Tensor<B, 2>,
                    scalar3: f64
                ) -> Tensor<B, 2> {
                    let tensor4 = Tensor::<B, 2, burn::tensor::Float>::full([1, 1], scalar3, &*self.device)
                        .mask_where(tensor1, tensor2);

                    tensor4
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_where_all_scalar() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(WhereNode::new(
            Type::Scalar(ScalarType::new("scalar1", ScalarKind::Bool)),
            Type::Scalar(ScalarType::new("scalar2", ScalarKind::Float64)),
            Type::Scalar(ScalarType::new("scalar3", ScalarKind::Float64)),
            Type::Scalar(ScalarType::new("scalar4", ScalarKind::Float64)),
        ));

        graph.register_input_output(
            vec![
                "scalar1".to_string(),
                "scalar2".to_string(),
                "scalar3".to_string(),
            ],
            vec!["scalar4".to_string()],
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

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }

                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(
                    &self,
                    scalar1: bool,
                    scalar2: f64,
                    scalar3: f64
                ) -> f64 {
                    let scalar4: f64 = if scalar1 { scalar2 } else { scalar3 };

                    scalar4
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
