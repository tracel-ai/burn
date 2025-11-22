use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[allow(clippy::too_many_arguments)]
#[derive(Debug, Clone, new)]
pub struct GemmNode {
    pub a: TensorType,
    pub b: TensorType,
    pub c: Option<Type>,
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
            match c {
                Type::Tensor(tensor) => inputs.push(Type::Tensor(tensor.clone())),
                Type::Scalar(scalar) => inputs.push(Type::Scalar(scalar.clone())),
                _ => panic!("C should be Tensor or Scalar!"),
            }
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

        let scaled_product = match alpha {
            1.0 => product,
            _ => quote! {#product * #alpha},
        };

        if let Some(ref c) = self.c {
            match (c, beta) {
                (Type::Tensor(tensor), 1.0) => {
                    let c_tensor = scope.tensor_use_owned(tensor, node_position);
                    quote! {
                        let #output = #scaled_product + #c_tensor.unsqueeze();
                    }
                }
                (Type::Scalar(scalar), 1.0) => {
                    let c_scalar = &scalar.name;
                    quote! {
                        let #output = #scaled_product + #c_scalar;
                    }
                }
                (Type::Tensor(tensor), _) => {
                    let c_tensor = scope.tensor_use_owned(tensor, node_position);
                    quote! {
                        let #output = #scaled_product + (#c_tensor.unsqueeze() * #beta);
                    }
                }
                (Type::Scalar(scalar), _) => {
                    let c_scalar = &scalar.name;
                    quote! {
                        let #output = #scaled_product + (#c_scalar * #beta);
                    }
                }
                _ => panic!("C should be Tensor or a Scalar!"),
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

impl OnnxIntoNode for GemmNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Gemm(n) = node else {
            panic!("Expected Gemm node");
        };
        let a = TensorType::from(n.inputs.first().unwrap());
        let b = TensorType::from(n.inputs.get(1).unwrap());
        let c = n.inputs.get(2).map(Type::from);
        let output = TensorType::from(n.outputs.first().unwrap());
        Self::new(
            a,
            b,
            c,
            output,
            n.config.alpha,
            n.config.beta,
            n.config.trans_a,
            n.config.trans_b,
        )
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        ScalarKind, ScalarType, TensorType,
        graph::BurnGraph,
        node::{gemm::GemmNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_nodes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(GemmNode::new(
            TensorType::new_float("tensor1", 2),
            TensorType::new_float("tensor2", 2),
            Some(Type::Scalar(ScalarType::new(
                "scalar1",
                ScalarKind::Float32,
            ))),
            TensorType::new_float("tensor3", 2),
            1.0,
            1.0,
            0,
            0,
        ));

        graph.register_input_output(
            vec![
                "tensor1".to_string(),
                "tensor2".to_string(),
                "scalar1".to_string(),
            ],
            vec!["tensor3".to_string()],
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
                pub fn forward(&self, tensor1: Tensor<B, 2>, tensor2: Tensor<B, 2>, scalar1: f32) -> Tensor<B, 2> {
                    let tensor3 = tensor1.matmul(tensor2) + scalar1;
                    tensor3
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
    #[test]
    fn test_codegen_non_unit_alpha_beta() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(GemmNode::new(
            TensorType::new_float("tensor1", 2),
            TensorType::new_float("tensor2", 2),
            Some(Type::Scalar(ScalarType::new(
                "scalar1",
                ScalarKind::Float32,
            ))),
            TensorType::new_float("tensor3", 2),
            0.5,
            0.5,
            0,
            0,
        ));

        graph.register_input_output(
            vec![
                "tensor1".to_string(),
                "tensor2".to_string(),
                "scalar1".to_string(),
            ],
            vec!["tensor3".to_string()],
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
                pub fn forward(&self, tensor1: Tensor<B, 2>, tensor2: Tensor<B, 2>, scalar1: f32) -> Tensor<B, 2> {
                    let tensor3 = tensor1.matmul(tensor2) * 0.5f32 + (scalar1 * 0.5f32);
                    tensor3
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
    #[test]
    fn test_codegen_no_c() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(GemmNode::new(
            TensorType::new_float("tensor1", 2),
            TensorType::new_float("tensor2", 2),
            None,
            TensorType::new_float("tensor3", 2),
            1.,
            1.,
            0,
            0,
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
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
                pub fn forward(&self, tensor1: Tensor<B, 2>, tensor2: Tensor<B, 2>) -> Tensor<B, 2> {
                    let tensor3 = tensor1.matmul(tensor2);
                    tensor3
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
