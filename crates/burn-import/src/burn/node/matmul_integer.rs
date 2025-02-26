use core::cmp::Ordering;

use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorKind, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone)]
pub struct MatMulIntegerNode {
    pub lhs: TensorType,
    pub rhs: TensorType,
    pub output: TensorType,
    pub a_zero_point: Option<TensorType>,
    pub b_zero_point: Option<TensorType>,
}

impl MatMulIntegerNode {
    pub fn new(
        lhs: TensorType,
        rhs: TensorType,
        output: TensorType,
        a_zero_point: Option<TensorType>,
        b_zero_point: Option<TensorType>,
    ) -> Self {
        // Validate tensor types - using Int for quantized tensors
        if lhs.kind != TensorKind::Int || rhs.kind != TensorKind::Int {
            panic!("MatMulInteger is only implemented for integer tensors");
        }

        // Output is typically an Int32 tensor in ONNX
        if output.kind != TensorKind::Int {
            panic!("MatMulInteger output must be an integer tensor");
        }

        // Validate zero points if provided
        if let Some(a_zero) = &a_zero_point {
            if a_zero.kind != TensorKind::Int {
                panic!("A zero point must be an integer tensor");
            }
        }

        if let Some(b_zero) = &b_zero_point {
            if b_zero.kind != TensorKind::Int {
                panic!("B zero point must be an integer tensor");
            }
        }

        Self {
            lhs,
            rhs,
            output,
            a_zero_point,
            b_zero_point,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for MatMulIntegerNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        let mut input_types = vec![
            Type::Tensor(self.lhs.clone()),
            Type::Tensor(self.rhs.clone()),
        ];
        if let Some(a_zero_point) = &self.a_zero_point {
            input_types.push(Type::Tensor(a_zero_point.clone()));
        }
        if let Some(b_zero_point) = &self.b_zero_point {
            input_types.push(Type::Tensor(b_zero_point.clone()));
        }
        input_types
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let lhs = scope.tensor_use_owned(&self.lhs, node_position);
        let rhs = scope.tensor_use_owned(&self.rhs, node_position);
        let output = &self.output.name;

        let a_zero_point = if let Some(a_zero_point) = &self.a_zero_point {
            scope.tensor_use_owned(a_zero_point, node_position)
        } else {
            quote! { 0 }
        };

        let b_zero_point = if let Some(b_zero_point) = &self.b_zero_point {
            scope.tensor_use_owned(b_zero_point, node_position)
        } else {
            quote! { 0 }
        };

        let lhs_dim = self.lhs.dim;
        let rhs_dim = self.rhs.dim;

        // Support broadcasting for missing dimensions
        match lhs_dim.cmp(&rhs_dim) {
            Ordering::Greater => {
                let axes = (0..lhs_dim - rhs_dim)
                    .map(|i| if i % 2 == 0 { 0 } else { -1 })
                    .collect::<Vec<i64>>();
                let axes = axes.to_tokens();

                if rhs_dim == 1 {
                    let squeeze_dim = lhs_dim - 1;
                    quote! {
                        let #output = (#lhs - #a_zero_point).matmul((#rhs.unsqueeze_dims(&#axes) - #b_zero_point)).squeeze(#squeeze_dim);
                    }
                } else {
                    quote! {
                        let #output = (#lhs - #a_zero_point).matmul((#rhs.unsqueeze_dims(&#axes) - #b_zero_point));
                    }
                }
            }
            Ordering::Less => {
                let axes = [0i64].repeat(rhs_dim - lhs_dim).to_tokens();

                if lhs_dim == 1 {
                    let squeeze_dim = rhs_dim - 2;
                    quote! {
                        let #output = (#lhs.unsqueeze_dims(&#axes) - #a_zero_point).matmul((#rhs - #b_zero_point)).squeeze(#squeeze_dim);
                    }
                } else {
                    quote! {
                        let #output = (#lhs.unsqueeze_dims(&#axes) - #a_zero_point).matmul((#rhs - #b_zero_point));
                    }
                }
            }
            Ordering::Equal => quote! {
                let #output = (#lhs - #a_zero_point).matmul((#rhs - #b_zero_point));
            },
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::MatmulInteger(self)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        graph::BurnGraph,
        node::{matmul_integer::MatMulIntegerNode, test::assert_tokens},
        TensorType,
    };

    #[test]
    fn test_codegen_matmul_integer() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(MatMulIntegerNode::new(
            TensorType::new_int("tensor1", 4),
            TensorType::new_int("tensor2", 4),
            TensorType::new_int("tensor3", 4),
            Some(TensorType::new_int("a_zero_point", 1)),
            Some(TensorType::new_int("b_zero_point", 1)),
        ));

        graph.register_input_output(
            vec![
                "tensor1".to_string(),
                "tensor2".to_string(),
                "a_zero_point".to_string(),
                "b_zero_point".to_string(),
            ],
            vec!["tensor3".to_string()],
        );

        let expected = quote! {
            use burn::tensor::Int;
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
                pub fn forward(
                    &self,
                    tensor1: Tensor<B, 4, Int>,
                    tensor2: Tensor<B, 4, Int>,
                    a_zero_point: Tensor<B, 1, Int>,
                    b_zero_point: Tensor<B, 1, Int>,
                ) -> Tensor<B, 4, Int> {
                    let tensor3 = (tensor1 - a_zero_point).matmul((tensor2 - b_zero_point));
                    tensor3
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
