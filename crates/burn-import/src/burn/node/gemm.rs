use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct GemmNode {
    pub A: TensorType,
    pub B: TensorType,
    pub C: Option<TensorType>,
    pub output: TensorType,
    pub alpha: f32,
    pub beta: f32,
    pub transA: i64,
    pub transB: i64,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for GemmNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        let mut inputs = vec![Type::Tensor(self.A.clone()), Type::Tensor(self.B.clone())];

        if let Some(ref c) = self.C {
            inputs.push(Type::Tensor(c.clone()));
        }

        inputs
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let a = scope.tensor_use_owned(&self.A, node_position);
        let b = scope.tensor_use_owned(&self.B, node_position);

        let output = &self.output.name;
        let alpha = self.alpha;
        let beta = self.beta;
        let trans_a = self.transA;
        let trans_b = self.transB;

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
        let scaled_product = quote! {#alpha * #product};

        let final_expr = if let Some(ref c) = self.C {
            let c = scope.tensor_use_owned(c, node_position);

            quote! {
                let #output = (#scaled_product) + (#beta * #c);
            }
        } else {
            quote! {
                let #output = #scaled_product;
            }
        };

        final_expr
    }

    fn into_node(self) -> Node<PS> {
        Node::Gemm(self)
    }
}
