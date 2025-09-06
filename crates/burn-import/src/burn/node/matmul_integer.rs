use core::cmp::Ordering;

use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorKind, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

/// ONNX MatMulInteger: (A - a_zp) @ (B - b_zp) -> int32
#[derive(Debug, Clone)]
pub struct MatMulIntegerNode {
    pub lhs: TensorType,                    // u8 or i8
    pub rhs: TensorType,                    // u8 or i8
    pub lhs_zero_point: Option<TensorType>, // optional zp
    pub rhs_zero_point: Option<TensorType>, // optional zp
    pub output: TensorType,                 // i32
}

impl MatMulIntegerNode {
    pub fn new(
        lhs: TensorType,
        rhs: TensorType,
        lhs_zero_point: Option<TensorType>,
        rhs_zero_point: Option<TensorType>,
        output: TensorType,
    ) -> Self {
        if lhs.kind != TensorKind::Int || rhs.kind != TensorKind::Int {
            panic!("MatMulInteger expects integer tensors (u8/i8) for lhs/rhs");
        }
        if output.kind != TensorKind::Int {
            panic!("MatMulInteger output must be int32");
        }
        Self {
            lhs,
            rhs,
            lhs_zero_point,
            rhs_zero_point,
            output,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for MatMulIntegerNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        let mut v = vec![
            Type::Tensor(self.lhs.clone()),
            Type::Tensor(self.rhs.clone()),
        ];
        if let Some(zp) = &self.lhs_zero_point {
            v.push(Type::Tensor(zp.clone()));
        }
        if let Some(zp) = &self.rhs_zero_point {
            v.push(Type::Tensor(zp.clone()));
        }
        v
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let lhs = scope.tensor_use_owned(&self.lhs, node_position);
        let rhs = scope.tensor_use_owned(&self.rhs, node_position);
        let out = &self.output.name;

        let lhs_dim = self.lhs.rank;
        let rhs_dim = self.rhs.rank;

        // Zero-points: default = zeros_like(same shape)
        let a_zp = if let Some(zp) = &self.lhs_zero_point {
            scope.tensor_use_owned(zp, node_position)
        } else {
            quote! { Tensor::<B, #lhs_dim>::zeros_like(&#lhs) }
        };
        let b_zp = if let Some(zp) = &self.rhs_zero_point {
            scope.tensor_use_owned(zp, node_position)
        } else {
            quote! { Tensor::<B, #rhs_dim>::zeros_like(&#rhs) }
        };

        let lhs_c = quote! { (#lhs).to_dtype(DType::Int32).sub((#a_zp).to_dtype(DType::Int32)) };
        let rhs_c = quote! { (#rhs).to_dtype(DType::Int32).sub((#b_zp).to_dtype(DType::Int32)) };

        match lhs_dim.cmp(&rhs_dim) {
            Ordering::Greater => {
                if rhs_dim == 1 {
                    let squeeze_dim = lhs_dim - 1;
                    let out_rank = self.output.rank;
                    let mut unsqueeze_dims = vec![-1isize];
                    let num_unsqueezes = lhs_dim - rhs_dim;
                    if num_unsqueezes > 1 {
                        unsqueeze_dims.extend(std::iter::repeat_n(0isize, num_unsqueezes - 1));
                    }
                    quote! {
                        let rhs_c = (#rhs_c).unsqueeze_dims(&[#(#unsqueeze_dims),*]);
                        let #out = (#lhs_c).matmul(rhs_c).squeeze::<#out_rank>(#squeeze_dim);
                    }
                } else {
                    let target_rank = lhs_dim;
                    quote! {
                        let rhs_c = (#rhs_c).unsqueeze::<#target_rank>();
                        let #out = (#lhs_c).matmul(rhs_c);
                    }
                }
            }
            Ordering::Less => {
                if lhs_dim == 1 {
                    let squeeze_dim = rhs_dim - 2;
                    let out_rank = self.output.rank;
                    let target_rank = rhs_dim;
                    quote! {
                        let lhs_c = (#lhs_c).unsqueeze::<#target_rank>();
                        let #out = lhs_c.matmul(#rhs_c).squeeze::<#out_rank>(#squeeze_dim);
                    }
                } else {
                    let target_rank = rhs_dim;
                    quote! {
                        let lhs_c = (#lhs_c).unsqueeze::<#target_rank>();
                        let #out = lhs_c.matmul(#rhs_c);
                    }
                }
            }
            Ordering::Equal => quote! {
                let #out = (#lhs_c).matmul(#rhs_c);
            },
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::MatmulInteger(self)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::node::test::assert_tokens;
    use crate::burn::{TensorType, graph::BurnGraph};
    use burn::record::FullPrecisionSettings;
    use quote::quote;

    #[test]
    fn codegen_basic_no_zp() {
        let mut g = BurnGraph::<FullPrecisionSettings>::default();
        g.register(MatMulIntegerNode::new(
            TensorType::new_int("a", 2),
            TensorType::new_int("b", 2),
            None,
            None,
            TensorType::new_int("y", 2),
        ));
        g.register_input_output(vec!["a".into(), "b".into()], vec!["y".into()]);

        let expected = quote! {
            use burn::prelude::*;
            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }
            impl<B: Backend> Model<B> {
                pub fn new(device: &B::Device) -> Self {
                    Self { phantom: core::marker::PhantomData, device: burn::module::Ignored(device.clone()) }
                }
                pub fn forward(&self, a: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 2> {
                    let y = a
                        .to_dtype(DType::Int32)
                        .sub(Tensor::<B, 2>::zeros_like(&a).to_dtype(DType::Int32))
                        .matmul(b.to_dtype(DType::Int32).sub(Tensor::<B, 2>::zeros_like(&b).to_dtype(DType::Int32)));
                    y
                }
            }
        };
    }
}
