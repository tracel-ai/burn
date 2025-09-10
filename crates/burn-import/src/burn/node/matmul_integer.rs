use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorKind, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::Ident;
use proc_macro2::TokenStream;
use quote::quote;
use syn::parse_str;

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

        let out: Ident = parse_str(&self.output.name.to_string()).expect("Valid Rust identifier");

        let lhs_dim = self.lhs.rank;
        let rhs_dim = self.rhs.rank;

        // ---- Zero-points: synthesize when missing, otherwise lift to input rank ----
        let a_zp_raw: TokenStream = if let Some(zp) = &self.lhs_zero_point {
            scope.tensor_use_owned(zp, node_position)
        } else {
            quote! { Tensor::zeros_like(&#lhs) }
        };
        let b_zp_raw: TokenStream = if let Some(zp) = &self.rhs_zero_point {
            scope.tensor_use_owned(zp, node_position)
        } else {
            quote! { Tensor::zeros_like(&#rhs) }
        };

        // If a ZP is provided (scalar or 1-D), unsqueeze it to the input rank so `sub` has matching rank.
        let a_zp = if self.lhs_zero_point.is_some() && lhs_dim > 1 {
            let tr = lhs_dim;
            quote! { (#a_zp_raw).unsqueeze::<#tr>() }
        } else {
            quote! { #a_zp_raw }
        };
        let b_zp = if self.rhs_zero_point.is_some() && rhs_dim > 1 {
            let tr = rhs_dim;
            quote! { (#b_zp_raw).unsqueeze::<#tr>() }
        } else {
            quote! { #b_zp_raw }
        };

        // Centered inputs (already Int tensors)
        let lhs_c = quote! { ((#lhs).sub(#a_zp)) };
        let rhs_c = quote! { ((#rhs).sub(#b_zp)) };

        // ---- Rank handling for matmul broadcasting ----
        match lhs_dim.cmp(&rhs_dim) {
            core::cmp::Ordering::Greater => {
                if rhs_dim == 1 {
                    // (…, M, K) @ (K) -> (…, M)
                    let squeeze_dim = lhs_dim - 1;
                    let out_rank = self.output.rank;
                    let target_rank = lhs_dim;
                    quote! {
                        let rhs_c = (#rhs_c).unsqueeze::<#target_rank>();
                        let #out = (#lhs_c).matmul(rhs_c).squeeze::<#out_rank>(#squeeze_dim);
                    }
                } else {
                    // Broadcast rhs leading dims up to lhs rank
                    let target_rank = lhs_dim;
                    quote! {
                        let rhs_c = (#rhs_c).unsqueeze::<#target_rank>();
                        let #out = (#lhs_c).matmul(rhs_c);
                    }
                }
            }
            core::cmp::Ordering::Less => {
                if lhs_dim == 1 {
                    // (K) @ (…, K, N) -> (…, N)
                    let squeeze_dim = rhs_dim - 2;
                    let out_rank = self.output.rank;
                    let target_rank = rhs_dim;
                    quote! {
                        let lhs_c = (#lhs_c).unsqueeze::<#target_rank>();
                        let #out = lhs_c.matmul(#rhs_c).squeeze::<#out_rank>(#squeeze_dim);
                    }
                } else {
                    // Broadcast lhs leading dims up to rhs rank
                    let target_rank = rhs_dim;
                    quote! {
                        let lhs_c = (#lhs_c).unsqueeze::<#target_rank>();
                        let #out = lhs_c.matmul(#rhs_c);
                    }
                }
            }
            core::cmp::Ordering::Equal => quote! {
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

        let _expected = quote! {
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
                pub fn forward(&self, a: Tensor<B, 2, Int>, b: Tensor<B, 2, Int>) -> Tensor<B, 2, Int> {
                    let y = a.int()
                        .sub(Tensor::zeros_like(&a).int())
                        .matmul(b.int().sub(Tensor::zeros_like(&b).int()));
                    y
                }
            }
        };
    }
}
