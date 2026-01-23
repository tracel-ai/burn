use super::prelude::*;

impl NodeCodegen for onnx_ir::node::arithmetic::AddNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let lhs_arg = self.inputs.first().unwrap();
        let rhs_arg = self.inputs.get(1).unwrap();
        let output = arg_to_ident(self.outputs.first().unwrap());

        let lhs = scope.arg(lhs_arg);

        let rhs = scope.arg(rhs_arg);

        let function = match (&lhs_arg.ty, &rhs_arg.ty) {
            (ArgType::Tensor(lhs_tensor), ArgType::Tensor(rhs_tensor)) => {
                let lhs_rank = lhs_tensor.rank;
                let rhs_rank = rhs_tensor.rank;

                if lhs_rank == rhs_rank {
                    quote! { #lhs.add(#rhs) }
                } else if lhs_rank > rhs_rank {
                    let num_dims = lhs_rank - rhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.add(#rhs.unsqueeze_dims(&[#(#dims),*])) }
                } else {
                    let num_dims = rhs_rank - lhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.unsqueeze_dims(&[#(#dims),*]).add(#rhs) }
                }
            }
            (ArgType::Tensor(_), ArgType::Scalar(_)) => quote! { #lhs.add_scalar(#rhs) },
            (ArgType::Scalar(_), ArgType::Tensor(_)) => quote! { #rhs.add_scalar(#lhs) },
            (ArgType::Scalar(_), ArgType::Scalar(_)) => quote! { #lhs + #rhs },
            (ArgType::Shape(_), ArgType::Shape(_)) => quote! {
                {
                    let mut result = #lhs;
                    for (result_item, rhs_item) in result.iter_mut().zip(#rhs.iter()) {
                        *result_item = result_item.saturating_add(*rhs_item);
                    }
                    result
                }
            },
            (ArgType::Shape(_), ArgType::Scalar(_)) => quote! {
                {
                    let mut result = #lhs;
                    for result_item in result.iter_mut() {
                        *result_item = result_item.saturating_add(#rhs as i64);
                    }
                    result
                }
            },
            (ArgType::Scalar(_), ArgType::Shape(_)) => quote! {
                {
                    let mut result = #rhs;
                    for result_item in result.iter_mut() {
                        *result_item = result_item.saturating_add(#lhs as i64);
                    }
                    result
                }
            },
            (ArgType::Shape(_), ArgType::Tensor(tensor_type)) => {
                // Use from_data_dtype to ensure the dtype matches the tensor's dtype,
                // not the backend's default IntElem type
                let dtype_tokens = tensor_type.dtype.to_tokens();
                quote! {
                    Tensor::<B, 1, burn::tensor::Int>::from_data_dtype(
                        burn::tensor::TensorData::from(&#lhs as &[i64]),
                        &*self.device,
                        #dtype_tokens
                    ).add(#rhs)
                }
            }
            (ArgType::Tensor(tensor_type), ArgType::Shape(_)) => {
                // Use from_data_dtype to ensure the dtype matches the tensor's dtype
                let dtype_tokens = tensor_type.dtype.to_tokens();
                quote! {
                    #lhs.add(Tensor::<B, 1, burn::tensor::Int>::from_data_dtype(
                        burn::tensor::TensorData::from(&#rhs as &[i64]),
                        &*self.device,
                        #dtype_tokens
                    ))
                }
            }
        };

        quote! {
            let #output = #function;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::arithmetic::{AddNode, AddNodeBuilder};

    fn create_add_node_tensor_tensor(name: &str, lhs_rank: usize, rhs_rank: usize) -> AddNode {
        AddNodeBuilder::new(name)
            .input_tensor("lhs", lhs_rank, DType::F32)
            .input_tensor("rhs", rhs_rank, DType::F32)
            .output_tensor("output", lhs_rank.max(rhs_rank), DType::F32)
            .build()
    }

    fn create_add_node_tensor_scalar(name: &str) -> AddNode {
        AddNodeBuilder::new(name)
            .input_tensor("lhs", 2, DType::F32)
            .input_scalar("rhs", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build()
    }

    #[test]
    fn test_add_forward_tensor_tensor() {
        let node = create_add_node_tensor_tensor("add1", 2, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<B, 2>, rhs: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = lhs.add(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_add_forward_tensor_scalar() {
        let node = create_add_node_tensor_scalar("add1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<B, 2>, rhs: f32) -> Tensor<B, 2> {
            let output = lhs.add_scalar(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_add_forward_broadcast() {
        let node = create_add_node_tensor_tensor("add1", 3, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<B, 3>, rhs: Tensor<B, 2>) -> Tensor<B, 3> {
            let output = lhs.add(rhs.unsqueeze_dims(&[0isize]));
            output
        }
        ");
    }
}
