use super::prelude::*;

impl NodeCodegen for onnx_ir::expand::ExpandNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();

        let input = scope.arg(input_arg);
        let output = arg_to_ident(output_arg);
        let output_rank = output_arg.ty.rank();

        // For scalar inputs, convert to rank-1 tensor
        let (input_init, input_expr, input_rank) = if input_arg.ty.is_scalar() {
            let dtype = input_arg.ty.elem_type();
            let dtype_tokens = dtype.to_tokens();
            let kind = match dtype {
                DType::Bool => quote! { , Bool },
                _ if dtype.is_float() => quote! {},
                _ => quote! { , Int },
            };
            let init = quote! {
                let input = Tensor::<B, 1 #kind>::from_data_dtype(
                    burn::tensor::TensorData::from([#input]),
                    &*self.device,
                    #dtype_tokens
                );
            };
            (init, quote! { input }, 1usize)
        } else {
            (quote! {}, input.clone(), input_arg.ty.rank())
        };

        let shape = match &self.config {
            onnx_ir::expand::ExpandConfig::Static(s) => s.to_tokens(),
            onnx_ir::expand::ExpandConfig::Runtime(r) => {
                let shape_arg = &self.inputs[r.input_index];
                match &shape_arg.ty {
                    ArgType::Tensor(_) => {
                        let name = arg_to_ident(shape_arg);
                        quote! {
                            TryInto::<[i64; #output_rank]>::try_into(
                                #name.to_data().convert::<i64>().as_slice().unwrap()
                            ).unwrap()
                        }
                    }
                    ArgType::Shape(_) => {
                        let name = arg_to_ident(shape_arg);
                        quote! { #name }
                    }
                    _ => panic!("Invalid shape source {:?}", shape_arg.ty),
                }
            }
        };

        // ONNX Expand uses max-semantics: output_dim = max(input_dim, shape_dim)
        quote! {
            let #output = {
                #input_init
                let onnx_shape: [i64; #output_rank] = #shape;
                let input_dims = #input_expr.dims();
                let mut shape = onnx_shape;
                #[allow(clippy::needless_range_loop)]
                for i in 0..#input_rank {
                    let dim_offset = #output_rank - #input_rank + i;
                    if shape[dim_offset] == 1 && input_dims[i] > 1 {
                        shape[dim_offset] = input_dims[i] as i64;
                    }
                }
                #input_expr.expand(shape)
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::expand::{ExpandConfig, ExpandNode, ExpandNodeBuilder};

    fn create_expand_node_static(name: &str, input_rank: usize, shape: Vec<i64>) -> ExpandNode {
        let output_rank = shape.len();
        let config = ExpandConfig::Static(shape);

        ExpandNodeBuilder::new(name)
            .input_tensor("input", input_rank, DType::F32)
            .output_tensor("output", output_rank, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_expand_static() {
        let node = create_expand_node_static("expand1", 2, vec![2, 3, 4]);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 3> {
            let output = {
                let onnx_shape: [i64; 3usize] = [2, 3, 4];
                let input_dims = input.dims();
                let mut shape = onnx_shape;
                #[allow(clippy::needless_range_loop)]
                for i in 0..2usize {
                    let dim_offset = 3usize - 2usize + i;
                    if shape[dim_offset] == 1 && input_dims[i] > 1 {
                        shape[dim_offset] = input_dims[i] as i64;
                    }
                }
                input.expand(shape)
            };
            output
        }
        ");
    }

    #[test]
    fn test_expand_broadcast() {
        let node = create_expand_node_static("expand1", 2, vec![1, 5, 10]);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 3> {
            let output = {
                let onnx_shape: [i64; 3usize] = [1, 5, 10];
                let input_dims = input.dims();
                let mut shape = onnx_shape;
                #[allow(clippy::needless_range_loop)]
                for i in 0..2usize {
                    let dim_offset = 3usize - 2usize + i;
                    if shape[dim_offset] == 1 && input_dims[i] > 1 {
                        shape[dim_offset] = input_dims[i] as i64;
                    }
                }
                input.expand(shape)
            };
            output
        }
        ");
    }

    #[test]
    fn test_expand_scalar_int64() {
        let config = ExpandConfig::Static(vec![2, 3]);
        let node = ExpandNodeBuilder::new("expand_scalar")
            .input_scalar("input", DType::I64)
            .output_tensor("output", 2, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: i64) -> Tensor<B, 2, Int> {
            let output = {
                let input = Tensor::<
                    B,
                    1,
                    Int,
                >::from_data_dtype(
                    burn::tensor::TensorData::from([input]),
                    &*self.device,
                    burn::tensor::DType::I64,
                );
                let onnx_shape: [i64; 2usize] = [2, 3];
                let input_dims = input.dims();
                let mut shape = onnx_shape;
                #[allow(clippy::needless_range_loop)]
                for i in 0..1usize {
                    let dim_offset = 2usize - 1usize + i;
                    if shape[dim_offset] == 1 && input_dims[i] > 1 {
                        shape[dim_offset] = input_dims[i] as i64;
                    }
                }
                input.expand(shape)
            };
            output
        }
        ");
    }

    #[test]
    fn test_expand_scalar_f32() {
        let config = ExpandConfig::Static(vec![2, 3]);
        let node = ExpandNodeBuilder::new("expand_scalar")
            .input_scalar("input", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: f32) -> Tensor<B, 2> {
            let output = {
                let input = Tensor::<
                    B,
                    1,
                >::from_data_dtype(
                    burn::tensor::TensorData::from([input]),
                    &*self.device,
                    burn::tensor::DType::F32,
                );
                let onnx_shape: [i64; 2usize] = [2, 3];
                let input_dims = input.dims();
                let mut shape = onnx_shape;
                #[allow(clippy::needless_range_loop)]
                for i in 0..1usize {
                    let dim_offset = 2usize - 1usize + i;
                    if shape[dim_offset] == 1 && input_dims[i] > 1 {
                        shape[dim_offset] = input_dims[i] as i64;
                    }
                }
                input.expand(shape)
            };
            output
        }
        ");
    }

    #[test]
    fn test_expand_scalar_bool() {
        let config = ExpandConfig::Static(vec![2, 3]);
        let node = ExpandNodeBuilder::new("expand_scalar")
            .input_scalar("input", DType::Bool)
            .output_tensor("output", 2, DType::Bool)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: bool) -> Tensor<B, 2, Bool> {
            let output = {
                let input = Tensor::<
                    B,
                    1,
                    Bool,
                >::from_data_dtype(
                    burn::tensor::TensorData::from([input]),
                    &*self.device,
                    burn::tensor::DType::Bool,
                );
                let onnx_shape: [i64; 2usize] = [2, 3];
                let input_dims = input.dims();
                let mut shape = onnx_shape;
                #[allow(clippy::needless_range_loop)]
                for i in 0..1usize {
                    let dim_offset = 2usize - 1usize + i;
                    if shape[dim_offset] == 1 && input_dims[i] > 1 {
                        shape[dim_offset] = input_dims[i] as i64;
                    }
                }
                input.expand(shape)
            };
            output
        }
        ");
    }
}
