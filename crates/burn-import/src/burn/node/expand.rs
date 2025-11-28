use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::expand::ExpandNode {
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

        let input_rank = input_arg.ty.rank();
        let output_rank = output_arg.ty.rank();

        let shape = match &self.config {
            onnx_ir::expand::ExpandConfig::Static(static_shape) => static_shape.to_tokens(),
            onnx_ir::expand::ExpandConfig::Runtime(shape_ref) => {
                let shape_arg = &self.inputs[shape_ref.input_index];
                match &shape_arg.ty {
                    ArgType::Tensor(_) => {
                        let tensor_name = arg_to_ident(shape_arg);
                        quote! {
                            TryInto::<[i64; #output_rank]>::try_into(#tensor_name.to_data().convert::<i64>().as_slice().unwrap()).unwrap()
                        }
                    }
                    ArgType::Shape(_) => {
                        let shape_name = arg_to_ident(shape_arg);
                        quote! { #shape_name }
                    }
                    _ => panic!("Invalid shape source {:?}", shape_arg.ty),
                }
            }
        };

        // ONNX Expand uses max-semantics: output_dim = max(input_dim, shape_dim)
        // When shape_dim == 1 but input_dim > 1, ONNX keeps the input_dim.
        // Burn's expand uses -1 to mean "keep input dim".
        // We transform the shape by replacing 1 with -1 when input_dim > 1.
        quote! {
            let #output = {
                let onnx_shape: [i64; #output_rank] = #shape;
                let input_dims = #input.dims();
                let mut burn_shape = onnx_shape;
                for i in 0..#input_rank {
                    let dim_offset = #output_rank - #input_rank + i;
                    if burn_shape[dim_offset] == 1 && input_dims[i] > 1 {
                        burn_shape[dim_offset] = -1;
                    }
                }
                #input.expand(burn_shape)
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
                let mut burn_shape = onnx_shape;
                for i in 0..2usize {
                    let dim_offset = 3usize - 2usize + i;
                    if burn_shape[dim_offset] == 1 && input_dims[i] > 1 {
                        burn_shape[dim_offset] = -1;
                    }
                }
                input.expand(burn_shape)
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
                let mut burn_shape = onnx_shape;
                for i in 0..2usize {
                    let dim_offset = 3usize - 2usize + i;
                    if burn_shape[dim_offset] == 1 && input_dims[i] > 1 {
                        burn_shape[dim_offset] = -1;
                    }
                }
                input.expand(burn_shape)
            };
            output
        }
        ");
    }
}
