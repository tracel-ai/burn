use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::expand::ExpandNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        let output_rank = match &self.outputs.first().unwrap().ty {
            ArgType::Tensor(tensor) => tensor.rank,
            _ => panic!("Expand output must be a tensor"),
        };

        let shape = match &self.config {
            onnx_ir::expand::ExpandConfig::Static(static_shape) => static_shape.to_tokens(),
            onnx_ir::expand::ExpandConfig::Runtime(shape_ref) => {
                // Get the actual argument using the RuntimeInputRef
                let shape_arg = &self.inputs[shape_ref.input_index];
                match &shape_arg.ty {
                    ArgType::Tensor(_) => {
                        let tensor_name = arg_to_ident(shape_arg);
                        // Convert to i64 for `AsIndex`
                        quote! {
                            TryInto::<[i64; #output_rank]>::try_into(#tensor_name.to_data().convert::<i64>().as_slice().unwrap()).unwrap()
                        }
                    }
                    ArgType::Shape(_) => {
                        // Shape arrays are [i64; N] and expand now accepts them directly via Element trait
                        let shape_name = arg_to_ident(shape_arg);
                        quote! { #shape_name }
                    }
                    _ => panic!("Invalid shape source {:?}", shape_arg.ty),
                }
            }
        };

        quote! {
            let #output = #input.expand(#shape);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::expand::{ExpandConfig, ExpandNode, ExpandNodeBuilder};

    fn create_expand_node_static(name: &str, shape: Vec<i64>) -> ExpandNode {
        let output_rank = shape.len();
        let config = ExpandConfig::Static(shape);

        ExpandNodeBuilder::new(name)
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", output_rank, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_expand_static() {
        let node = create_expand_node_static("expand1", vec![2, 3, 4]);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 3> {
            let output = input.expand([2, 3, 4]);
            output
        }
        ");
    }

    #[test]
    fn test_expand_broadcast() {
        let node = create_expand_node_static("expand1", vec![1, 5, 10]);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 3> {
            let output = input.expand([1, 5, 10]);
            output
        }
        ");
    }
}
