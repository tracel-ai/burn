use super::prelude::*;

impl NodeCodegen for onnx_ir::split::SplitNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let axis = self.config.axis.to_tokens();

        let outputs = self.outputs.iter().map(arg_to_ident).collect::<Vec<_>>();

        let unpack_outputs = quote! {
            let [#(#outputs),*] = split_tensors.try_into().unwrap();
        };

        if let Some(split_sizes_input) = &self.config.split_sizes {
            // Extract static split sizes from the enum wrapper
            let split_sizes = match split_sizes_input {
                onnx_ir::split::SplitSizesInput::Static(sizes) => sizes,
                onnx_ir::split::SplitSizesInput::Runtime(_) => {
                    panic!("Runtime split sizes are not supported in burn-import")
                }
            };
            let split_sizes_tokens = split_sizes.iter().map(|s| s.to_tokens());
            quote! {
                let split_tensors = #input.split_with_sizes(vec![#(#split_sizes_tokens),*], #axis);
                #unpack_outputs
            }
        } else if let Some(split_size) = &self.config.split_size {
            let split_size_tokens = split_size.to_tokens();
            quote! {
                let split_tensors = #input.split(#split_size_tokens, #axis);
                #unpack_outputs
            }
        } else if let Some(num_outputs) = &self.config.num_outputs {
            // Runtime split size calculation: dim_size / num_outputs
            let num_outputs_tokens = num_outputs.to_tokens();
            quote! {
                let split_size = #input.dims()[#axis] / #num_outputs_tokens;
                let split_tensors = #input.split(split_size, #axis);
                #unpack_outputs
            }
        } else {
            panic!("Split node must have either split_size, split_sizes, or num_outputs")
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        // When split_sizes is used, we generate vec![...] which needs the vec macro
        if self.config.split_sizes.is_some() {
            imports.register("alloc::vec");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::split::{SplitConfig, SplitNodeBuilder, SplitSizesInput};

    #[test]
    fn test_split_equal() {
        let config = SplitConfig {
            axis: 0,
            split_size: Some(2),
            split_sizes: None,
            num_outputs: None,
        };
        let node = SplitNodeBuilder::new("split1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output0", 2, DType::F32)
            .output_tensor("output1", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
            let split_tensors = input.split(2, 0);
            let [output0, output1] = split_tensors.try_into().unwrap();
            (output0, output1)
        }
        ");
    }

    #[test]
    fn test_split_sizes() {
        let config = SplitConfig {
            axis: 1,
            split_size: None,
            split_sizes: Some(SplitSizesInput::Static(vec![1, 3, 2])),
            num_outputs: None,
        };
        let node = SplitNodeBuilder::new("split1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output0", 2, DType::F32)
            .output_tensor("output1", 2, DType::F32)
            .output_tensor("output2", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 2>,
        ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
            let split_tensors = input.split_with_sizes(vec![1, 3, 2], 1);
            let [output0, output1, output2] = split_tensors.try_into().unwrap();
            (output0, output1, output2)
        }
        ");
    }

    #[test]
    fn test_split_runtime_num_outputs() {
        // Test runtime split size calculation using num_outputs
        let config = SplitConfig {
            axis: 2,
            split_size: None,
            split_sizes: None,
            num_outputs: Some(3),
        };
        let node = SplitNodeBuilder::new("split1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output0", 3, DType::F32)
            .output_tensor("output1", 3, DType::F32)
            .output_tensor("output2", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 3>,
        ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
            let split_size = input.dims()[2] / 3;
            let split_tensors = input.split(split_size, 2);
            let [output0, output1, output2] = split_tensors.try_into().unwrap();
            (output0, output1, output2)
        }
        ");
    }
}
