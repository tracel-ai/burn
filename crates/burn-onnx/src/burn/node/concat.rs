use super::prelude::*;

impl NodeCodegen for onnx_ir::concat::ConcatNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());
        let dim = self.config.axis.to_tokens();

        // Check if any inputs are scalars
        let has_scalar = self.inputs.iter().any(|arg| arg.ty.is_scalar());

        // Determine if this is tensor or shape concatenation based on output type
        match &self.outputs.first().unwrap().ty {
            ArgType::Tensor(_) if has_scalar => {
                // Mixed scalar/tensor concatenation - convert scalars to rank-1 tensors first
                let mut inits = Vec::new();
                let mut input_exprs = Vec::new();

                for (i, input_arg) in self.inputs.iter().enumerate() {
                    let input = scope.arg(input_arg);

                    if input_arg.ty.is_scalar() {
                        // Convert scalar to rank-1 tensor
                        let dtype = input_arg.ty.elem_type();
                        let dtype_tokens = dtype.to_tokens();
                        let kind = match dtype {
                            DType::Bool => quote! { , Bool },
                            _ if dtype.is_float() => quote! {},
                            _ => quote! { , Int },
                        };
                        let temp_name =
                            Ident::new(&format!("scalar_as_tensor_{}", i), Span::call_site());
                        let init = quote! {
                            let #temp_name: Tensor<B, 1 #kind> = Tensor::from_data_dtype(
                                burn::tensor::TensorData::from([#input]),
                                &*self.device,
                                #dtype_tokens
                            );
                        };
                        inits.push(init);
                        input_exprs.push(quote! { #temp_name });
                    } else {
                        input_exprs.push(input);
                    }
                }

                quote! {
                    #(#inits)*
                    let #output = burn::tensor::Tensor::cat([#(#input_exprs),*].into(), #dim);
                }
            }
            ArgType::Tensor(_) => {
                // Tensor concatenation (no scalars)
                let inputs = self.inputs.iter().map(|arg| scope.arg(arg));

                quote! {
                    let #output = burn::tensor::Tensor::cat([#(#inputs),*].into(), #dim);
                }
            }
            ArgType::Shape(shape) => {
                // Shape concatenation - shapes are 1D so concat is always on axis 0
                if self.config.axis != 0 {
                    panic!(
                        "Shape concatenation only supports dim=0, got dim={}",
                        self.config.axis
                    );
                }
                let output_rank = shape;

                // Generate code to concatenate shape arrays
                let mut shape_parts = Vec::new();
                for input in &self.inputs {
                    let input_name = arg_to_ident(input);
                    shape_parts.push(quote! { &#input_name[..] });
                }

                quote! {
                    let #output: [i64; #output_rank] = [#(#shape_parts),*].concat().try_into().unwrap();
                }
            }
            _ => panic!("Concat only supports Tensor or Shape outputs"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::concat::{ConcatConfig, ConcatNode, ConcatNodeBuilder};

    fn create_concat_node(name: &str, num_inputs: usize, axis: usize) -> ConcatNode {
        let config = ConcatConfig { axis };
        let mut builder = ConcatNodeBuilder::new(name);

        for i in 0..num_inputs {
            builder = builder.input_tensor(&format!("input{}", i), 2, DType::F32);
        }

        builder
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_concat_two_tensors() {
        let node = create_concat_node("concat1", 2, 0);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input0: Tensor<B, 2>, input1: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = burn::tensor::Tensor::cat([input0, input1].into(), 0);
            output
        }
        ");
    }

    #[test]
    fn test_concat_three_tensors() {
        let node = create_concat_node("concat1", 3, 1);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input0: Tensor<B, 2>,
            input1: Tensor<B, 2>,
            input2: Tensor<B, 2>,
        ) -> Tensor<B, 2> {
            let output = burn::tensor::Tensor::cat([input0, input1, input2].into(), 1);
            output
        }
        ");
    }
}
