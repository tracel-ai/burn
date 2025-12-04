use super::prelude::*;

impl NodeCodegen for onnx_ir::unsqueeze::UnsqueezeNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        use onnx_ir::ir::ArgType;

        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();
        let output = arg_to_ident(output_arg);

        // Generate axes token stream
        let axes = match &self.config {
            onnx_ir::unsqueeze::UnsqueezeConfig::Static(static_axes) => static_axes.to_tokens(),
            onnx_ir::unsqueeze::UnsqueezeConfig::Runtime(axes_ref) => {
                let axes_arg = &self.inputs[axes_ref.input_index];
                match &axes_arg.ty {
                    ArgType::Tensor(_) => {
                        let tensor_name = arg_to_ident(axes_arg);
                        quote! {
                            #tensor_name.to_data().as_slice::<B::IntElem>().unwrap().iter().map(|&x| x.to_isize()).collect::<Vec<isize>>()
                        }
                    }
                    _ => panic!(
                        "UnsqueezeNode received invalid axes type: expected tensor but got {:?}",
                        axes_arg.ty
                    ),
                }
            }
        };

        match (&input_arg.ty, &output_arg.ty) {
            (ArgType::Tensor(_input_tensor), ArgType::Tensor(output_tensor)) => {
                let input = scope.arg(input_arg);
                let output_rank = output_tensor.rank.to_tokens();

                // Generate the correct output type based on the tensor kind
                let output_type = match &output_tensor.dtype {
                    dtype if dtype.is_int() || dtype.is_uint() => {
                        quote! { Tensor<B, #output_rank, Int> }
                    }
                    dtype if dtype.is_float() => {
                        quote! { Tensor<B, #output_rank> }
                    }
                    dtype if dtype.is_bool() => {
                        quote! { Tensor<B, #output_rank, Bool> }
                    }
                    _ => panic!("Unsupported tensor dtype: {:?}", output_tensor.dtype),
                };

                quote! {
                    let #output: #output_type = #input.unsqueeze_dims::<#output_rank>(&#axes);
                }
            }
            (ArgType::Scalar(_scalar_type), ArgType::Tensor(output_tensor)) => {
                let scalar_name = arg_to_ident(input_arg);
                let output_rank = output_tensor.rank.to_tokens();

                // Determine the element type based on the output tensor type
                let tensor_creation = match &output_tensor.dtype {
                    dtype if dtype.is_int() || dtype.is_uint() => {
                        let elem_conversion = quote! { #scalar_name.elem::<B::IntElem>() };
                        quote! { Tensor::<B, #output_rank, Int>::from_data([#elem_conversion], &self.device).unsqueeze() }
                    }
                    dtype if dtype.is_float() => {
                        let elem_conversion = quote! { #scalar_name.elem::<B::FloatElem>() };
                        quote! { Tensor::<B, #output_rank>::from_data([#elem_conversion], &self.device).unsqueeze() }
                    }
                    dtype if dtype.is_bool() => {
                        let elem_conversion = quote! { #scalar_name != 0 };
                        quote! { Tensor::<B, #output_rank, Bool>::from_data([#elem_conversion], &self.device).unsqueeze() }
                    }
                    _ => panic!("Unsupported tensor dtype: {:?}", output_tensor.dtype),
                };

                quote! {
                    let #output = #tensor_creation;
                }
            }
            (ArgType::Scalar(scalar_type), ArgType::Shape(_)) => {
                // Scalar(Int) -> Shape[1] conversion
                let input_name = arg_to_ident(input_arg);

                use onnx_ir::ir::DType;
                let value_expr = match scalar_type {
                    DType::I64 => quote! { #input_name },
                    DType::I32 => quote! { #input_name as i64 },
                    _ => panic!(
                        "Unsqueeze from Scalar to Shape only supports Int32/Int64 input types, but got: {:?}",
                        scalar_type
                    ),
                };

                quote! {
                    let #output = [#value_expr];
                }
            }
            _ => panic!(
                "UnsqueezeNode received unsupported input/output combination: {:?} -> {:?}",
                input_arg.ty, output_arg.ty
            ),
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        match &self.config {
            onnx_ir::unsqueeze::UnsqueezeConfig::Runtime(_) => {
                imports.register("alloc::vec::Vec");
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::unsqueeze::{UnsqueezeConfig, UnsqueezeNode, UnsqueezeNodeBuilder};

    fn create_unsqueeze_node(name: &str, axes: Vec<i64>) -> UnsqueezeNode {
        let config = UnsqueezeConfig::Static(axes);

        UnsqueezeNodeBuilder::new(name)
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_unsqueeze_forward_single_axis() {
        let node = create_unsqueeze_node("unsqueeze1", vec![0]);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 3> {
            let output: Tensor<B, 3> = input.unsqueeze_dims::<3>(&[0]);
            output
        }
        ");
    }

    #[test]
    fn test_unsqueeze_forward_axis_1() {
        let node = create_unsqueeze_node("unsqueeze1", vec![1]);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 3> {
            let output: Tensor<B, 3> = input.unsqueeze_dims::<3>(&[1]);
            output
        }
        ");
    }

    #[test]
    fn test_unsqueeze_forward_axis_2() {
        let node = create_unsqueeze_node("unsqueeze1", vec![2]);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 3> {
            let output: Tensor<B, 3> = input.unsqueeze_dims::<3>(&[2]);
            output
        }
        ");
    }
}
