use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::reshape::ReshapeNode {
    fn inputs(&self) -> &[Argument] {
        // Reshape has input tensor and shape argument
        // Filter to include dynamic and constant inputs
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();
        let output = arg_to_ident(output_arg);

        // Determine if we have static or runtime shape
        match &self.config.shape {
            onnx_ir::reshape::ReshapeInput::Static(shape_values) => {
                // Static shape - simple reshape
                use onnx_ir::ir::ArgType;
                match &input_arg.ty {
                    ArgType::Tensor(_) => {
                        let input = scope.arg(input_arg);

                        // Check if output is a scalar
                        match &output_arg.ty {
                            ArgType::Scalar(elem_type) => {
                                use onnx_ir::ir::DType;
                                let elem_cast = match elem_type {
                                    DType::F32 => quote! { .elem::<f32>() },
                                    DType::F64 => quote! { .elem::<f64>() },
                                    DType::I32 => quote! { .elem::<i32>() },
                                    DType::I64 => quote! { .elem::<i64>() },
                                    DType::Bool => quote! { .elem::<bool>() },
                                    _ => panic!("Unsupported scalar type: {:?}", elem_type),
                                };
                                quote! {
                                    let #output = #input.into_scalar()#elem_cast;
                                }
                            }
                            ArgType::Tensor(_) => {
                                let shape_values = shape_values.to_tokens();
                                quote! {
                                    let #output = #input.reshape(#shape_values);
                                }
                            }
                            ArgType::Shape(_) => {
                                panic!("Tensor to Shape reshape not supported")
                            }
                        }
                    }
                    ArgType::Shape(input_rank) => {
                        // Shape input path
                        let input_name = arg_to_ident(input_arg);

                        match &output_arg.ty {
                            ArgType::Scalar(elem_type) => {
                                if *input_rank != 1 {
                                    panic!(
                                        "Shape to scalar requires Shape(1), got Shape({})",
                                        input_rank
                                    );
                                }
                                use onnx_ir::ir::DType;
                                let cast_expr = match elem_type {
                                    DType::I64 => quote! { #input_name[0] as i64 },
                                    DType::I32 => quote! { #input_name[0] as i32 },
                                    _ => panic!(
                                        "Shape to Scalar only supports Int32/Int64 output types"
                                    ),
                                };
                                quote! {
                                    let #output = #cast_expr;
                                }
                            }
                            ArgType::Shape(output_rank) => {
                                if input_rank == output_rank {
                                    quote! {
                                        let #output = #input_name;
                                    }
                                } else {
                                    quote! {
                                        let #output: [i64; #output_rank] = {
                                            let mut result = [0i64; #output_rank];
                                            let copy_len = #input_rank.min(#output_rank);
                                            result[..copy_len].copy_from_slice(&#input_name[..copy_len]);
                                            result
                                        };
                                    }
                                }
                            }
                            ArgType::Tensor(_) => {
                                // Convert Shape to Tensor first, then reshape
                                let shape_values = shape_values.to_tokens();
                                quote! {
                                    let #output = {
                                        let shape_array = #input_name as [i64; #input_rank];
                                        Tensor::<B, 1, Int>::from_data(
                                            TensorData::from(shape_array),
                                            &self.device
                                        )
                                    }.reshape(#shape_values);
                                }
                            }
                        }
                    }
                    ArgType::Scalar(_) => {
                        panic!("Reshape: unexpected scalar input")
                    }
                }
            }
            onnx_ir::reshape::ReshapeInput::Runtime(shape_ref) => {
                // Runtime shape - need to extract shape from second input
                let shape_arg = &self.inputs[shape_ref.input_index];
                use onnx_ir::ir::ArgType;

                let input = scope.arg(input_arg);

                match &shape_arg.ty {
                    ArgType::Shape(_) => {
                        let shape_name = arg_to_ident(shape_arg);
                        quote! {
                            let #output = #input.reshape(#shape_name);
                        }
                    }
                    ArgType::Tensor(_) => {
                        let shape_name = arg_to_ident(shape_arg);
                        let output_rank = match &output_arg.ty {
                            ArgType::Tensor(t) => t.rank,
                            _ => panic!("Runtime reshape with tensor shape expects tensor output"),
                        };
                        let array_init = (0..output_rank)
                            .map(|i| {
                                let idx = proc_macro2::Literal::usize_unsuffixed(i);
                                quote! { shape_array[#idx] as usize }
                            })
                            .collect::<Vec<_>>();

                        quote! {
                            let shape_data = #shape_name.to_data();
                            let shape_array = shape_data.as_slice::<i64>().unwrap();
                            let #output = #input.reshape([#(#array_init),*]);
                        }
                    }
                    ArgType::Scalar(_) => {
                        panic!("Reshape: shape argument cannot be scalar")
                    }
                }
            }
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        // Check if we need TensorData for shape-to-tensor conversion
        match &self.inputs.first().unwrap().ty {
            onnx_ir::ir::ArgType::Shape(_) => match &self.outputs.first().unwrap().ty {
                onnx_ir::ir::ArgType::Tensor(_) => {
                    imports.register("burn::tensor::TensorData");
                }
                _ => {}
            },
            _ => {}
        }
    }
}
