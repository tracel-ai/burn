use super::prelude::*;

impl NodeCodegen for onnx_ir::reshape::ReshapeNode {
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
                    ArgType::Scalar(elem_type) => {
                        // Scalar input - convert scalar to tensor then reshape
                        let input_name = arg_to_ident(input_arg);

                        match &output_arg.ty {
                            ArgType::Tensor(tensor_type) => {
                                use onnx_ir::ir::DType;
                                let shape_values = shape_values.to_tokens();
                                let output_rank = tensor_type.rank;
                                // Create a tensor with the output rank directly from the scalar
                                // We use TensorData::from([scalar]) to create a 1-element tensor,
                                // then reshape to the target shape
                                match elem_type {
                                    DType::F32 | DType::F64 => {
                                        quote! {
                                            let #output = Tensor::<B, #output_rank>::from_data(
                                                TensorData::from([#input_name]).convert::<f32>(),
                                                &self.device
                                            ).reshape(#shape_values);
                                        }
                                    }
                                    DType::I32 | DType::I64 => {
                                        quote! {
                                            let #output = Tensor::<B, #output_rank, Int>::from_data(
                                                TensorData::from([#input_name]),
                                                &self.device
                                            ).reshape(#shape_values);
                                        }
                                    }
                                    DType::Bool => {
                                        quote! {
                                            let #output = Tensor::<B, #output_rank, Bool>::from_data(
                                                TensorData::from([#input_name]),
                                                &self.device
                                            ).reshape(#shape_values);
                                        }
                                    }
                                    _ => panic!(
                                        "Reshape: unsupported scalar type {:?}",
                                        elem_type
                                    ),
                                }
                            }
                            ArgType::Scalar(_) => {
                                // Scalar to scalar - just pass through
                                quote! {
                                    let #output = #input_name;
                                }
                            }
                            _ => panic!("Reshape: scalar input to {:?} not supported", output_arg.ty),
                        }
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

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::ir::RuntimeInputRef;
    use onnx_ir::reshape::{ReshapeConfig, ReshapeInput, ReshapeNodeBuilder};

    // Static Tensor -> Tensor reshapes
    #[test]
    fn test_reshape_static_tensor_to_tensor() {
        let config = ReshapeConfig {
            shape: ReshapeInput::Static(vec![2, 3]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("data", 3, DType::F32)
            .output_tensor("reshaped", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<B, 3>) -> Tensor<B, 2> {
            let reshaped = data.reshape([2, 3]);
            reshaped
        }
        ");
    }

    #[test]
    fn test_reshape_static_with_neg_one() {
        let config = ReshapeConfig {
            shape: ReshapeInput::Static(vec![2, -1]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("tensor", 3, DType::F32)
            .output_tensor("result", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, tensor: Tensor<B, 3>) -> Tensor<B, 2> {
            let result = tensor.reshape([2, -1]);
            result
        }
        ");
    }

    #[test]
    fn test_reshape_3d_to_1d() {
        let config = ReshapeConfig {
            shape: ReshapeInput::Static(vec![-1]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("flattened", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 1> {
            let flattened = input.reshape([-1]);
            flattened
        }
        ");
    }

    // Static Tensor -> Scalar (all scalar types)
    #[test]
    fn test_reshape_tensor_to_scalar_f32() {
        let config = ReshapeConfig {
            shape: ReshapeInput::Static(vec![]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("tensor", 1, DType::F32)
            .output_scalar("scalar", DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, tensor: Tensor<B, 1>) -> f32 {
            let scalar = tensor.into_scalar().elem::<f32>();
            scalar
        }
        ");
    }

    #[test]
    fn test_reshape_tensor_to_scalar_f64() {
        let config = ReshapeConfig {
            shape: ReshapeInput::Static(vec![]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("input", 1, DType::F64)
            .output_scalar("value", DType::F64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 1>) -> f64 {
            let value = input.into_scalar().elem::<f64>();
            value
        }
        ");
    }

    #[test]
    fn test_reshape_tensor_to_scalar_i32() {
        let config = ReshapeConfig {
            shape: ReshapeInput::Static(vec![]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("data", 1, DType::I32)
            .output_scalar("int_val", DType::I32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<B, 1, Int>) -> i32 {
            let int_val = data.into_scalar().elem::<i32>();
            int_val
        }
        ");
    }

    #[test]
    fn test_reshape_tensor_to_scalar_i64() {
        let config = ReshapeConfig {
            shape: ReshapeInput::Static(vec![]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("input", 1, DType::I64)
            .output_scalar("long_val", DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 1, Int>) -> i64 {
            let long_val = input.into_scalar().elem::<i64>();
            long_val
        }
        ");
    }

    #[test]
    fn test_reshape_tensor_to_scalar_bool() {
        let config = ReshapeConfig {
            shape: ReshapeInput::Static(vec![]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("mask", 1, DType::Bool)
            .output_scalar("flag", DType::Bool)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, mask: Tensor<B, 1, Bool>) -> bool {
            let flag = mask.into_scalar().elem::<bool>();
            flag
        }
        ");
    }

    // Static Shape -> Scalar
    #[test]
    fn test_reshape_shape_to_scalar_i64() {
        let config = ReshapeConfig {
            shape: ReshapeInput::Static(vec![]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_shape("shape_in")
            .output_scalar("dim", DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, shape_in: [i64; 1]) -> i64 {
            let dim = shape_in[0] as i64;
            dim
        }
        ");
    }

    #[test]
    fn test_reshape_shape_to_scalar_i32() {
        let config = ReshapeConfig {
            shape: ReshapeInput::Static(vec![]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_shape("shape_data")
            .output_scalar("size", DType::I32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, shape_data: [i64; 1]) -> i32 {
            let size = shape_data[0] as i32;
            size
        }
        ");
    }

    // Static Shape -> Shape (same rank)
    #[test]
    fn test_reshape_shape_to_shape_same_rank() {
        let config = ReshapeConfig {
            shape: ReshapeInput::Static(vec![]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_shape("input_shape")
            .output_shape("output_shape")
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input_shape: [i64; 1]) -> [i64; 1] {
            let output_shape = input_shape;
            output_shape
        }
        ");
    }

    // Static Shape -> Shape (different rank)
    #[test]
    fn test_reshape_shape_to_shape_expand() {
        let config = ReshapeConfig {
            shape: ReshapeInput::Static(vec![]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_shape("small_shape")
            .output_shape("large_shape")
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, small_shape: [i64; 1]) -> [i64; 1] {
            let large_shape = small_shape;
            large_shape
        }
        ");
    }

    #[test]
    fn test_reshape_shape_to_shape_shrink() {
        let config = ReshapeConfig {
            shape: ReshapeInput::Static(vec![]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_shape("big_shape")
            .output_shape("tiny_shape")
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, big_shape: [i64; 1]) -> [i64; 1] {
            let tiny_shape = big_shape;
            tiny_shape
        }
        ");
    }

    // Static Shape -> Tensor
    #[test]
    fn test_reshape_shape_to_tensor() {
        let config = ReshapeConfig {
            shape: ReshapeInput::Static(vec![3]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_shape("dims")
            .output_tensor("tensor_dims", 1, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, dims: [i64; 1]) -> Tensor<B, 1, Int> {
            let tensor_dims = {
                let shape_array = dims as [i64; 1usize];
                Tensor::<B, 1, Int>::from_data(TensorData::from(shape_array), &self.device)
            }
                .reshape([3]);
            tensor_dims
        }
        ");
    }

    // Runtime shape with Shape argument
    #[test]
    fn test_reshape_runtime_with_shape_arg() {
        let config = ReshapeConfig {
            shape: ReshapeInput::Runtime(RuntimeInputRef {
                name: "target_shape".to_string(),
                input_index: 1,
            }),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("data", 3, DType::F32)
            .input_shape("target_shape")
            .output_tensor("reshaped", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<B, 3>, target_shape: [i64; 1]) -> Tensor<B, 2> {
            let reshaped = data.reshape(target_shape);
            reshaped
        }
        ");
    }

    // Runtime shape with Tensor argument (rank 2)
    #[test]
    fn test_reshape_runtime_with_tensor_rank2() {
        let config = ReshapeConfig {
            shape: ReshapeInput::Runtime(RuntimeInputRef {
                name: "new_shape".to_string(),
                input_index: 1,
            }),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("x", 3, DType::F32)
            .input_tensor("new_shape", 1, DType::I64)
            .output_tensor("y", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, x: Tensor<B, 3>, new_shape: Tensor<B, 1, Int>) -> Tensor<B, 2> {
            let shape_data = new_shape.to_data();
            let shape_array = shape_data.as_slice::<i64>().unwrap();
            let y = x.reshape([shape_array[0] as usize, shape_array[1] as usize]);
            y
        }
        ");
    }

    // Runtime shape with Tensor argument (rank 3)
    #[test]
    fn test_reshape_runtime_with_tensor_rank3() {
        let config = ReshapeConfig {
            shape: ReshapeInput::Runtime(RuntimeInputRef {
                name: "shape_tensor".to_string(),
                input_index: 1,
            }),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("input", 4, DType::F32)
            .input_tensor("shape_tensor", 1, DType::I64)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 4>,
            shape_tensor: Tensor<B, 1, Int>,
        ) -> Tensor<B, 3> {
            let shape_data = shape_tensor.to_data();
            let shape_array = shape_data.as_slice::<i64>().unwrap();
            let output = input
                .reshape([
                    shape_array[0] as usize,
                    shape_array[1] as usize,
                    shape_array[2] as usize,
                ]);
            output
        }
        ");
    }

    // Runtime shape with Tensor argument (rank 4)
    #[test]
    fn test_reshape_runtime_with_tensor_rank4() {
        let config = ReshapeConfig {
            shape: ReshapeInput::Runtime(RuntimeInputRef {
                name: "dims".to_string(),
                input_index: 1,
            }),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("tensor_in", 2, DType::F32)
            .input_tensor("dims", 1, DType::I64)
            .output_tensor("tensor_out", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, tensor_in: Tensor<B, 2>, dims: Tensor<B, 1, Int>) -> Tensor<B, 4> {
            let shape_data = dims.to_data();
            let shape_array = shape_data.as_slice::<i64>().unwrap();
            let tensor_out = tensor_in
                .reshape([
                    shape_array[0] as usize,
                    shape_array[1] as usize,
                    shape_array[2] as usize,
                    shape_array[3] as usize,
                ]);
            tensor_out
        }
        ");
    }
}
