use super::prelude::*;
use onnx_ir::ir::ArgType;

impl<PS: PrecisionSettings> NodeCodegen<PS>
    for onnx_ir::node::constant_of_shape::ConstantOfShapeNode
{
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, _scope: &mut super::super::scope::ScopeAtPosition<'_>) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Extract fill value from config
        let value = if let Some(tensor_data) = &self.config.value {
            // Extract the scalar value from the tensor data
            match tensor_data.dtype {
                onnx_ir::ir::DType::F32 => {
                    let val = tensor_data.as_slice::<f32>().unwrap()[0];
                    quote! { #val }
                }
                onnx_ir::ir::DType::F64 => {
                    let val = tensor_data.as_slice::<f64>().unwrap()[0];
                    quote! { #val }
                }
                onnx_ir::ir::DType::I32 => {
                    let val = tensor_data.as_slice::<i32>().unwrap()[0];
                    quote! { #val }
                }
                onnx_ir::ir::DType::I64 => {
                    let val = tensor_data.as_slice::<i64>().unwrap()[0];
                    quote! { #val }
                }
                onnx_ir::ir::DType::Bool => {
                    let val = tensor_data.as_slice::<bool>().unwrap()[0];
                    quote! { #val }
                }
                _ => quote! { 0.0f32 }, // Default fallback
            }
        } else {
            quote! { 0.0f32 } // Default value per ONNX spec
        };

        // Generate shape expression based on Static or Runtime
        let shape_expr = match &self.config.shape {
            onnx_ir::node::constant_of_shape::ConstantOfShapeShape::Static(static_shape) => {
                // Static shape values - embed them directly in the code
                let shape_values = static_shape.iter().map(|v| {
                    let val = *v as usize;
                    quote! { #val }
                });
                quote! { [#(#shape_values),*] }
            }
            onnx_ir::node::constant_of_shape::ConstantOfShapeShape::Runtime(runtime_ref) => {
                // Runtime shape input
                let arg = &self.inputs[runtime_ref.input_index];
                let input_name = arg_to_ident(arg);
                quote! { #input_name }
            }
        };

        // Generate code based on output type
        match &self.outputs[0].ty {
            ArgType::Scalar(_) => {
                // For scalar output, the input shape should be empty (rank 0)
                // Just return the constant value directly
                quote! {
                    let #output = #value;
                }
            }
            ArgType::Tensor(tensor) => {
                let output_rank = tensor.rank.to_tokens();

                // Check if value is boolean - special handling needed
                let is_bool_value = if let Some(tensor_data) = &self.config.value {
                    matches!(tensor_data.dtype, onnx_ir::ir::DType::Bool)
                } else {
                    false
                };

                if is_bool_value {
                    // Boolean tensors need special handling
                    let bool_val = if let Some(tensor_data) = &self.config.value {
                        tensor_data.as_slice::<bool>().unwrap()[0]
                    } else {
                        false
                    };

                    if bool_val {
                        quote! {
                            let #output = Tensor::<B, #output_rank, Int>::ones(#shape_expr, &*self.device).bool();
                        }
                    } else {
                        quote! {
                            let #output = Tensor::<B, #output_rank, Int>::zeros(#shape_expr, &*self.device).bool();
                        }
                    }
                } else {
                    quote! {
                        let #output = Tensor::full(#shape_expr, #value, &*self.device);
                    }
                }
            }
            ArgType::Shape(_) => {
                // Optimization: When ConstantOfShape outputs Shape(1) with Int64,
                // we directly create a shape array instead of a tensor
                quote! {
                    let #output: [i64; 1] = [#value];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use insta::assert_snapshot;
    use onnx_ir::ir::{DType, RuntimeInputRef, TensorData};
    use onnx_ir::node::constant_of_shape::{
        ConstantOfShapeConfig, ConstantOfShapeNodeBuilder, ConstantOfShapeShape,
    };

    // ==================== Scalar Output Tests ====================

    #[test]
    fn test_constant_of_shape_scalar_f32() {
        let config = ConstantOfShapeConfig {
            shape: ConstantOfShapeShape::Static(vec![]),
            value: Some(TensorData::new(vec![3.14f32], vec![])),
        };
        let node = ConstantOfShapeNodeBuilder::new("const1")
            .input_shape("dims")
            .output_scalar("result", DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let result = 3.14f32;");
    }

    #[test]
    fn test_constant_of_shape_scalar_f64() {
        let config = ConstantOfShapeConfig {
            shape: ConstantOfShapeShape::Static(vec![]),
            value: Some(TensorData::new(vec![2.718f64], vec![])),
        };
        let node = ConstantOfShapeNodeBuilder::new("const1")
            .input_shape("shape_in")
            .output_scalar("value", DType::F64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let value = 2.718f64;");
    }

    #[test]
    fn test_constant_of_shape_scalar_i32() {
        let config = ConstantOfShapeConfig {
            shape: ConstantOfShapeShape::Static(vec![]),
            value: Some(TensorData::new(vec![42i32], vec![])),
        };
        let node = ConstantOfShapeNodeBuilder::new("const1")
            .input_shape("s")
            .output_scalar("num", DType::I32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let num = 42i32;");
    }

    #[test]
    fn test_constant_of_shape_scalar_i64() {
        let config = ConstantOfShapeConfig {
            shape: ConstantOfShapeShape::Static(vec![]),
            value: Some(TensorData::new(vec![999i64], vec![])),
        };
        let node = ConstantOfShapeNodeBuilder::new("const1")
            .input_shape("shape_data")
            .output_scalar("output", DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = 999i64;");
    }

    #[test]
    fn test_constant_of_shape_scalar_bool() {
        let config = ConstantOfShapeConfig {
            shape: ConstantOfShapeShape::Static(vec![]),
            value: Some(TensorData::new(vec![true], vec![])),
        };
        let node = ConstantOfShapeNodeBuilder::new("const1")
            .input_shape("shape_vec")
            .output_scalar("flag", DType::Bool)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let flag = true;");
    }

    #[test]
    fn test_constant_of_shape_scalar_default() {
        let config = ConstantOfShapeConfig {
            shape: ConstantOfShapeShape::Static(vec![]),
            value: None, // Should default to 0.0f32
        };
        let node = ConstantOfShapeNodeBuilder::new("const1")
            .input_shape("s")
            .output_scalar("zero", DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let zero = 0.0f32;");
    }

    // ==================== Tensor with Static Shape Tests ====================

    #[test]
    fn test_constant_of_shape_tensor_f32_static() {
        let config = ConstantOfShapeConfig {
            shape: ConstantOfShapeShape::Static(vec![2, 3, 4]),
            value: Some(TensorData::new(vec![1.5f32], vec![])),
        };
        let node = ConstantOfShapeNodeBuilder::new("const1")
            .input_shape("target_shape")
            .output_tensor("filled", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let filled = Tensor::full([2usize, 3usize, 4usize], 1.5f32, &*self.device);");
    }

    #[test]
    fn test_constant_of_shape_tensor_f64_static() {
        let config = ConstantOfShapeConfig {
            shape: ConstantOfShapeShape::Static(vec![10, 20]),
            value: Some(TensorData::new(vec![0.5f64], vec![])),
        };
        let node = ConstantOfShapeNodeBuilder::new("const1")
            .input_shape("dims")
            .output_tensor("matrix", 2, DType::F64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let matrix = Tensor::full([10usize, 20usize], 0.5f64, &*self.device);");
    }

    #[test]
    fn test_constant_of_shape_tensor_i32_static() {
        let config = ConstantOfShapeConfig {
            shape: ConstantOfShapeShape::Static(vec![5, 5]),
            value: Some(TensorData::new(vec![7i32], vec![])),
        };
        let node = ConstantOfShapeNodeBuilder::new("const1")
            .input_shape("size")
            .output_tensor("grid", 2, DType::I32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let grid = Tensor::full([5usize, 5usize], 7i32, &*self.device);");
    }

    #[test]
    fn test_constant_of_shape_tensor_i64_static() {
        let config = ConstantOfShapeConfig {
            shape: ConstantOfShapeShape::Static(vec![8]),
            value: Some(TensorData::new(vec![100i64], vec![])),
        };
        let node = ConstantOfShapeNodeBuilder::new("const1")
            .input_shape("length")
            .output_tensor("vector", 1, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let vector = Tensor::full([8usize], 100i64, &*self.device);");
    }

    #[test]
    fn test_constant_of_shape_tensor_bool_true_static() {
        let config = ConstantOfShapeConfig {
            shape: ConstantOfShapeShape::Static(vec![3, 4]),
            value: Some(TensorData::new(vec![true], vec![])),
        };
        let node = ConstantOfShapeNodeBuilder::new("const1")
            .input_shape("shape_dims")
            .output_tensor("mask", 2, DType::Bool)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let mask = Tensor::<B, 2, Int>::ones([3usize, 4usize], &*self.device).bool();");
    }

    #[test]
    fn test_constant_of_shape_tensor_bool_false_static() {
        let config = ConstantOfShapeConfig {
            shape: ConstantOfShapeShape::Static(vec![6, 7, 8]),
            value: Some(TensorData::new(vec![false], vec![])),
        };
        let node = ConstantOfShapeNodeBuilder::new("const1")
            .input_shape("dimensions")
            .output_tensor("flags", 3, DType::Bool)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        let flags = Tensor::<B, 3, Int>::zeros([6usize, 7usize, 8usize], &*self.device)
                .bool();
        ");
    }

    #[test]
    fn test_constant_of_shape_tensor_default_static() {
        let config = ConstantOfShapeConfig {
            shape: ConstantOfShapeShape::Static(vec![2, 2]),
            value: None, // Defaults to 0.0f32
        };
        let node = ConstantOfShapeNodeBuilder::new("const1")
            .input_shape("size")
            .output_tensor("zeros", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let zeros = Tensor::full([2usize, 2usize], 0.0f32, &*self.device);");
    }

    // ==================== Tensor with Runtime Shape Tests ====================

    #[test]
    fn test_constant_of_shape_tensor_runtime_f32() {
        let config = ConstantOfShapeConfig {
            shape: ConstantOfShapeShape::Runtime(RuntimeInputRef {
                name: "dynamic_shape".to_string(),
                input_index: 0,
            }),
            value: Some(TensorData::new(vec![2.5f32], vec![])),
        };
        let node = ConstantOfShapeNodeBuilder::new("const1")
            .input_shape("dynamic_shape")
            .output_tensor("tensor", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let tensor = Tensor::full(dynamic_shape, 2.5f32, &*self.device);");
    }

    #[test]
    fn test_constant_of_shape_tensor_runtime_i64() {
        let config = ConstantOfShapeConfig {
            shape: ConstantOfShapeShape::Runtime(RuntimeInputRef {
                name: "shape_param".to_string(),
                input_index: 0,
            }),
            value: Some(TensorData::new(vec![255i64], vec![])),
        };
        let node = ConstantOfShapeNodeBuilder::new("const1")
            .input_shape("shape_param")
            .output_tensor("data", 2, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let data = Tensor::full(shape_param, 255i64, &*self.device);");
    }

    #[test]
    fn test_constant_of_shape_tensor_runtime_bool_true() {
        let config = ConstantOfShapeConfig {
            shape: ConstantOfShapeShape::Runtime(RuntimeInputRef {
                name: "sz".to_string(),
                input_index: 0,
            }),
            value: Some(TensorData::new(vec![true], vec![])),
        };
        let node = ConstantOfShapeNodeBuilder::new("const1")
            .input_shape("sz")
            .output_tensor("bitmask", 4, DType::Bool)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let bitmask = Tensor::<B, 4, Int>::ones(sz, &*self.device).bool();");
    }

    #[test]
    fn test_constant_of_shape_tensor_runtime_bool_false() {
        let config = ConstantOfShapeConfig {
            shape: ConstantOfShapeShape::Runtime(RuntimeInputRef {
                name: "target_dims".to_string(),
                input_index: 0,
            }),
            value: Some(TensorData::new(vec![false], vec![])),
        };
        let node = ConstantOfShapeNodeBuilder::new("const1")
            .input_shape("target_dims")
            .output_tensor("empty_mask", 2, DType::Bool)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let empty_mask = Tensor::<B, 2, Int>::zeros(target_dims, &*self.device).bool();");
    }

    #[test]
    fn test_constant_of_shape_tensor_runtime_default() {
        let config = ConstantOfShapeConfig {
            shape: ConstantOfShapeShape::Runtime(RuntimeInputRef {
                name: "runtime_shape".to_string(),
                input_index: 0,
            }),
            value: None, // Defaults to 0.0f32
        };
        let node = ConstantOfShapeNodeBuilder::new("const1")
            .input_shape("runtime_shape")
            .output_tensor("zeros", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let zeros = Tensor::full(runtime_shape, 0.0f32, &*self.device);");
    }

    // ==================== Shape Output Tests ====================

    #[test]
    fn test_constant_of_shape_shape_output_i64() {
        let config = ConstantOfShapeConfig {
            shape: ConstantOfShapeShape::Static(vec![]),
            value: Some(TensorData::new(vec![10i64], vec![])),
        };
        let node = ConstantOfShapeNodeBuilder::new("const1")
            .input_shape("in_shape")
            .output_shape("out_shape")
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let out_shape: [i64; 1] = [10i64];");
    }

    #[test]
    fn test_constant_of_shape_shape_output_with_default() {
        let config = ConstantOfShapeConfig {
            shape: ConstantOfShapeShape::Static(vec![]),
            value: None, // Defaults to 0.0f32, but for shape output context this might be handled differently
        };
        let node = ConstantOfShapeNodeBuilder::new("const1")
            .input_shape("dims")
            .output_shape("result")
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let result: [i64; 1] = [0.0f32];");
    }
}
