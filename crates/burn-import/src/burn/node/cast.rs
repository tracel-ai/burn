use super::prelude::*;
use crate::burn::TensorKind;
use onnx_ir::ir::DType;

impl NodeCodegen for onnx_ir::cast::CastNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();

        match (&input_arg.ty, &output_arg.ty) {
            // -----------------------
            // Scalar -> Scalar
            // -----------------------
            (ArgType::Scalar(input_dtype), ArgType::Scalar(_output_dtype)) => {
                let input = arg_to_ident(input_arg);
                let output = arg_to_ident(output_arg);

                // Check if the cast is a no-op within the same dtype "family"
                let is_noop = input_dtype == &self.config.to
                    || (input_dtype.is_float()
                        && self.config.to.is_float()
                        && input_dtype != &DType::F64
                        && self.config.to != DType::F64)
                    || ((input_dtype.is_int() || input_dtype.is_uint())
                        && (self.config.to.is_int() || self.config.to.is_uint()));

                if is_noop {
                    // No-op cast within same scalar "family".
                    quote! {
                        let #output = #input;
                    }
                } else {
                    // Concrete Rust target type for the cast expression.
                    let ty = match self.config.to {
                        DType::F32 | DType::F16 => quote! { f32 },
                        DType::F64 => quote! { f64 },
                        DType::I32 => quote! { i32 },
                        DType::I64 => quote! { i64 },
                        DType::U16 => quote! { u16 },
                        DType::I8 => quote! { i8 },
                        DType::U8 => quote! { u8 },
                        DType::Bool => quote! { bool },
                        _ => panic!("Unsupported DType for Cast: {:?}", self.config.to),
                    };
                    quote! {
                        let #output = #input as #ty;
                    }
                }
            }

            // -----------------------
            // Tensor -> Tensor
            // -----------------------
            (ArgType::Tensor(input_tensor), ArgType::Tensor(_output_tensor)) => {
                let input = scope.arg(input_arg);
                let output = arg_to_ident(output_arg);

                // Map ONNX element types to Burn TensorKind categories.
                // Burn only distinguishes Float / Int / Bool at the Tensor level.
                let target_kind = match &self.config.to {
                    dtype if dtype.is_float() => TensorKind::Float,
                    dtype if dtype.is_int() || dtype.is_uint() => TensorKind::Int,
                    dtype if dtype.is_bool() => TensorKind::Bool,
                    _ => panic!("Unsupported DType for Cast: {:?}", self.config.to),
                };

                let input_kind: TensorKind = input_tensor.dtype.into();

                if input_kind == target_kind {
                    // No-op cast if already in the correct TensorKind category.
                    quote! {
                        let #output = #input;
                    }
                } else {
                    // Burn exposes category-level casts: .float(), .int(), .bool()
                    let cast_fn = match target_kind {
                        TensorKind::Bool => quote! { bool() },
                        TensorKind::Int => quote! { int() },
                        TensorKind::Float => quote! { float() },
                    };
                    quote! {
                        let #output = #input.#cast_fn;
                    }
                }
            }

            // -----------------------
            // Shape -> Shape
            // -----------------------
            (ArgType::Shape(_), ArgType::Shape(_)) => {
                let input = arg_to_ident(input_arg);
                let output = arg_to_ident(output_arg);
                // Shapes stay as [i64; N] regardless of ONNX target. No cast.
                quote! {
                    let #output = #input;
                }
            }

            // -----------------------
            // Shape -> Tensor
            // (Mostly for float/bool visualization or downstream ops.)
            // -----------------------
            (ArgType::Shape(input_rank), ArgType::Tensor(_)) => {
                let input = arg_to_ident(input_arg);
                let output = arg_to_ident(output_arg);
                let rank = *input_rank;

                match &self.config.to {
                    dtype if dtype.is_float() => {
                        // Emit f32 tensor; Float64 target collapses to f32 at runtime side.
                        quote! {
                            let #output = {
                                let shape_array = #input as [i64; #rank];
                                let float_array: [f32; #rank] = shape_array.map(|x| x as f32);
                                Tensor::<B, 1>::from_data(
                                    TensorData::from(float_array),
                                    &self.device
                                )
                            };
                        }
                    }
                    dtype if dtype.is_bool() => {
                        quote! {
                            let #output = {
                                let shape_array = #input as [i64; #rank];
                                let bool_array: [bool; #rank] = shape_array.map(|x| x != 0);
                                Tensor::<B, 1, Bool>::from_data(
                                    TensorData::from(bool_array),
                                    &self.device
                                )
                            };
                        }
                    }
                    // For all integer widths (Int32, Int64, Int8, Uint8), we keep Shape as Shape
                    // in onnx-ir and shouldn't go through Shape->Tensor here.
                    dtype if dtype.is_int() || dtype.is_uint() => {
                        panic!(
                            "Cast: Shape to Int tensor should be handled as Shape->Shape in onnx-ir"
                        )
                    }
                    _ => panic!("Unsupported DType for Cast: {:?}", self.config.to),
                }
            }

            _ => panic!(
                "Cast: unsupported type combination: input={:?}, output={:?}",
                input_arg.ty, output_arg.ty
            ),
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();

        // Only register imports when actually needed
        match (&input_arg.ty, &output_arg.ty) {
            // Shape -> Tensor casts need TensorData
            (ArgType::Shape(_), ArgType::Tensor(_)) => {
                imports.register("burn::tensor::TensorData");
                // Bool is only needed for Shape -> Bool tensor
                if self.config.to.is_bool() {
                    imports.register("burn::tensor::Bool");
                }
            }
            _ => {
                // Other cast types don't need these imports
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::cast::{CastConfig, CastNode, CastNodeBuilder};

    fn create_cast_node_tensor(name: &str, input_dtype: DType, output_dtype: DType) -> CastNode {
        let config = CastConfig::new(output_dtype);
        CastNodeBuilder::new(name)
            .input_tensor("input", 2, input_dtype)
            .output_tensor("output", 2, output_dtype)
            .config(config)
            .build()
    }

    fn create_cast_node_scalar(name: &str, input_dtype: DType, output_dtype: DType) -> CastNode {
        let config = CastConfig::new(output_dtype);
        CastNodeBuilder::new(name)
            .input_scalar("input", input_dtype)
            .output_scalar("output", output_dtype)
            .config(config)
            .build()
    }

    #[test]
    fn test_cast_int_to_float() {
        let node = create_cast_node_tensor("cast1", DType::I32, DType::F32);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 2> {
            let output = input.float();
            output
        }
        ");
    }

    #[test]
    fn test_cast_float_to_int() {
        let node = create_cast_node_tensor("cast1", DType::F32, DType::I32);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2, Int> {
            let output = input.int();
            output
        }
        ");
    }

    #[test]
    fn test_cast_float_to_bool() {
        let node = create_cast_node_tensor("cast1", DType::F32, DType::Bool);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2, Bool> {
            let output = input.bool();
            output
        }
        ");
    }

    #[test]
    fn test_cast_noop_float32_to_float32() {
        let node = create_cast_node_tensor("cast1", DType::F32, DType::F32);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input;
            output
        }
        ");
    }

    #[test]
    fn test_cast_scalar_int_to_float() {
        let node = create_cast_node_scalar("cast1", DType::I32, DType::F32);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: i32) -> f32 {
            let output = input as f32;
            output
        }
        ");
    }

    #[test]
    fn test_cast_scalar_float_to_int() {
        let node = create_cast_node_scalar("cast1", DType::F32, DType::I64);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: f32) -> i64 {
            let output = input as i64;
            output
        }
        ");
    }

    #[test]
    fn test_cast_scalar_noop() {
        let node = create_cast_node_scalar("cast1", DType::F32, DType::F32);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: f32) -> f32 {
            let output = input;
            output
        }
        ");
    }
}
