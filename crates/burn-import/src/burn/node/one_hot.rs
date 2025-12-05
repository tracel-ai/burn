use super::prelude::*;
use crate::burn::TensorKind;

impl NodeCodegen for onnx_ir::one_hot::OneHotNode {
    fn inputs(&self) -> &[Argument] {
        // Only the first input (indices) is a dynamic tensor
        // depth and values are either static or runtime inputs
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Extract num_classes from config.depth
        let num_classes = match &self.config.depth {
            onnx_ir::one_hot::OneHotDepthInput::Static(d) => quote! { #d },
            onnx_ir::one_hot::OneHotDepthInput::Runtime(_) => {
                panic!("OneHot with runtime depth is not supported in burn-import")
            }
        };

        // Extract values from config.values
        let (on_value, off_value) = match &self.config.values {
            onnx_ir::one_hot::OneHotValuesInput::Static(v) => {
                let off = v[0];
                let on = v[1];
                (quote! { #on }, quote! { #off })
            }
            onnx_ir::one_hot::OneHotValuesInput::Runtime(_) => {
                panic!("OneHot with runtime values is not supported in burn-import")
            }
        };

        let axis = self.config.axis;

        // Determine input and output tensor kinds
        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();

        let input_kind = match &input_arg.ty {
            ArgType::Tensor(t) => TensorKind::from(t.dtype),
            _ => panic!("Expected tensor input"),
        };

        let output_kind = match &output_arg.ty {
            ArgType::Tensor(t) => TensorKind::from(t.dtype),
            _ => panic!("Expected tensor output"),
        };

        match (input_kind, output_kind) {
            (TensorKind::Int, TensorKind::Int) | (TensorKind::Float, TensorKind::Float) => {
                quote! {
                    let #output = #input.one_hot_fill(#num_classes, #on_value, #off_value, #axis);
                }
            }
            (TensorKind::Int, TensorKind::Float) => {
                quote! {
                    let #output = #input.one_hot_fill(#num_classes, #on_value, #off_value, #axis).float();
                }
            }
            (TensorKind::Float, TensorKind::Int) => {
                quote! {
                    let #output = #input.one_hot_fill(#num_classes, #on_value, #off_value, #axis).int();
                }
            }
            (TensorKind::Int, TensorKind::Bool) | (TensorKind::Float, TensorKind::Bool) => {
                quote! {
                    let #output = #input.one_hot_fill(#num_classes, #on_value, #off_value, #axis).bool();
                }
            }
            (TensorKind::Bool, _) => panic!("Input should be numeric"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::one_hot::{OneHotConfig, OneHotDepthInput, OneHotNodeBuilder, OneHotValuesInput};

    #[test]
    fn test_one_hot() {
        let config = OneHotConfig::new(
            OneHotDepthInput::Static(10),
            OneHotValuesInput::Static([0.0, 1.0]),
            -1,
        );
        let node = OneHotNodeBuilder::new("onehot1")
            .input_tensor("indices", 1, DType::I32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, indices: Tensor<B, 1, Int>) -> Tensor<B, 2> {
            let output = indices.one_hot_fill(10usize, 1f32, 0f32, -1i64).float();
            output
        }
        ");
    }

    #[test]
    fn test_one_hot_int_to_int() {
        let config = OneHotConfig::new(
            OneHotDepthInput::Static(5),
            OneHotValuesInput::Static([0.0, 1.0]),
            -1,
        );
        let node = OneHotNodeBuilder::new("onehot2")
            .input_tensor("indices", 1, DType::I32)
            .output_tensor("output", 2, DType::I32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, indices: Tensor<B, 1, Int>) -> Tensor<B, 2, Int> {
            let output = indices.one_hot_fill(5usize, 1f32, 0f32, -1i64);
            output
        }
        ");
    }

    #[test]
    fn test_one_hot_float_to_float() {
        let config = OneHotConfig::new(
            OneHotDepthInput::Static(5),
            OneHotValuesInput::Static([0.0, 1.0]),
            0,
        );
        let node = OneHotNodeBuilder::new("onehot3")
            .input_tensor("indices", 1, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, indices: Tensor<B, 1>) -> Tensor<B, 2> {
            let output = indices.one_hot_fill(5usize, 1f32, 0f32, 0i64);
            output
        }
        ");
    }

    #[test]
    fn test_one_hot_float_to_int() {
        let config = OneHotConfig::new(
            OneHotDepthInput::Static(5),
            OneHotValuesInput::Static([0.0, 1.0]),
            0,
        );
        let node = OneHotNodeBuilder::new("onehot4")
            .input_tensor("indices", 1, DType::F32)
            .output_tensor("output", 2, DType::I32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, indices: Tensor<B, 1>) -> Tensor<B, 2, Int> {
            let output = indices.one_hot_fill(5usize, 1f32, 0f32, 0i64).int();
            output
        }
        ");
    }

    #[test]
    fn test_one_hot_int_to_bool() {
        let config = OneHotConfig::new(
            OneHotDepthInput::Static(5),
            OneHotValuesInput::Static([0.0, 1.0]),
            -1,
        );
        let node = OneHotNodeBuilder::new("onehot5")
            .input_tensor("indices", 1, DType::I32)
            .output_tensor("output", 2, DType::Bool)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, indices: Tensor<B, 1, Int>) -> Tensor<B, 2, Bool> {
            let output = indices.one_hot_fill(5usize, 1f32, 0f32, -1i64).bool();
            output
        }
        ");
    }

    #[test]
    fn test_one_hot_float_to_bool() {
        let config = OneHotConfig::new(
            OneHotDepthInput::Static(5),
            OneHotValuesInput::Static([0.0, 1.0]),
            0,
        );
        let node = OneHotNodeBuilder::new("onehot6")
            .input_tensor("indices", 1, DType::F32)
            .output_tensor("output", 2, DType::Bool)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, indices: Tensor<B, 1>) -> Tensor<B, 2, Bool> {
            let output = indices.one_hot_fill(5usize, 1f32, 0f32, 0i64).bool();
            output
        }
        ");
    }
}
