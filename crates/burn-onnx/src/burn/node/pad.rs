use super::prelude::*;
use std::str::FromStr;

impl NodeCodegen for onnx_ir::pad::PadNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Extract static pads from the enum wrapper
        let pads_vec = match &self.config.pads {
            onnx_ir::pad::PadInput::Static(pads) => pads,
            onnx_ir::pad::PadInput::Runtime(_) => {
                panic!("Runtime pads are not supported in burn-onnx")
            }
        };
        let pads = pads_vec.iter().map(|p| p.to_tokens());

        // Generate PadMode based on the mode in config (using fully qualified path)
        let pad_mode = match &self.config.mode {
            onnx_ir::pad::PadMode::Constant => {
                // Extract static constant value from the enum wrapper
                let constant_value_f32 = match &self.config.constant_value {
                    onnx_ir::pad::ConstantValueInput::Static(value) => value,
                    onnx_ir::pad::ConstantValueInput::Runtime(_) => {
                        panic!("Runtime constant value is not supported in burn-onnx")
                    }
                };
                let constant_value_string = format!("{}_f32", constant_value_f32);
                let constant_value = TokenStream::from_str(&constant_value_string).unwrap();
                quote! { burn::tensor::ops::PadMode::Constant(#constant_value) }
            }
            onnx_ir::pad::PadMode::Reflect => {
                quote! { burn::tensor::ops::PadMode::Reflect }
            }
            onnx_ir::pad::PadMode::Edge => {
                quote! { burn::tensor::ops::PadMode::Edge }
            }
        };

        quote! {
            let #output = #input.pad((#(#pads),*), #pad_mode);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::pad::{ConstantValueInput, PadConfig, PadInput, PadMode, PadNode, PadNodeBuilder};

    fn create_pad_node(
        name: &str,
        pads: Vec<usize>,
        constant_value: f32,
        mode: PadMode,
    ) -> PadNode {
        let config = PadConfig {
            pads: PadInput::Static(pads),
            constant_value: ConstantValueInput::Static(constant_value),
            mode,
        };

        PadNodeBuilder::new(name)
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_pad_constant_simple() {
        let node = create_pad_node("pad1", vec![1, 1, 1, 1], 0.0, PadMode::Constant);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input.pad((1, 1, 1, 1), burn::tensor::ops::PadMode::Constant(0_f32));
            output
        }
        ");
    }

    #[test]
    fn test_pad_constant_asymmetric() {
        let node = create_pad_node("pad1", vec![0, 2, 1, 0], 5.5, PadMode::Constant);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input.pad((0, 2, 1, 0), burn::tensor::ops::PadMode::Constant(5.5_f32));
            output
        }
        ");
    }

    #[test]
    fn test_pad_reflect() {
        let node = create_pad_node("pad1", vec![1, 1, 1, 1], 0.0, PadMode::Reflect);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input.pad((1, 1, 1, 1), burn::tensor::ops::PadMode::Reflect);
            output
        }
        ");
    }

    #[test]
    fn test_pad_edge() {
        let node = create_pad_node("pad1", vec![1, 1, 1, 1], 0.0, PadMode::Edge);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input.pad((1, 1, 1, 1), burn::tensor::ops::PadMode::Edge);
            output
        }
        ");
    }
}
