use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::is_inf::IsInfNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();

        let input = scope.arg(input_arg);
        let output = arg_to_ident(output_arg);

        let function = match &output_arg.ty {
            ArgType::Scalar(_) => {
                match (self.config.detect_negative, self.config.detect_positive) {
                    (true, true) => quote! { #input.is_infinite() },
                    (false, true) => quote! { #input.is_infinite() && #input.is_sign_positive() },
                    (true, false) => quote! { #input.is_infinite() && #input.is_sign_negative() },
                    (false, false) => quote! { false },
                }
            }
            ArgType::Tensor(_) => {
                match (self.config.detect_negative, self.config.detect_positive) {
                    (true, true) => quote! { #input.is_inf() },
                    (false, true) => {
                        quote! { #input.clone().is_inf().bool_and(#input.greater_elem(0.0)) }
                    }
                    (true, false) => {
                        quote! { #input.clone().is_inf().bool_and(#input.lower_elem(0.0)) }
                    }
                    (false, false) => quote! { #input.zeros_like().bool() },
                }
            }
            _ => panic!("IsInf only supports scalar or tensor outputs"),
        };

        quote! {
            let #output = #function;
        }
    }

    fn register_imports(&self, _imports: &mut BurnImports) {
        // No special imports needed - is_inf() is a tensor method
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::is_inf::{IsInfConfig, IsInfNodeBuilder};

    #[test]
    fn test_is_inf_both() {
        let config = IsInfConfig::new(true, true);
        let node = IsInfNodeBuilder::new("isinf1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::Bool)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2, Bool> {
            let output = input.is_inf();
            output
        }
        ");
    }

    #[test]
    fn test_is_inf_positive_only() {
        let config = IsInfConfig::new(false, true);
        let node = IsInfNodeBuilder::new("isinf2")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::Bool)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2, Bool> {
            let output = input.clone().is_inf().bool_and(input.greater_elem(0.0));
            output
        }
        ");
    }

    #[test]
    fn test_is_inf_negative_only() {
        let config = IsInfConfig::new(true, false);
        let node = IsInfNodeBuilder::new("isinf3")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::Bool)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2, Bool> {
            let output = input.clone().is_inf().bool_and(input.lower_elem(0.0));
            output
        }
        ");
    }

    #[test]
    fn test_is_inf_neither() {
        let config = IsInfConfig::new(false, false);
        let node = IsInfNodeBuilder::new("isinf4")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::Bool)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2, Bool> {
            let output = input.zeros_like().bool();
            output
        }
        ");
    }

    #[test]
    fn test_is_inf_scalar_both() {
        let config = IsInfConfig::new(true, true);
        let node = IsInfNodeBuilder::new("isinf5")
            .input_scalar("input", DType::F32)
            .output_scalar("output", DType::Bool)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: f32) -> bool {
            let output = input.is_infinite();
            output
        }
        ");
    }

    #[test]
    fn test_is_inf_scalar_positive_only() {
        let config = IsInfConfig::new(false, true);
        let node = IsInfNodeBuilder::new("isinf6")
            .input_scalar("input", DType::F32)
            .output_scalar("output", DType::Bool)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: f32) -> bool {
            let output = input.is_infinite() && input.is_sign_positive();
            output
        }
        ");
    }

    #[test]
    fn test_is_inf_scalar_negative_only() {
        let config = IsInfConfig::new(true, false);
        let node = IsInfNodeBuilder::new("isinf7")
            .input_scalar("input", DType::F32)
            .output_scalar("output", DType::Bool)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: f32) -> bool {
            let output = input.is_infinite() && input.is_sign_negative();
            output
        }
        ");
    }

    #[test]
    fn test_is_inf_scalar_neither() {
        let config = IsInfConfig::new(false, false);
        let node = IsInfNodeBuilder::new("isinf8")
            .input_scalar("input", DType::F32)
            .output_scalar("output", DType::Bool)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: f32) -> bool {
            let output = false;
            output
        }
        ");
    }
}
