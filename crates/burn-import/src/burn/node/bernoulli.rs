use super::prelude::*;
use onnx_ir::ir::ArgType;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::bernoulli::BernoulliNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Use Default distribution for Bernoulli
        let dist = quote! { Distribution::Default };

        // Generate random values and compare with input to get binary output
        let input_random = quote! { #input.random_like(#dist).lower(#input) };

        // Convert to the output type based on the output tensor kind
        let output_ty = &self.outputs.first().unwrap().ty;
        let output_random = match output_ty {
            ArgType::Tensor(t) => match &t.dtype {
                dtype if dtype.is_bool() => input_random,
                dtype if dtype.is_int() || dtype.is_uint() => quote! { #input_random.int() },
                dtype if dtype.is_float() => quote! { #input_random.float() },
                _ => input_random, // Fallback
            },
            _ => input_random,
        };

        quote! {
            let #output = #output_random;
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::tensor::Distribution");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::bernoulli::BernoulliNodeBuilder;

    #[test]
    fn test_bernoulli_bool() {
        let node = BernoulliNodeBuilder::new("bernoulli1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::Bool)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2, Bool> {
            let output = input.random_like(Distribution::Default).lower(input);
            output
        }
        ");
    }

    #[test]
    fn test_bernoulli_int() {
        let node = BernoulliNodeBuilder::new("bernoulli2")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2, Int> {
            let output = input.random_like(Distribution::Default).lower(input).int();
            output
        }
        ");
    }

    #[test]
    fn test_bernoulli_float() {
        let node = BernoulliNodeBuilder::new("bernoulli3")
            .input_tensor("input", 2, DType::F64)
            .output_tensor("output", 2, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input.random_like(Distribution::Default).lower(input).float();
            output
        }
        ");
    }

    #[test]
    fn test_bernoulli_int64() {
        let node = BernoulliNodeBuilder::new("bernoulli4")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::I64)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2, Int> {
            let output = input.random_like(Distribution::Default).lower(input).int();
            output
        }
        ");
    }
}
