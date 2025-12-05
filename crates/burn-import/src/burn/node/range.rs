use super::prelude::*;
use onnx_ir::ir::ArgType;
use proc_macro2::Literal;

impl NodeCodegen for onnx_ir::node::range::RangeNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, _scope: &mut super::super::scope::ScopeAtPosition<'_>) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Generate values for start, limit, and delta based on Static or Runtime
        let start = match &self.config.start {
            onnx_ir::node::range::RangeInput::Static(value) => {
                let literal = Literal::i64_suffixed(*value);
                quote! { #literal }
            }
            onnx_ir::node::range::RangeInput::Runtime(runtime_ref) => {
                let arg = &self.inputs[runtime_ref.input_index];
                match &arg.ty {
                    ArgType::Scalar(_) => {
                        let name = arg_to_ident(arg);
                        quote! { #name }
                    }
                    _ => panic!("Range parameter must be a scalar"),
                }
            }
        };

        let limit = match &self.config.limit {
            onnx_ir::node::range::RangeInput::Static(value) => {
                let literal = Literal::i64_suffixed(*value);
                quote! { #literal }
            }
            onnx_ir::node::range::RangeInput::Runtime(runtime_ref) => {
                let arg = &self.inputs[runtime_ref.input_index];
                match &arg.ty {
                    ArgType::Scalar(_) => {
                        let name = arg_to_ident(arg);
                        quote! { #name }
                    }
                    _ => panic!("Range parameter must be a scalar"),
                }
            }
        };

        let delta = match &self.config.delta {
            onnx_ir::node::range::RangeInput::Static(value) => {
                let literal = Literal::i64_suffixed(*value);
                quote! { #literal }
            }
            onnx_ir::node::range::RangeInput::Runtime(runtime_ref) => {
                let arg = &self.inputs[runtime_ref.input_index];
                match &arg.ty {
                    ArgType::Scalar(_) => {
                        let name = arg_to_ident(arg);
                        quote! { #name }
                    }
                    _ => panic!("Range parameter must be a scalar"),
                }
            }
        };

        quote! {
            let #output = Tensor::arange_step(#start..#limit, #delta as usize, &*self.device);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::range::{RangeConfig, RangeInput, RangeNodeBuilder};

    #[test]
    fn test_range_static() {
        let config = RangeConfig::new(
            RangeInput::Static(0),
            RangeInput::Static(10),
            RangeInput::Static(2),
        );
        let node = RangeNodeBuilder::new("range1")
            .output_tensor("output", 1, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> Tensor<B, 1, Int> {
            let output = Tensor::arange_step(0i64..10i64, 2i64 as usize, &*self.device);
            output
        }
        ");
    }
}
