use super::prelude::*;

impl NodeCodegen for onnx_ir::clip::ClipNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Extract static values from ClipInput enum
        let min = match &self.config.min {
            Some(onnx_ir::node::clip::ClipInput::Static(v)) => Some(*v),
            Some(onnx_ir::node::clip::ClipInput::Runtime(_)) => {
                panic!("Clip: runtime min values are not supported in burn-onnx")
            }
            None => None,
        };
        let max = match &self.config.max {
            Some(onnx_ir::node::clip::ClipInput::Static(v)) => Some(*v),
            Some(onnx_ir::node::clip::ClipInput::Runtime(_)) => {
                panic!("Clip: runtime max values are not supported in burn-onnx")
            }
            None => None,
        };

        if let Some(min) = min {
            if let Some(max) = max {
                quote! {
                    let #output = #input.clamp(#min, #max);
                }
            } else {
                quote! {
                    let #output = #input.clamp_min(#min);
                }
            }
        } else if let Some(max) = max {
            quote! {
                let #output = #input.clamp_max(#max);
            }
        } else {
            panic!("Clip node must have at least one min or max value");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::clip::{ClipConfig, ClipNode, ClipNodeBuilder};
    use onnx_ir::node::clip::ClipInput;

    fn create_clip_node(name: &str, min: Option<f64>, max: Option<f64>) -> ClipNode {
        let config = ClipConfig {
            min: min.map(ClipInput::Static),
            max: max.map(ClipInput::Static),
        };

        ClipNodeBuilder::new(name)
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_clip_both_bounds() {
        let node = create_clip_node("clip1", Some(-1.0), Some(1.0));
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input.clamp(-1f64, 1f64);
            output
        }
        ");
    }

    #[test]
    fn test_clip_min_only() {
        let node = create_clip_node("clip1", Some(0.0), None);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input.clamp_min(0f64);
            output
        }
        ");
    }

    #[test]
    fn test_clip_max_only() {
        let node = create_clip_node("clip1", None, Some(10.0));
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input.clamp_max(10f64);
            output
        }
        ");
    }
}
