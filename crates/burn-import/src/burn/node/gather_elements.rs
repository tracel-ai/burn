use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::gather_elements::GatherElementsNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> proc_macro2::TokenStream {
        let dim = self.config.axis.to_tokens();
        let input = scope.arg(self.inputs.first().unwrap());
        let index = scope.arg(&self.inputs[1]);
        let output = arg_to_ident(self.outputs.first().unwrap());

        quote! {
            let #output = #input.gather(#dim, #index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::gather_elements::{
        GatherElementsConfig, GatherElementsInput, GatherElementsNodeBuilder,
    };

    #[test]
    fn test_gather_elements() {
        let config = GatherElementsConfig {
            indices: GatherElementsInput::Static(vec![]),
            axis: 1,
        };
        let node = GatherElementsNodeBuilder::new("gather1")
            .input_tensor("input", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = input.gather(1, indices);");
    }
}
