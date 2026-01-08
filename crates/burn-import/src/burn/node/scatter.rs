use super::prelude::*;

impl NodeCodegen for onnx_ir::scatter::ScatterNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> proc_macro2::TokenStream {
        let data_arg = self.inputs.first().unwrap();
        let indices_arg = &self.inputs[1];
        let updates_arg = &self.inputs[2];
        let output_arg = self.outputs.first().unwrap();

        let data = scope.arg(data_arg);
        let indices = scope.arg(indices_arg);
        let updates = scope.arg(updates_arg);
        let output = arg_to_ident(output_arg);

        // Get input rank for axis normalization
        let input_rank = match &data_arg.ty {
            ArgType::Tensor(tensor) => tensor.rank as i64,
            _ => panic!("Scatter data input must be a tensor"),
        };

        // Normalize negative axis
        let axis = if self.config.axis < 0 {
            (input_rank + self.config.axis) as usize
        } else {
            self.config.axis as usize
        };

        let dim = axis.to_tokens();

        // Note: Using IndexingUpdateOp::Add - ONNX Scatter semantics (replace) will
        // produce correct results when scattering into a tensor with zeros at target positions.
        // Future: When IndexingUpdateOp::Assign is available, this can be updated.
        quote! {
            let #output = #data.scatter(#dim, #indices, #updates, burn::tensor::IndexingUpdateOp::Add);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::scatter::{ScatterConfig, ScatterNodeBuilder};

    #[test]
    fn test_scatter_axis0() {
        let config = ScatterConfig { axis: 0 };
        let node = ScatterNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code);
    }

    #[test]
    fn test_scatter_axis1() {
        let config = ScatterConfig { axis: 1 };
        let node = ScatterNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code);
    }

    #[test]
    fn test_scatter_negative_axis() {
        let config = ScatterConfig { axis: -1 };
        let node = ScatterNodeBuilder::new("scatter1")
            .input_tensor("data", 3, DType::F32)
            .input_tensor("indices", 3, DType::I64)
            .input_tensor("updates", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code);
    }
}

