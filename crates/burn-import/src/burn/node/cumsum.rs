use super::prelude::*;
use onnx_ir::cumsum::CumSumAxis;

impl NodeCodegen for onnx_ir::cumsum::CumSumNode {
    fn inputs(&self) -> &[Argument] {
        // Only the data input (inputs[0]), not axis (handled in config)
        &self.inputs[..1]
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let exclusive = self.config.exclusive;
        let reverse = self.config.reverse;

        // Extract axis value (static only for now)
        let axis = match &self.config.axis {
            CumSumAxis::Static(axis) => axis.to_tokens(),
            CumSumAxis::Runtime(_) => {
                panic!("Runtime CumSum axis not yet supported in burn-import")
            }
        };

        match (exclusive, reverse) {
            (false, false) => {
                // Default: simple cumsum
                quote! {
                    let #output = #input.cumsum(#axis);
                }
            }
            (false, true) => {
                // Reverse only: flip along axis, cumsum, flip back
                quote! {
                    let #output = #input.flip([#axis]).cumsum(#axis).flip([#axis]);
                }
            }
            (true, false) => {
                // Exclusive only: cumsum, then shift (prepend zeros, drop last)
                // exclusive[i] = sum(input[0..i]) (excludes current element)
                // Use block scope for temporary variables
                quote! {
                    let #output = {
                        let cumsum_result = #input.cumsum(#axis);
                        let shape = cumsum_result.shape();
                        let dim_size = shape.dims[#axis];
                        let sliced = cumsum_result.narrow(#axis, 0, dim_size - 1);
                        let zeros = sliced.zeros_like().narrow(#axis, 0, 1);
                        Tensor::cat(vec![zeros, sliced], #axis)
                    };
                }
            }
            (true, true) => {
                // Both exclusive and reverse
                // Reverse cumsum: output[i] = sum(input[i+1..n])
                // Use block scope for temporary variables
                quote! {
                    let #output = {
                        let flipped = #input.flip([#axis]);
                        let cumsum_result = flipped.cumsum(#axis);
                        let cumsum_back = cumsum_result.flip([#axis]);
                        let shape = cumsum_back.shape();
                        let dim_size = shape.dims[#axis];
                        let sliced = cumsum_back.narrow(#axis, 1, dim_size - 1);
                        let zeros = sliced.zeros_like().narrow(#axis, 0, 1);
                        Tensor::cat(vec![sliced, zeros], #axis)
                    };
                }
            }
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        // Tensor is needed for Tensor::cat in exclusive mode
        if self.config.exclusive {
            imports.register("burn::tensor::Tensor");
            imports.register("alloc::vec");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use super::CumSumAxis;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::cumsum::{CumSumConfig, CumSumNode, CumSumNodeBuilder};

    fn create_cumsum_node(
        name: &str,
        axis: usize,
        exclusive: bool,
        reverse: bool,
        rank: usize,
    ) -> CumSumNode {
        let config = CumSumConfig::new(CumSumAxis::Static(axis), exclusive, reverse);

        CumSumNodeBuilder::new(name)
            .input_tensor("input", rank, DType::F32)
            .input_tensor("axis", 0, DType::I64)
            .output_tensor("output", rank, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_cumsum_default() {
        let node = create_cumsum_node("cumsum1", 0, false, false, 1);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
            let output = input.cumsum(0);
            output
        }
        ");
    }

    #[test]
    fn test_cumsum_axis_1() {
        let node = create_cumsum_node("cumsum1", 1, false, false, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input.cumsum(1);
            output
        }
        ");
    }

    #[test]
    fn test_cumsum_reverse() {
        let node = create_cumsum_node("cumsum1", 0, false, true, 1);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
            let output = input.flip([0]).cumsum(0).flip([0]);
            output
        }
        ");
    }

    #[test]
    fn test_cumsum_exclusive() {
        let node = create_cumsum_node("cumsum1", 0, true, false, 1);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
            let output = {
                let cumsum_result = input.cumsum(0);
                let shape = cumsum_result.shape();
                let dim_size = shape.dims[0];
                let sliced = cumsum_result.narrow(0, 0, dim_size - 1);
                let zeros = sliced.zeros_like().narrow(0, 0, 1);
                Tensor::cat(vec![zeros, sliced], 0)
            };
            output
        }
        ");
    }

    #[test]
    fn test_cumsum_exclusive_reverse() {
        let node = create_cumsum_node("cumsum1", 0, true, true, 1);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
            let output = {
                let flipped = input.flip([0]);
                let cumsum_result = flipped.cumsum(0);
                let cumsum_back = cumsum_result.flip([0]);
                let shape = cumsum_back.shape();
                let dim_size = shape.dims[0];
                let sliced = cumsum_back.narrow(0, 1, dim_size - 1);
                let zeros = sliced.zeros_like().narrow(0, 0, 1);
                Tensor::cat(vec![sliced, zeros], 0)
            };
            output
        }
        ");
    }

    #[test]
    fn test_cumsum_reverse_axis_1() {
        let node = create_cumsum_node("cumsum1", 1, false, true, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input.flip([1]).cumsum(1).flip([1]);
            output
        }
        ");
    }

    #[test]
    fn test_cumsum_exclusive_axis_1() {
        let node = create_cumsum_node("cumsum1", 1, true, false, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let cumsum_result = input.cumsum(1);
                let shape = cumsum_result.shape();
                let dim_size = shape.dims[1];
                let sliced = cumsum_result.narrow(1, 0, dim_size - 1);
                let zeros = sliced.zeros_like().narrow(1, 0, 1);
                Tensor::cat(vec![zeros, sliced], 1)
            };
            output
        }
        ");
    }
}
