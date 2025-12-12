use super::prelude::*;
use onnx_ir::cumsum::CumSumAxis;
use onnx_ir::ir::ArgType;

impl NodeCodegen for onnx_ir::cumsum::CumSumNode {
    fn inputs(&self) -> &[Argument] {
        match &self.config.axis {
            // Static axis: only data input needed
            CumSumAxis::Static(_) => &self.inputs[..1],
            // Runtime axis: include both data and axis inputs
            CumSumAxis::Runtime(_) => &self.inputs,
        }
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let exclusive = self.config.exclusive;
        let reverse = self.config.reverse;

        match &self.config.axis {
            CumSumAxis::Static(axis) => {
                generate_static_cumsum(&input, &output, *axis, exclusive, reverse)
            }
            CumSumAxis::Runtime(_) => {
                let axis_input = arg_to_ident(&self.inputs[1]);
                // Check if axis is a scalar or a shape array
                let axis_is_scalar = matches!(&self.inputs[1].ty, ArgType::Scalar(_));
                generate_runtime_cumsum(
                    &input,
                    &output,
                    &axis_input,
                    exclusive,
                    reverse,
                    axis_is_scalar,
                )
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

/// Generate code for static (compile-time known) axis
fn generate_static_cumsum(
    input: &TokenStream,
    output: &syn::Ident,
    axis: usize,
    exclusive: bool,
    reverse: bool,
) -> TokenStream {
    let axis = axis.to_tokens();

    match (exclusive, reverse) {
        (false, false) => {
            quote! {
                let #output = #input.cumsum(#axis);
            }
        }
        (false, true) => {
            quote! {
                let #output = #input.flip([#axis]).cumsum(#axis).flip([#axis]);
            }
        }
        (true, false) => {
            // Exclusive: output[i] = sum(input[0..i]), excludes current element
            // Shift cumsum right by prepending zeros and dropping last element
            quote! {
                let #output = {
                    let cumsum_result = #input.cumsum(#axis);
                    let shape = cumsum_result.shape();
                    let dim_size = shape.dims[#axis];
                    if dim_size <= 1 {
                        // For empty or single element, exclusive cumsum is all zeros
                        cumsum_result.zeros_like()
                    } else {
                        let sliced = cumsum_result.narrow(#axis, 0, dim_size - 1);
                        let zeros = sliced.zeros_like().narrow(#axis, 0, 1);
                        Tensor::cat(vec![zeros, sliced], #axis)
                    }
                };
            }
        }
        (true, true) => {
            // Exclusive + Reverse: output[i] = sum(input[i+1..n])
            // Reverse cumsum, then shift left by dropping first and appending zeros
            quote! {
                let #output = {
                    let flipped = #input.flip([#axis]);
                    let cumsum_result = flipped.cumsum(#axis);
                    let cumsum_back = cumsum_result.flip([#axis]);
                    let shape = cumsum_back.shape();
                    let dim_size = shape.dims[#axis];
                    if dim_size <= 1 {
                        // For empty or single element, exclusive reverse cumsum is all zeros
                        cumsum_back.zeros_like()
                    } else {
                        let sliced = cumsum_back.narrow(#axis, 1, dim_size - 1);
                        let zeros = sliced.zeros_like().narrow(#axis, 0, 1);
                        Tensor::cat(vec![sliced, zeros], #axis)
                    }
                };
            }
        }
    }
}

/// Generate code for runtime axis - uses axis value directly since Burn methods accept runtime usize
fn generate_runtime_cumsum(
    input: &TokenStream,
    output: &syn::Ident,
    axis_input: &syn::Ident,
    exclusive: bool,
    reverse: bool,
    axis_is_scalar: bool,
) -> TokenStream {
    // Generate axis extraction code based on whether it's a scalar or array
    let axis_expr = if axis_is_scalar {
        quote! { #axis_input as usize }
    } else {
        quote! { #axis_input[0] as usize }
    };

    match (exclusive, reverse) {
        (false, false) => {
            quote! {
                let #output = #input.cumsum(#axis_expr);
            }
        }
        (false, true) => {
            quote! {
                let #output = {
                    let axis = #axis_expr;
                    #input.flip([axis]).cumsum(axis).flip([axis])
                };
            }
        }
        (true, false) => {
            // Exclusive: output[i] = sum(input[0..i]), excludes current element
            quote! {
                let #output = {
                    let axis = #axis_expr;
                    let cumsum_result = #input.cumsum(axis);
                    let shape = cumsum_result.shape();
                    let dim_size = shape.dims[axis];
                    if dim_size <= 1 {
                        // For empty or single element, exclusive cumsum is all zeros
                        cumsum_result.zeros_like()
                    } else {
                        let sliced = cumsum_result.narrow(axis, 0, dim_size - 1);
                        let zeros = sliced.zeros_like().narrow(axis, 0, 1);
                        Tensor::cat(vec![zeros, sliced], axis)
                    }
                };
            }
        }
        (true, true) => {
            // Exclusive + Reverse: output[i] = sum(input[i+1..n])
            quote! {
                let #output = {
                    let axis = #axis_expr;
                    let flipped = #input.flip([axis]);
                    let cumsum_result = flipped.cumsum(axis);
                    let cumsum_back = cumsum_result.flip([axis]);
                    let shape = cumsum_back.shape();
                    let dim_size = shape.dims[axis];
                    if dim_size <= 1 {
                        // For empty or single element, exclusive reverse cumsum is all zeros
                        cumsum_back.zeros_like()
                    } else {
                        let sliced = cumsum_back.narrow(axis, 1, dim_size - 1);
                        let zeros = sliced.zeros_like().narrow(axis, 0, 1);
                        Tensor::cat(vec![sliced, zeros], axis)
                    }
                };
            }
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
    use onnx_ir::ir::RuntimeInputRef;

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

    fn create_runtime_cumsum_node(
        name: &str,
        exclusive: bool,
        reverse: bool,
        rank: usize,
    ) -> CumSumNode {
        let config = CumSumConfig::new(
            CumSumAxis::Runtime(RuntimeInputRef::new("axis".to_string(), 1)),
            exclusive,
            reverse,
        );

        CumSumNodeBuilder::new(name)
            .input_tensor("input", rank, DType::F32)
            .input_shape("axis") // Shape type for runtime axis (rank 1 by default)
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
                if dim_size <= 1 {
                    cumsum_result.zeros_like()
                } else {
                    let sliced = cumsum_result.narrow(0, 0, dim_size - 1);
                    let zeros = sliced.zeros_like().narrow(0, 0, 1);
                    Tensor::cat(vec![zeros, sliced], 0)
                }
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
                if dim_size <= 1 {
                    cumsum_back.zeros_like()
                } else {
                    let sliced = cumsum_back.narrow(0, 1, dim_size - 1);
                    let zeros = sliced.zeros_like().narrow(0, 0, 1);
                    Tensor::cat(vec![sliced, zeros], 0)
                }
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
                if dim_size <= 1 {
                    cumsum_result.zeros_like()
                } else {
                    let sliced = cumsum_result.narrow(1, 0, dim_size - 1);
                    let zeros = sliced.zeros_like().narrow(1, 0, 1);
                    Tensor::cat(vec![zeros, sliced], 1)
                }
            };
            output
        }
        ");
    }

    #[test]
    fn test_cumsum_runtime_axis_rank2() {
        let node = create_runtime_cumsum_node("cumsum1", false, false, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>, axis: [i64; 1]) -> Tensor<B, 2> {
            let output = input.cumsum(axis[0] as usize);
            output
        }
        ");
    }

    #[test]
    fn test_cumsum_runtime_axis_rank3() {
        let node = create_runtime_cumsum_node("cumsum1", false, false, 3);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>, axis: [i64; 1]) -> Tensor<B, 3> {
            let output = input.cumsum(axis[0] as usize);
            output
        }
        ");
    }

    #[test]
    fn test_cumsum_runtime_axis_exclusive() {
        let node = create_runtime_cumsum_node("cumsum1", true, false, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>, axis: [i64; 1]) -> Tensor<B, 2> {
            let output = {
                let axis = axis[0] as usize;
                let cumsum_result = input.cumsum(axis);
                let shape = cumsum_result.shape();
                let dim_size = shape.dims[axis];
                if dim_size <= 1 {
                    cumsum_result.zeros_like()
                } else {
                    let sliced = cumsum_result.narrow(axis, 0, dim_size - 1);
                    let zeros = sliced.zeros_like().narrow(axis, 0, 1);
                    Tensor::cat(vec![zeros, sliced], axis)
                }
            };
            output
        }
        ");
    }

    #[test]
    fn test_cumsum_runtime_axis_reverse() {
        let node = create_runtime_cumsum_node("cumsum1", false, true, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>, axis: [i64; 1]) -> Tensor<B, 2> {
            let output = {
                let axis = axis[0] as usize;
                input.flip([axis]).cumsum(axis).flip([axis])
            };
            output
        }
        ");
    }
}
