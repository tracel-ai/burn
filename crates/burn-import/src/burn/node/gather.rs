use super::prelude::*;
use crate::burn::scalar_type_tokens;
use onnx_ir::ir::TensorDataExt;

impl NodeCodegen for onnx_ir::gather::GatherNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> proc_macro2::TokenStream {
        let input_arg = self.inputs.first().unwrap();

        match &input_arg.ty {
            ArgType::Shape(_) => forward_shape_gather(self),
            ArgType::Tensor(_) => forward_tensor_gather(self, scope),
            ArgType::Scalar(_) => forward_scalar_gather(self),
        }
    }
}

fn forward_shape_gather(node: &onnx_ir::gather::GatherNode) -> proc_macro2::TokenStream {
    let input_arg = node.inputs.first().unwrap();
    let index_arg = &node.inputs[1];
    let output_arg = node.outputs.first().unwrap();

    let input_shape_name = arg_to_ident(input_arg);
    let output = arg_to_ident(output_arg);

    match &output_arg.ty {
        ArgType::Scalar(dtype) => {
            // Gathering a single element from a shape produces a scalar
            let scalar_ty = scalar_type_tokens(dtype);
            match &index_arg.ty {
                ArgType::Scalar(_) => {
                    let index = arg_to_ident(index_arg);
                    // Handle negative indices properly for runtime scalars
                    quote! {
                        let actual_idx = if #index < 0 {
                            (#input_shape_name.len() as i64 + #index) as usize
                        } else {
                            #index as usize
                        };
                        let #output = #input_shape_name[actual_idx] as #scalar_ty;
                    }
                }
                _ => panic!(
                    "Gather from Shape to Scalar needs scalar index, got {:?}!",
                    index_arg.ty
                ),
            }
        }
        ArgType::Shape(out_rank) => {
            match &index_arg.ty {
                ArgType::Tensor(idx_tensor) => {
                    let index = arg_to_ident(index_arg);
                    let index_rank = idx_tensor.rank;
                    let output_rank = out_rank;

                    if index_rank == 1 {
                        // Handle negative indices properly for runtime tensors
                        quote! {
                            let #output: [i64; #output_rank] = #index.to_data()
                                .iter::<i64>()
                                .map(|idx| {
                                    let actual_idx = if idx < 0 {
                                        (#input_shape_name.len() as i64 + idx) as usize
                                    } else {
                                        idx as usize
                                    };
                                    #input_shape_name[actual_idx]
                                })
                                .collect::<alloc::vec::Vec<_>>()
                                .try_into()
                                .unwrap();
                        }
                    } else {
                        panic!(
                            "Multi-dimensional indices for Shape gather should be 1-dimensional, but got rank {}",
                            index_rank
                        );
                    }
                }
                ArgType::Shape(_idx_rank) => {
                    // Shape indices for gathering from Shape
                    let index_name = arg_to_ident(index_arg);
                    let output_rank = out_rank;

                    // Handle negative indices properly for runtime shape indices
                    quote! {
                        let #output: [i64; #output_rank] = #index_name
                            .iter()
                            .map(|&idx| {
                                let actual_idx = if idx < 0 {
                                    (#input_shape_name.len() as i64 + idx) as usize
                                } else {
                                    idx as usize
                                };
                                #input_shape_name[actual_idx]
                            })
                            .collect::<alloc::vec::Vec<_>>()
                            .try_into()
                            .unwrap();
                    }
                }
                _ => panic!(
                    "Gather from Shape to Shape needs Tensor or Shape index, got {:?}!",
                    index_arg.ty
                ),
            }
        }
        _ => panic!(
            "Gather from Shape input can only output Shape or Scalar, got {:?}!",
            output_arg.ty
        ),
    }
}

/// Forward for scalar input - just pass through the scalar value
/// When gathering from a scalar, there's only one element, so the result is the scalar itself
fn forward_scalar_gather(node: &onnx_ir::gather::GatherNode) -> proc_macro2::TokenStream {
    let input_arg = node.inputs.first().unwrap();
    let output_arg = node.outputs.first().unwrap();

    let input = arg_to_ident(input_arg);
    let output = arg_to_ident(output_arg);

    // A scalar only has one element, so gathering from it just returns the scalar
    // This handles cases like Reshape(scalar, [-1]) -> Scalar followed by Gather
    quote! {
        let #output = #input;
    }
}

fn forward_tensor_gather(
    node: &onnx_ir::gather::GatherNode,
    scope: &mut super::super::scope::ScopeAtPosition<'_>,
) -> proc_macro2::TokenStream {
    let dim = node.config.axis.to_tokens();
    let input_arg = node.inputs.first().unwrap();
    let index_arg = &node.inputs[1];
    let output_arg = node.outputs.first().unwrap();

    let input_rank = match &input_arg.ty {
        ArgType::Tensor(tensor) => tensor.rank,
        _ => unreachable!(),
    };
    let input = scope.arg(input_arg);
    let output = arg_to_ident(output_arg);

    match &output_arg.ty {
        ArgType::Scalar(dtype) => {
            // Gathering a single element from a tensor produces a scalar
            let scalar_ty = scalar_type_tokens(dtype);
            match &index_arg.ty {
                ArgType::Scalar(_) => {
                    let index = arg_to_ident(index_arg);
                    quote! {
                        let #output = {
                            let indices = Tensor::<B, 1, _>::from_data([#index], &*self.device);
                            let selected = Tensor::select(#input, #dim, indices);
                            selected.into_scalar().elem::<#scalar_ty>()
                        };
                    }
                }
                _ => panic!(
                    "Gather from Tensor to Scalar needs scalar index, got {:?}!",
                    index_arg.ty
                ),
            }
        }
        ArgType::Tensor(_) => {
            match &index_arg.ty {
                ArgType::Scalar(_) => {
                    // Use tensor.slice(...) for efficient gather operation
                    let output_rank = input_rank - 1;
                    let axis = node.config.axis;

                    // Either a resolved literal or the runtime variable
                    let index_token = index_arg
                        .value()
                        .and_then(|v| v.scalar_i64().ok())
                        .map(|resolved| {
                            let lit = proc_macro2::Literal::i64_unsuffixed(resolved);
                            quote! { #lit }
                        })
                        .unwrap_or_else(|| {
                            let index = arg_to_ident(index_arg);
                            quote! { #index }
                        });

                    let slice_args = (0..input_rank)
                        .map(|i| {
                            if i == axis {
                                index_token.clone()
                            } else {
                                quote! { .. }
                            }
                        })
                        .collect::<Vec<_>>();

                    quote! {
                        let #output = {
                            let sliced = #input.slice(s![#(#slice_args),*]);
                            sliced.squeeze_dim::<#output_rank>(#dim)
                        };
                    }
                }
                ArgType::Tensor(idx_tensor) => {
                    let index = scope.arg(index_arg);
                    let index_rank = idx_tensor.rank;
                    let output_rank = index_rank + input_rank - 1;
                    let final_rank = output_rank.max(1); // Ensure minimum rank of 1

                    // Use proc_macro2::Literal to avoid usize suffix
                    let index_rank_lit = proc_macro2::Literal::usize_unsuffixed(index_rank);
                    let final_rank_lit = proc_macro2::Literal::usize_unsuffixed(final_rank);

                    quote! {
                        let #output = #input.take::<#index_rank_lit, #final_rank_lit>(#dim, #index);
                    }
                }
                ArgType::Shape(_) => {
                    let shape_name = arg_to_ident(index_arg);

                    // Shape array can be directly used to create tensor data
                    quote! {
                        let #output = {
                            let indices = Tensor::<B, 1, _>::from_data(#shape_name, &*self.device);
                            Tensor::select(#input, #dim, indices)
                        };
                    }
                }
            }
        }
        _ => panic!("Gather needs Tensor output, got {:?}!", output_arg.ty),
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::gather::{GatherConfig, GatherNodeBuilder};

    // ==================== Shape Gather Tests ====================

    #[test]
    fn test_gather_shape_to_scalar_i32() {
        let config = GatherConfig { axis: 0 };
        let node = GatherNodeBuilder::new("extract_dim")
            .input_shape("input_shape")
            .input_scalar("dim_idx", DType::I32)
            .output_scalar("dim_value", DType::I32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input_shape: [i64; 1], dim_idx: i32) -> i32 {
            let actual_idx = if dim_idx < 0 {
                (input_shape.len() as i64 + dim_idx) as usize
            } else {
                dim_idx as usize
            };
            let dim_value = input_shape[actual_idx] as i32;
            dim_value
        }
        ");
    }

    #[test]
    fn test_gather_shape_to_scalar_i64() {
        let config = GatherConfig { axis: 0 };
        let node = GatherNodeBuilder::new("get_batch_size")
            .input_shape("shape_arr")
            .input_scalar("position", DType::I64)
            .output_scalar("size", DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, shape_arr: [i64; 1], position: i64) -> i64 {
            let actual_idx = if position < 0 {
                (shape_arr.len() as i64 + position) as usize
            } else {
                position as usize
            };
            let size = shape_arr[actual_idx] as i64;
            size
        }
        ");
    }

    #[test]
    fn test_gather_shape_to_shape_tensor_index() {
        let config = GatherConfig { axis: 0 };
        let node = GatherNodeBuilder::new("select_dims")
            .input_shape("full_shape")
            .input_tensor("dim_indices", 1, DType::I64)
            .output_shape("selected_shape")
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, full_shape: [i64; 1], dim_indices: Tensor<B, 1, Int>) -> [i64; 1] {
            let selected_shape: [i64; 1usize] = dim_indices
                .to_data()
                .iter::<i64>()
                .map(|idx| {
                    let actual_idx = if idx < 0 {
                        (full_shape.len() as i64 + idx) as usize
                    } else {
                        idx as usize
                    };
                    full_shape[actual_idx]
                })
                .collect::<alloc::vec::Vec<_>>()
                .try_into()
                .unwrap();
            selected_shape
        }
        ");
    }

    #[test]
    fn test_gather_shape_to_shape_tensor_index_rank3() {
        let config = GatherConfig { axis: 0 };
        let node = GatherNodeBuilder::new("pick_dimensions")
            .input_shape("dimensions")
            .input_tensor("choices", 1, DType::I64)
            .output_shape("result_dims")
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, dimensions: [i64; 1], choices: Tensor<B, 1, Int>) -> [i64; 1] {
            let result_dims: [i64; 1usize] = choices
                .to_data()
                .iter::<i64>()
                .map(|idx| {
                    let actual_idx = if idx < 0 {
                        (dimensions.len() as i64 + idx) as usize
                    } else {
                        idx as usize
                    };
                    dimensions[actual_idx]
                })
                .collect::<alloc::vec::Vec<_>>()
                .try_into()
                .unwrap();
            result_dims
        }
        ");
    }

    #[test]
    fn test_gather_shape_to_shape_shape_index() {
        let config = GatherConfig { axis: 0 };
        let node = GatherNodeBuilder::new("reorder_shape")
            .input_shape("original")
            .input_shape("indices")
            .output_shape("reordered")
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, original: [i64; 1], indices: [i64; 1]) -> [i64; 1] {
            let reordered: [i64; 1usize] = indices
                .iter()
                .map(|&idx| {
                    let actual_idx = if idx < 0 {
                        (original.len() as i64 + idx) as usize
                    } else {
                        idx as usize
                    };
                    original[actual_idx]
                })
                .collect::<alloc::vec::Vec<_>>()
                .try_into()
                .unwrap();
            reordered
        }
        ");
    }

    #[test]
    fn test_gather_shape_to_shape_shape_index_rank2() {
        let config = GatherConfig { axis: 0 };
        let node = GatherNodeBuilder::new("transpose_dims")
            .input_shape("shape_vec")
            .input_shape("order")
            .output_shape("transposed")
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, shape_vec: [i64; 1], order: [i64; 1]) -> [i64; 1] {
            let transposed: [i64; 1usize] = order
                .iter()
                .map(|&idx| {
                    let actual_idx = if idx < 0 {
                        (shape_vec.len() as i64 + idx) as usize
                    } else {
                        idx as usize
                    };
                    shape_vec[actual_idx]
                })
                .collect::<alloc::vec::Vec<_>>()
                .try_into()
                .unwrap();
            transposed
        }
        ");
    }

    // ==================== Tensor Gather to Scalar Tests ====================

    #[test]
    fn test_gather_tensor_to_scalar_axis0() {
        let config = GatherConfig { axis: 0 };
        let node = GatherNodeBuilder::new("extract_elem")
            .input_tensor("values", 2, DType::F32)
            .input_scalar("idx", DType::I32)
            .output_scalar("elem", DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, values: Tensor<B, 2>, idx: i32) -> f32 {
            let elem = {
                let indices = Tensor::<B, 1, _>::from_data([idx], &*self.device);
                let selected = Tensor::select(values, 0, indices);
                selected.into_scalar().elem::<f32>()
            };
            elem
        }
        ");
    }

    #[test]
    fn test_gather_tensor_to_scalar_axis1() {
        let config = GatherConfig { axis: 1 };
        let node = GatherNodeBuilder::new("get_value")
            .input_tensor("matrix", 3, DType::F64)
            .input_scalar("col_idx", DType::I64)
            .output_scalar("value", DType::F64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, matrix: Tensor<B, 3>, col_idx: i64) -> f64 {
            let value = {
                let indices = Tensor::<B, 1, _>::from_data([col_idx], &*self.device);
                let selected = Tensor::select(matrix, 1, indices);
                selected.into_scalar().elem::<f64>()
            };
            value
        }
        ");
    }

    #[test]
    fn test_gather_tensor_to_scalar_i32() {
        let config = GatherConfig { axis: 0 };
        let node = GatherNodeBuilder::new("pick_int")
            .input_tensor("int_array", 1, DType::I32)
            .input_scalar("position", DType::I32)
            .output_scalar("result", DType::I32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, int_array: Tensor<B, 1, Int>, position: i32) -> i32 {
            let result = {
                let indices = Tensor::<B, 1, _>::from_data([position], &*self.device);
                let selected = Tensor::select(int_array, 0, indices);
                selected.into_scalar().elem::<i32>()
            };
            result
        }
        ");
    }

    // ==================== Tensor Gather to Tensor - Scalar Index Tests ====================

    #[test]
    fn test_gather_tensor_scalar_index_axis0() {
        let config = GatherConfig { axis: 0 };
        let node = GatherNodeBuilder::new("select_row")
            .input_tensor("table", 3, DType::F32)
            .input_scalar("row_idx", DType::I32)
            .output_tensor("row", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, table: Tensor<B, 3>, row_idx: i32) -> Tensor<B, 2> {
            let row = {
                let sliced = table.slice(s![row_idx, .., ..]);
                sliced.squeeze_dim::<2usize>(0)
            };
            row
        }
        ");
    }

    #[test]
    fn test_gather_tensor_scalar_index_axis1() {
        let config = GatherConfig { axis: 1 };
        let node = GatherNodeBuilder::new("extract_col")
            .input_tensor("data", 2, DType::F32)
            .input_scalar("col_num", DType::I64)
            .output_tensor("column", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<B, 2>, col_num: i64) -> Tensor<B, 1> {
            let column = {
                let sliced = data.slice(s![.., col_num]);
                sliced.squeeze_dim::<1usize>(1)
            };
            column
        }
        ");
    }

    #[test]
    fn test_gather_tensor_scalar_index_axis2() {
        let config = GatherConfig { axis: 2 };
        let node = GatherNodeBuilder::new("slice_depth")
            .input_tensor("volume", 4, DType::F32)
            .input_scalar("depth_idx", DType::I32)
            .output_tensor("slice", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, volume: Tensor<B, 4>, depth_idx: i32) -> Tensor<B, 3> {
            let slice = {
                let sliced = volume.slice(s![.., .., depth_idx, ..]);
                sliced.squeeze_dim::<3usize>(2)
            };
            slice
        }
        ");
    }

    #[test]
    fn test_gather_tensor_scalar_index_rank4_axis3() {
        let config = GatherConfig { axis: 3 };
        let node = GatherNodeBuilder::new("pick_channel")
            .input_tensor("features", 4, DType::F64)
            .input_scalar("ch_idx", DType::I64)
            .output_tensor("channel", 3, DType::F64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, features: Tensor<B, 4>, ch_idx: i64) -> Tensor<B, 3> {
            let channel = {
                let sliced = features.slice(s![.., .., .., ch_idx]);
                sliced.squeeze_dim::<3usize>(3)
            };
            channel
        }
        ");
    }

    // ==================== Tensor Gather to Tensor - Tensor Index Tests ====================

    #[test]
    fn test_gather_tensor_tensor_index_axis0() {
        let config = GatherConfig { axis: 0 };
        let node = GatherNodeBuilder::new("gather_rows")
            .input_tensor("embedding", 2, DType::F32)
            .input_tensor("row_indices", 1, DType::I64)
            .output_tensor("gathered", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            embedding: Tensor<B, 2>,
            row_indices: Tensor<B, 1, Int>,
        ) -> Tensor<B, 2> {
            let gathered = embedding.take::<1, 2>(0, row_indices);
            gathered
        }
        ");
    }

    #[test]
    fn test_gather_tensor_tensor_index_axis1() {
        let config = GatherConfig { axis: 1 };
        let node = GatherNodeBuilder::new("select_features")
            .input_tensor("feature_map", 3, DType::F32)
            .input_tensor("feature_ids", 1, DType::I64)
            .output_tensor("selected_features", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            feature_map: Tensor<B, 3>,
            feature_ids: Tensor<B, 1, Int>,
        ) -> Tensor<B, 3> {
            let selected_features = feature_map.take::<1, 3>(1, feature_ids);
            selected_features
        }
        ");
    }

    #[test]
    fn test_gather_tensor_tensor_index_rank2_idx() {
        let config = GatherConfig { axis: 0 };
        let node = GatherNodeBuilder::new("batch_gather")
            .input_tensor("source", 3, DType::F32)
            .input_tensor("indices_2d", 2, DType::I64)
            .output_tensor("result", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            source: Tensor<B, 3>,
            indices_2d: Tensor<B, 2, Int>,
        ) -> Tensor<B, 4> {
            let result = source.take::<2, 4>(0, indices_2d);
            result
        }
        ");
    }

    #[test]
    fn test_gather_tensor_tensor_index_rank3_idx() {
        let config = GatherConfig { axis: 1 };
        let node = GatherNodeBuilder::new("multi_gather")
            .input_tensor("input_data", 4, DType::F64)
            .input_tensor("index_tensor", 3, DType::I64)
            .output_tensor("output_data", 6, DType::F64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input_data: Tensor<B, 4>,
            index_tensor: Tensor<B, 3, Int>,
        ) -> Tensor<B, 6> {
            let output_data = input_data.take::<3, 6>(1, index_tensor);
            output_data
        }
        ");
    }

    // ==================== Scalar Input Gather Tests ====================

    #[test]
    fn test_gather_scalar_input_i64() {
        let config = GatherConfig { axis: 0 };
        let node = GatherNodeBuilder::new("pass_through")
            .input_scalar("scalar_val", DType::I64)
            .input_scalar("idx", DType::I64)
            .output_scalar("result", DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, scalar_val: i64, idx: i64) -> i64 {
            let result = scalar_val;
            result
        }
        ");
    }

    #[test]
    fn test_gather_scalar_input_f32() {
        let config = GatherConfig { axis: 0 };
        let node = GatherNodeBuilder::new("get_scalar")
            .input_scalar("value", DType::F32)
            .input_scalar("index", DType::I32)
            .output_scalar("output", DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, value: f32, index: i32) -> f32 {
            let output = value;
            output
        }
        ");
    }

    // ==================== Tensor Gather to Tensor - Shape Index Tests ====================

    #[test]
    fn test_gather_tensor_shape_index_axis0() {
        let config = GatherConfig { axis: 0 };
        let node = GatherNodeBuilder::new("gather_by_shape")
            .input_tensor("weights", 2, DType::F32)
            .input_shape("positions")
            .output_tensor("selected_weights", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, weights: Tensor<B, 2>, positions: [i64; 1]) -> Tensor<B, 2> {
            let selected_weights = {
                let indices = Tensor::<B, 1, _>::from_data(positions, &*self.device);
                Tensor::select(weights, 0, indices)
            };
            selected_weights
        }
        ");
    }

    #[test]
    fn test_gather_tensor_shape_index_axis1() {
        let config = GatherConfig { axis: 1 };
        let node = GatherNodeBuilder::new("index_columns")
            .input_tensor("matrix_data", 3, DType::F64)
            .input_shape("col_indices")
            .output_tensor("columns", 3, DType::F64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, matrix_data: Tensor<B, 3>, col_indices: [i64; 1]) -> Tensor<B, 3> {
            let columns = {
                let indices = Tensor::<B, 1, _>::from_data(col_indices, &*self.device);
                Tensor::select(matrix_data, 1, indices)
            };
            columns
        }
        ");
    }

    #[test]
    fn test_gather_tensor_shape_index_axis2() {
        let config = GatherConfig { axis: 2 };
        let node = GatherNodeBuilder::new("select_planes")
            .input_tensor("tensor3d", 4, DType::F32)
            .input_shape("plane_ids")
            .output_tensor("planes", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, tensor3d: Tensor<B, 4>, plane_ids: [i64; 1]) -> Tensor<B, 4> {
            let planes = {
                let indices = Tensor::<B, 1, _>::from_data(plane_ids, &*self.device);
                Tensor::select(tensor3d, 2, indices)
            };
            planes
        }
        ");
    }

    #[test]
    fn test_gather_const_positive_index() {
        let config = GatherConfig { axis: 0 };
        let node = GatherNodeBuilder::new("get_row")
            .input_tensor_shape("data", vec![10, 64], DType::F32)
            .input_const_i64("idx", 2)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<B, 2>) -> Tensor<B, 1> {
            let output = {
                let sliced = data.slice(s![2, ..]);
                sliced.squeeze_dim::<1usize>(0)
            };
            output
        }
        ");
    }

    #[test]
    fn test_gather_const_negative_index() {
        let config = GatherConfig { axis: 0 };
        let node = GatherNodeBuilder::new("get_last_row")
            .input_tensor_shape("data", vec![10, 64], DType::F32)
            .input_const_i64("idx", -1)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<B, 2>) -> Tensor<B, 1> {
            let output = {
                let sliced = data.slice(s![- 1, ..]);
                sliced.squeeze_dim::<1usize>(0)
            };
            output
        }
        ");
    }

    #[test]
    fn test_gather_const_negative_index_axis1() {
        let config = GatherConfig { axis: 1 };
        let node = GatherNodeBuilder::new("get_last_col")
            .input_tensor_shape("data", vec![8, 64], DType::F32)
            .input_const_i64("idx", -1)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<B, 2>) -> Tensor<B, 1> {
            let output = {
                let sliced = data.slice(s![.., - 1]);
                sliced.squeeze_dim::<1usize>(1)
            };
            output
        }
        ");
    }

    #[test]
    fn test_gather_const_negative_index_minus2() {
        let config = GatherConfig { axis: 0 };
        let node = GatherNodeBuilder::new("get_second_last")
            .input_tensor_shape("data", vec![10, 64], DType::F32)
            .input_const_i64("idx", -2)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<B, 2>) -> Tensor<B, 1> {
            let output = {
                let sliced = data.slice(s![- 2, ..]);
                sliced.squeeze_dim::<1usize>(0)
            };
            output
        }
        ");
    }
}
