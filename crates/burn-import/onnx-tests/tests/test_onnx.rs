// #![no_std]

// extern crate alloc;

// /// Include generated models in the `model` directory in the target directory.
// macro_rules! include_models {
//     ($($model:ident),*) => {
//         $(
//             // Allow type complexity for generated code
//             #[allow(clippy::type_complexity)]
//             pub mod $model {
//                 include!(concat!(env!("OUT_DIR"), concat!("/model/", stringify!($model), ".rs")));
//             }
//         )*
//     };
// }

// ATTENTION: Modify this macro to include all models in the `model` directory.
// Note: The following models have been moved to their own modules:
// - add, add_int -> tests/add/mod.rs
// - argmax -> tests/argmax/mod.rs
// - avg_pool1d, avg_pool2d -> tests/avg_pool/mod.rs
// - batch_norm -> tests/batch_norm/mod.rs
// - cast -> tests/cast/mod.rs
// - clip -> tests/clip/mod.rs
// - concat -> tests/concat/mod.rs
// - constant_f32, constant_f64, constant_i32, constant_i64 -> tests/constant/mod.rs
// - constant_of_shape, constant_of_shape_full_like -> tests/constant_of_shape/mod.rs
// - conv1d, conv2d, conv3d -> tests/conv/mod.rs
// - conv_transpose1d, conv_transpose2d, conv_transpose3d -> tests/conv_transpose/mod.rs
// - cos -> tests/cos/mod.rs
// - cosh -> tests/cosh/mod.rs
// - div -> tests/div/mod.rs
// - dropout -> tests/dropout/mod.rs
// - equal -> tests/equal/mod.rs
// - erf -> tests/erf/mod.rs
// - exp -> tests/exp/mod.rs
// - expand, expand_shape, expand_tensor -> tests/expand/mod.rs
// - flatten, flatten_2d -> tests/flatten/mod.rs
// - floor -> tests/floor/mod.rs
// - gather_1d_idx, gather_2d_idx, gather_elements, gather_scalar, gather_scalar_out, gather_shape -> tests/gather/mod.rs
// - gelu -> tests/gelu/mod.rs
// - gemm, gemm_no_c, gemm_non_unit_alpha_beta -> tests/gemm/mod.rs
// - global_avr_pool -> tests/global_avr_pool/mod.rs
// - graph_multiple_output_tracking -> tests/graph_multiple_output_tracking/mod.rs
// - greater, greater_scalar, greater_or_equal, greater_or_equal_scalar -> tests/greater/mod.rs
// - hard_sigmoid -> tests/hard_sigmoid/mod.rs
// - layer_norm -> tests/layer_norm/mod.rs
// - leaky_relu -> tests/leaky_relu/mod.rs
// - less, less_scalar, less_or_equal, less_or_equal_scalar -> tests/less/mod.rs
// - linear -> tests/linear/mod.rs
// - log -> tests/log/mod.rs
// - log_softmax -> tests/log_softmax/mod.rs
// - mask_where, mask_where_all_scalar, mask_where_broadcast, mask_where_scalar_x, mask_where_scalar_y -> tests/mask_where/mod.rs
// - matmul -> tests/matmul/mod.rs
// - max -> tests/max/mod.rs
// - maxpool1d, maxpool2d -> tests/maxpool/mod.rs
// - mean -> tests/mean/mod.rs
// - min -> tests/min/mod.rs
// - mul -> tests/mul/mod.rs
// - neg -> tests/neg/mod.rs
// - not -> tests/not/mod.rs
// - one_hot -> tests/one_hot/mod.rs
// - pad -> tests/pad/mod.rs
// - pow, pow_int -> tests/pow/mod.rs
// - prelu -> tests/prelu/mod.rs
// - random_normal, random_normal_like, random_uniform, random_uniform_like -> tests/random/mod.rs
// - range -> tests/range/mod.rs
// - recip -> tests/recip/mod.rs
// - reduce_max, reduce_min, reduce_mean, reduce_prod, reduce_sum -> tests/reduce/mod.rs
// - relu -> tests/relu/mod.rs
// - reshape -> tests/reshape/mod.rs
// - resize_1d_linear_scale, resize_1d_nearest_scale, resize_2d_bicubic_scale, resize_2d_bilinear_scale, resize_2d_nearest_scale, resize_with_sizes -> tests/resize/mod.rs
// - shape -> tests/shape/mod.rs
// - sigmoid -> tests/sigmoid/mod.rs
// - sign -> tests/sign/mod.rs
// - sin -> tests/sin/mod.rs
// - sinh -> tests/sinh/mod.rs
// - slice, slice_shape -> tests/slice/mod.rs
// - softmax -> tests/softmax/mod.rs
// - split -> tests/split/mod.rs
// - squeeze, squeeze_multiple -> tests/squeeze/mod.rs
// - sqrt -> tests/sqrt/mod.rs
// - sub, sub_int -> tests/sub/mod.rs
// - sum, sum_int -> tests/sum/mod.rs
// - tan -> tests/tan/mod.rs
// - tanh -> tests/tanh/mod.rs
// - tile -> tests/tile/mod.rs
// - topk -> tests/topk/mod.rs
// - transpose -> tests/transpose/mod.rs
// - trilu_lower, trilu_upper -> tests/trilu/mod.rs
// - unsqueeze_like, unsqueeze_runtime_axes -> tests/unsqueeze/mod.rs
// include_models!();

// #[cfg(test)]
// mod tests {
//     use burn::tensor::ops::FloatElem;

//     type Backend = burn_ndarray::NdArray<f32>;
//     type FT = FloatElem<Backend>;

//     // Note: All tests have been moved to separate modules.
// }
