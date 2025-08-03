//! Node module contains implementations of ONNX operations.
//!
//! Each submodule implements a specific ONNX operation, providing:
//! - Operation configuration and parameters
//! - Rank inference functionality
//!
//! This modular structure allows for clean separation of operation implementations
//! and facilitates easier maintenance and extension of the ONNX operation set.

#[cfg(test)]
pub mod test_utils;

pub mod argmax;
pub mod argmin;
pub mod attention;
pub mod avg_pool1d;
pub mod avg_pool2d;
pub mod batch_norm;
pub mod bernoulli;
pub mod bitshift;
pub mod cast;
pub mod clip;
pub mod comparison;
pub mod concat;
pub mod constant;
pub mod constant_of_shape;
pub mod conv1d;
pub mod conv2d;
pub mod conv3d;
pub mod conv_transpose1d;
pub mod conv_transpose2d;
pub mod conv_transpose3d;
pub mod depth_to_space;
pub mod dropout;
pub mod expand;
pub mod flatten;
pub mod gather;
pub mod gemm;
pub mod group_norm;
pub mod hard_sigmoid;
pub mod instance_norm;
pub mod is_inf;
pub mod layer_norm;
pub mod leaky_relu;
pub mod linear;
pub mod log_softmax;
pub mod matmul;
pub mod max_pool1d;
pub mod max_pool2d;
pub mod one_hot;
pub mod pad;
pub mod padding;
pub mod random;
pub mod random_like;
pub mod range;
pub mod reduce_max;
pub mod reduce_mean;
pub mod reduce_min;
pub mod reduce_prod;
pub mod reduce_sum;
pub mod reshape;
pub mod resize;
pub mod shape;
pub mod size;
pub mod slice;
pub mod softmax;
pub mod space_to_depth;
pub mod split;
pub mod squeeze;
pub mod tile;
pub mod topk;
pub mod transpose;
pub mod trilu;
pub mod unsqueeze;
pub mod where_op;
