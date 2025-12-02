//! Node module contains implementations of ONNX operations.
//!
//! Each submodule implements a specific ONNX operation, providing:
//! - Operation configuration and parameters
//! - Rank inference functionality
//! - NodeProcessor implementation for node-centric processing
//!
//! This modular structure allows for clean separation of operation implementations
//! and facilitates easier maintenance and extension of the ONNX operation set.

#[cfg(test)]
pub mod test_utils;

pub mod elementwise;

// Unary element-wise operations
pub mod abs;
pub mod bitwisenot;
pub mod ceil;
pub mod cos;
pub mod cosh;
pub mod erf;
pub mod exp;
pub mod floor;
pub mod gelu;
pub mod log;
pub mod neg;
pub mod not;
pub mod reciprocal;
pub mod round;
pub mod sigmoid;
pub mod sign;
pub mod sin;
pub mod sinh;
pub mod sqrt;
pub mod tan;
pub mod tanh;

// Binary element-wise operations
pub mod and;
pub mod argmax;
pub mod argmin;
pub mod arithmetic;
pub mod attention;
pub mod avg_pool1d;
pub mod avg_pool2d;
pub mod batch_norm;
pub mod bernoulli;
pub mod bitshift;
pub mod bitwiseand;
pub mod bitwiseor;
pub mod bitwisexor;
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
pub mod eye_like;
pub mod flatten;
pub mod gather;
pub mod gather_elements;
pub mod gemm;
pub mod global_avg_pool;
pub mod grid_sample;
pub mod group_norm;
pub mod hard_sigmoid;
pub mod identity;
pub mod if_node;
pub mod instance_norm;
pub mod is_inf;
pub mod is_nan;
pub mod layer_norm;
pub mod leaky_relu;
pub mod linear;
pub mod log_softmax;
pub mod loop_node;
pub mod matmul;
pub mod matmulinteger;
pub mod max;
pub mod max_pool1d;
pub mod max_pool2d;
pub mod mean;
pub mod min;
pub mod modulo;
pub mod nonzero;
pub mod one_hot;
pub mod or;
pub mod pad;
pub mod padding;
pub mod pow;
pub mod prelu;
pub mod random;
pub mod random_like;
pub mod range;
pub mod reduce;
pub mod relu;
pub mod reshape;
pub mod resize;
pub mod scan_node;
pub mod shape;
pub mod size;
pub mod slice;
pub mod softmax;
pub mod space_to_depth;
pub mod split;
pub mod squeeze;
pub mod sum;
pub mod tile;
pub mod topk;
pub mod transpose;
pub mod trilu;
pub mod unsqueeze;
pub mod unsupported;
pub mod where_op;
pub mod xor;
