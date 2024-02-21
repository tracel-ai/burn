mod base;
mod padding;

/// WGSL vec4 primitives are used on left and right hand tensor,
/// padding is avoided through the use of conditions in the kernel
pub mod unpadded;
/// WGSL vec4 primitives are used on left and right hand tensor
pub mod vec4;
/// WGSL vec4 primitives are used on left hand tensor
pub mod vec4_lhs;
