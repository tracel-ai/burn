mod base;

#[cfg(feature = "export_tests")]
#[allow(missing_docs)]
pub mod padding;

#[cfg(not(feature = "export_tests"))]
mod padding;

/// WGSL vec4 primitives are used on left and right hand tensor,
/// padding is avoided through the use of conditions in the kernel
pub mod unpadded;
/// WGSL vec4 primitives are used on left and right hand tensor
pub mod vec4;
