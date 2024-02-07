/// GPU dialect module that contains a representation that can be used to program any GPU.
///
/// This dialect should be used to perform most GPU-related optimizations, such as vectorization.
///
/// [Compilers](crate::codegen::Compiler) can be used to transform that representation into a lower
/// level one, such as [wgsl](crate::codegen::dialect::wgsl).
pub(crate) mod gpu;
/// WGSL dialect module that contains a representation that can be compiled to WebGPU shading
/// language (wgsl).
pub(crate) mod wgsl;
