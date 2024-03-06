/// GPU dialect module that contains a representation that can be used to program any GPU.
///
/// This dialect should be used to perform most GPU-related optimizations, such as vectorization.
///
/// [Compilers](crate::codegen::Compiler) can be used to transform that representation into a lower
/// level one.
pub mod gpu;
