use super::dialect::gpu;
use crate::{FloatElement, IntElement};
use std::fmt::Display;

/// Compiles the [gpu representation](gpu::ComputeShader) into its own representation that can be
/// formatted into tokens.
pub trait Compiler: Sync + Send + 'static + Clone + Default + core::fmt::Debug {
    /// The representation for the compiled code.
    type Representation: Display;
    /// The float element type used for compilation.
    type Float: FloatElement;
    /// The int element type used for compilation.
    type Int: IntElement;
    /// The compiler that can be used to generate full precision shaders.
    type FullPrecisionCompiler: Compiler<
        Representation = Self::Representation,
        Float = f32,
        Int = i32,
    >;

    /// Compiles the [gpu shader](gpu::ComputeShader) into the compiler's representation.
    fn compile(shader: gpu::ComputeShader) -> Self::Representation;
    /// The size of the given element in bytes.
    fn elem_size(elem: gpu::Elem) -> usize;
    /// The maximal size of a shared memory
    fn max_shared_memory_size() -> usize;
}
