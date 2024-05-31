extern crate alloc;

#[macro_use]
extern crate derive_new;

/// Cube Frontend Types.
pub mod frontend;

/// Cube Language Internal Representation.
pub mod ir;

pub mod codegen;
pub mod compute;
pub mod prelude;

mod pod;
mod runtime;

pub use codegen::*;
pub use pod::*;
pub use runtime::*;

pub use burn_cube_macros::cube;

/// An approximation of the subcube dimension.
pub const SUBCUBE_DIM_APPROX: usize = 16;

use crate::ir::KernelDefinition;
use frontend::LaunchArg;
use prelude::CubeCount;

/// Dynamic jit kernel to create a [compute shader](ComputeShader).
pub trait Kernel: Send + Sync + 'static {
    /// Convert to compute shader
    fn define(&self) -> KernelDefinition;
    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> String {
        format!("{:?}", core::any::TypeId::of::<Self>())
    }
}

pub fn calculate_cube_count_elemwise(num_elems: usize, cube_dim: usize) -> CubeCount {
    let num_elems_per_cube = cube_dim * cube_dim;
    let cube_counts = f32::ceil(num_elems as f32 / num_elems_per_cube as f32);
    let cube_count_x = f32::ceil(f32::sqrt(cube_counts));
    let cube_count_y = f32::ceil(num_elems as f32 / (cube_count_x * num_elems_per_cube as f32));

    CubeCount::new(cube_count_x as u32, cube_count_y as u32, 1)
}

pub type RuntimeArg<'a, T, R> = <T as LaunchArg>::RuntimeArg<'a, R>;

#[cfg(feature = "export_tests")]
pub mod runtime_tests;
