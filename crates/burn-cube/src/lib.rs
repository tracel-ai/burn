extern crate alloc;

#[macro_use]
extern crate derive_new;

/// Cube Frontend Types.
pub mod frontend;

use burn_compute::server::ComputeServer;
pub use frontend::cmma;

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
pub use burn_cube_macros::CubeLaunch;
pub use burn_cube_macros::CubeType;

/// An approximation of the subcube dimension.
pub const SUBCUBE_DIM_APPROX: usize = 16;

use crate::ir::KernelDefinition;
use frontend::LaunchArg;
use prelude::CubeCount;

/// Implement this trait to create a [kernel definition](KernelDefinition).
pub trait Kernel: Send + Sync + 'static {
    /// Convert to a kernel definition.
    fn define(&self) -> KernelDefinition;
    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> String {
        format!("{:?}", core::any::TypeId::of::<Self>())
    }
}

/// Calculate the number of cubes required to execute an operation where one cube unit is
/// assigned to one element.
pub fn calculate_cube_count_elemwise<S: ComputeServer>(
    num_elems: usize,
    cube_dim: usize,
) -> CubeCount<S> {
    let num_elems_per_cube = cube_dim * cube_dim;
    let cube_counts = f32::ceil(num_elems as f32 / num_elems_per_cube as f32);
    let cube_count_x = f32::ceil(f32::sqrt(cube_counts));
    let cube_count_y = f32::ceil(num_elems as f32 / (cube_count_x * num_elems_per_cube as f32));

    CubeCount::Static(cube_count_x as u32, cube_count_y as u32, 1)
}

pub fn tensor_vectorization_factor(
    factors: &[u8],
    shape: &[usize],
    strides: &[usize],
    dim: usize,
) -> u8 {
    if let Some(val) = strides.get(dim) {
        if *val != 1 {
            return 1;
        }
    } else {
        return 1;
    }

    let dim_size = match shape.get(dim) {
        Some(val) => val,
        None => return 1,
    };

    for factor in factors {
        if dim_size % *factor as usize == 0 {
            return *factor;
        }
    }

    1
}

/// Runtime arguments to launch a kernel.
pub type RuntimeArg<'a, T, R> = <T as LaunchArg>::RuntimeArg<'a, R>;

#[cfg(feature = "export_tests")]
/// Tests only useful for runtimes.
pub mod runtime_tests;
