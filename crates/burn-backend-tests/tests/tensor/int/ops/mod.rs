pub use super::*; // re-export test types

mod abs;
mod add;
mod aggregation;
mod all;
mod any;
mod arange;
mod arange_step;
mod arg;
mod bitwise;
mod cartesian_grid;
mod cast;
mod cat;
mod chunk;
mod clamp;
mod comparison;
mod create_like;
mod cumulative;
mod div;
mod einsum;
mod expand;
mod flip;
mod full;
mod gather_scatter;
mod gather_scatter_nd;
mod init;
mod mask;
mod mask_select;
mod matmul;
mod maxmin;
mod movedim;
mod mul;
mod one_hot;
// WGPU and remote can select a device without 64-bit integer arithmetic in mixed builds.
#[cfg(all(
    any(
        feature = "flex",
        feature = "cpu",
        feature = "cuda",
        feature = "rocm",
        feature = "metal"
    ),
    not(any(feature = "wgpu", feature = "remote")),
))]
mod overflow_64bit;
mod permute;
mod powi;
mod random;
mod remainder;
mod repeat;
mod repeat_dim;
mod reshape;
mod roll;
mod select;
mod sign;
mod slice;
mod slice_assign;
mod sort_argsort;
mod square;
mod stack;
mod sub;
mod take;
mod topk;
mod transpose;
mod tri;
mod unfold;
