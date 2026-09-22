use alloc::vec;
use alloc::vec::Vec;

use crate::{Shape, Tiling};

/// The permutation that gathers each logical dim's storage fragments together, coarsest first, and
/// the logical shape they multiply back to: what lays a level-major buffer back into rows, and the
/// shape a storage-tiled tensor stands for.
///
/// # Panics
///
/// When `tiling` does not describe a tensor of `physical`'s rank.
pub fn tiled_fragments(physical: &Shape, tiling: Tiling) -> (Vec<usize>, Shape) {
    let rank = tiling
        .logical_rank(physical.num_dims())
        .unwrap_or_else(|err| panic!("into_tiled: {err:?}"));
    let fragments = tiling.fragments(rank);
    let levels = fragments.iter().copied().max().unwrap_or(1);
    let mut groups = vec![Vec::new(); rank];
    let mut dim = 0;
    for level in 0..levels {
        for (logical, &count) in fragments.iter().enumerate() {
            if level < count {
                groups[logical].push(dim);
                dim += 1;
            }
        }
    }
    let logical = groups
        .iter()
        .map(|group| group.iter().map(|&dim| physical[dim]).product())
        .collect::<Vec<usize>>();
    (groups.concat(), Shape::from(logical))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `[k, n]` in `(tr, tc)` tiles is `[k / tr, n / tc, tr, tc]`: the row fragments are dims 0 and
    /// 2, the column fragments 1 and 3, and they multiply back to `[k, n]`.
    #[test]
    fn a_tiled_matrix_gathers_its_fragments_back() {
        let tiling = Tiling::new(&[2, 2]).unwrap();
        let (axes, logical) = tiled_fragments(&Shape::from(vec![4, 2, 8, 16]), tiling);
        assert_eq!(axes, vec![0, 2, 1, 3]);
        assert_eq!(logical, Shape::from(vec![32, 32]));
    }

    /// A batch dim stored as one fragment stays in front.
    #[test]
    fn a_batch_dim_passes_through() {
        let tiling = Tiling::new(&[1, 2, 2]).unwrap();
        let (axes, logical) = tiled_fragments(&Shape::from(vec![3, 4, 2, 8, 16]), tiling);
        assert_eq!(axes, vec![0, 1, 3, 2, 4]);
        assert_eq!(logical, Shape::from(vec![3, 32, 32]));
    }
}
