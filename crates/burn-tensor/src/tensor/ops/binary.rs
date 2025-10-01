use alloc::vec::Vec;

// TODO: move to `Shape`

/// Computes the output shape for binary operations with broadcasting support.
pub fn binary_ops_shape(lhs: &[usize], rhs: &[usize]) -> Vec<usize> {
    let mut shape_out = Vec::with_capacity(lhs.len());

    for (l, r) in lhs.iter().zip(rhs.iter()) {
        shape_out.push(usize::max(*l, *r));
    }

    shape_out
}
