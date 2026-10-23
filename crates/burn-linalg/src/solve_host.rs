//! LU factorization and triangular solves over owned host buffers.
//!
//! Panels use partial pivoting; matrix multiplication updates the rest of the
//! matrix. Keeping the factorization in one buffer avoids tensor dispatch and
//! allocation for every row, while retaining optimized GEMM for the cubic work.

use alloc::{vec, vec::Vec};
use burn_std::{DType, Element, TensorData};
use num_traits::Float;

const BLOCK_SIZE: usize = 32;

/// Solve matrices with broadcast batch dimensions. Inputs have equal ranks and
/// matrix right-hand sides; rank promotion and low precision casts happen in the
/// public tensor API.
pub(crate) fn solve_host_data(a: TensorData, b: TensorData) -> TensorData {
    let a_shape = a.shape().to_vec();
    let b_shape = b.shape().to_vec();
    let rank = a_shape.len();
    assert!(rank >= 2 && rank == b_shape.len());
    let n = a_shape[rank - 1];
    let columns = b_shape[rank - 1];
    assert_eq!(a_shape[rank - 2], n);
    assert_eq!(b_shape[rank - 2], n);
    assert_eq!(a.dtype(), b.dtype());

    let mut shape = b_shape.clone();
    for dim in 0..rank - 2 {
        assert!(
            a_shape[dim] == b_shape[dim] || a_shape[dim] == 1 || b_shape[dim] == 1,
            "linalg::solve: batch dimensions are not broadcast-compatible"
        );
        shape[dim] = if a_shape[dim] == 1 {
            b_shape[dim]
        } else {
            a_shape[dim]
        };
    }

    match a.dtype() {
        DType::F32 => solve_data::<f32>(a, b, &a_shape, &b_shape, shape, n, columns),
        DType::F64 => solve_data::<f64>(a, b, &a_shape, &b_shape, shape, n, columns),
        _ => panic!("linalg::solve: host computation requires F32 or F64"),
    }
}

fn solve_data<F: SolveFloat>(
    a: TensorData,
    b: TensorData,
    a_shape: &[usize],
    b_shape: &[usize],
    shape: Vec<usize>,
    n: usize,
    columns: usize,
) -> TensorData {
    let rank = shape.len();
    let batch: usize = shape[..rank - 2].iter().product();
    if n == 0 || batch == 0 {
        return TensorData::new(Vec::<F>::new(), shape);
    }

    let mut lu = a.try_into_vec::<F>().unwrap();
    let a_batch: usize = a_shape[..rank - 2].iter().product();
    let mut pivots = vec![0; a_batch * n];
    for (matrix, pivots) in lu.chunks_exact_mut(n * n).zip(pivots.chunks_exact_mut(n)) {
        factor(matrix, pivots, n);
    }

    // Singular matrices must still be rejected when there are no RHS columns.
    if columns == 0 {
        return TensorData::new(Vec::<F>::new(), shape);
    }

    let b = b.try_into_vec::<F>().unwrap();
    let matrix_size = n * columns;
    let mut output = vec![F::zero(); batch * matrix_size];
    let a_strides = batch_strides(a_shape);
    let b_strides = batch_strides(b_shape);
    for (batch_index, rhs) in output.chunks_exact_mut(matrix_size).enumerate() {
        let mut index = batch_index;
        let mut a_index = 0;
        let mut b_index = 0;
        for dim in (0..rank - 2).rev() {
            let coordinate = index % shape[dim];
            index /= shape[dim];
            a_index += coordinate * a_strides[dim];
            b_index += coordinate * b_strides[dim];
        }
        rhs.copy_from_slice(&b[b_index * matrix_size..(b_index + 1) * matrix_size]);
        substitute(
            &lu[a_index * n * n..(a_index + 1) * n * n],
            &pivots[a_index * n..(a_index + 1) * n],
            rhs,
            n,
            columns,
        );
    }
    TensorData::new(output, shape)
}

/// Strides in matrices, with zero strides for broadcast dimensions.
fn batch_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len() - 2];
    let mut stride = 1;
    for dim in (0..strides.len()).rev() {
        if shape[dim] != 1 {
            strides[dim] = stride;
        }
        stride *= shape[dim];
    }
    strides
}

/// Factor a row-major square matrix in place, storing L below the diagonal and
/// U on and above it. The unit diagonal of L is implicit.
fn factor<F: SolveFloat>(matrix: &mut [F], pivots: &mut [usize], n: usize) {
    if n <= BLOCK_SIZE {
        factor_panel(matrix, pivots, n, n);
        return;
    }
    // A compact panel stays in cache while its columns are eliminated. Using
    // the full matrix's row stride here is especially costly for large powers
    // of two, where adjacent rows map to the same cache sets.
    let mut scratch = vec![F::zero(); n * BLOCK_SIZE];
    for start in (0..n).step_by(BLOCK_SIZE) {
        let end = (start + BLOCK_SIZE).min(n);
        let width = end - start;
        let rows = n - start;
        let panel = &mut scratch[..rows * width];
        for (out, row) in panel
            .chunks_exact_mut(width)
            .zip(matrix[start * n..].chunks_exact(n))
        {
            out.copy_from_slice(&row[start..end]);
        }
        factor_panel(panel, &mut pivots[start..end], rows, width);
        for column in start..end {
            pivots[column] += start;
            let pivot = pivots[column];
            if pivot != column {
                let (top, bottom) = matrix.split_at_mut(pivot * n);
                top[column * n..(column + 1) * n].swap_with_slice(&mut bottom[..n]);
            }
        }
        for (input, row) in panel
            .chunks_exact(width)
            .zip(matrix[start * n..].chunks_exact_mut(n))
        {
            row[start..end].copy_from_slice(input);
        }

        if end == n {
            break;
        }

        // L11 U12 = A12. Each row update streams through the trailing columns.
        for row in start + 1..end {
            let (top, bottom) = matrix.split_at_mut(row * n);
            let current = &mut bottom[..n];
            for column in start..row {
                let multiplier = current[column];
                subtract_scaled(
                    &mut current[end..],
                    &top[column * n + end..(column + 1) * n],
                    multiplier,
                );
            }
        }

        // A22 -= L21 U12. The three submatrices occupy disjoint elements of the
        // same allocation, so GEMM may update A22 without copying either input.
        let ptr = matrix.as_mut_ptr();
        // SAFETY: All matrices fit within the n*n allocation. A is below/left
        // of the panel, B is above/right, and C is below/right; neither input
        // overlaps C. Positive row strides give distinct output elements.
        unsafe {
            F::gemm_subtract(
                n - end,
                end - start,
                n - end,
                (ptr.add(end * n + start), n),
                (ptr.add(start * n + end), n),
                (ptr.add(end * n + end), n),
            );
        }
    }
}

/// Unblocked factorization of one compact row-major panel.
fn factor_panel<F: SolveFloat>(panel: &mut [F], pivots: &mut [usize], rows: usize, width: usize) {
    for column in 0..width {
        let mut pivot = column;
        let mut maximum = panel[column * width + column].abs();
        for row in column + 1..rows {
            let candidate = panel[row * width + column].abs();
            if candidate > maximum {
                maximum = candidate;
                pivot = row;
            }
        }
        assert!(maximum != F::zero(), "linalg::solve: A is singular");
        pivots[column] = pivot;
        if pivot != column {
            let (top, bottom) = panel.split_at_mut(pivot * width);
            top[column * width..(column + 1) * width].swap_with_slice(&mut bottom[..width]);
        }
        let (top, bottom) = panel.split_at_mut((column + 1) * width);
        let pivot_row = &top[column * width..(column + 1) * width];
        let diagonal = pivot_row[column];
        // A reciprocal can overflow for subnormal pivots; divide directly then.
        if diagonal.abs() >= F::min_positive_value() {
            let inverse = F::one() / diagonal;
            for row in bottom.chunks_exact_mut(width) {
                row[column] = row[column] * inverse;
                let multiplier = row[column];
                subtract_scaled(&mut row[column + 1..], &pivot_row[column + 1..], multiplier);
            }
        } else {
            for row in bottom.chunks_exact_mut(width) {
                row[column] = row[column] / diagonal;
                let multiplier = row[column];
                subtract_scaled(&mut row[column + 1..], &pivot_row[column + 1..], multiplier);
            }
        }
    }
}

#[inline]
fn subtract_scaled<F: SolveFloat>(output: &mut [F], row: &[F], multiplier: F) {
    for (output, value) in output.iter_mut().zip(row) {
        *output = *output - multiplier * *value;
    }
}

fn substitute<F: SolveFloat>(lu: &[F], pivots: &[usize], rhs: &mut [F], n: usize, columns: usize) {
    for (row, &pivot) in pivots.iter().enumerate() {
        if pivot != row {
            let (before, after) = rhs.split_at_mut(pivot * columns);
            before[row * columns..(row + 1) * columns].swap_with_slice(&mut after[..columns]);
        }
    }

    // Solve diagonal blocks directly and update the remaining rows with GEMM.
    // This also benefits vector right-hand sides: the GEMM kernel can reduce
    // several LU columns together instead of dispatching a scalar row update.
    let block = BLOCK_SIZE;
    for start in (0..n).step_by(block) {
        let end = (start + block).min(n);
        for row in start..end {
            let (top, bottom) = rhs.split_at_mut(row * columns);
            let current = &mut bottom[..columns];
            for column in start..row {
                subtract_scaled(
                    current,
                    &top[column * columns..(column + 1) * columns],
                    lu[row * n + column],
                );
            }
        }
        if end < n {
            let (solved, remaining) = rhs.split_at_mut(end * columns);
            // SAFETY: L21 has (n-end)*block elements with row stride n. The
            // solved and remaining RHS rows are disjoint, correctly sized
            // slices with contiguous rows of `columns` elements.
            unsafe {
                F::gemm_subtract(
                    n - end,
                    end - start,
                    columns,
                    (lu.as_ptr().add(end * n + start), n),
                    (solved.as_ptr().add(start * columns), columns),
                    (remaining.as_mut_ptr(), columns),
                );
            }
        }
    }

    let mut end = n;
    while end > 0 {
        let start = end.saturating_sub(block);
        for row in (start..end).rev() {
            let (top, bottom) = rhs.split_at_mut((row + 1) * columns);
            let current = &mut top[row * columns..];
            for column in row + 1..end {
                let offset = (column - row - 1) * columns;
                subtract_scaled(
                    current,
                    &bottom[offset..offset + columns],
                    lu[row * n + column],
                );
            }
            let diagonal = lu[row * n + row];
            for value in current {
                *value = *value / diagonal;
            }
        }
        if start > 0 {
            let (remaining, solved) = rhs.split_at_mut(start * columns);
            // SAFETY: U12 fits in LU; the top and solved RHS rows are disjoint
            // and have positive, contiguous row strides.
            unsafe {
                F::gemm_subtract(
                    start,
                    end - start,
                    columns,
                    (lu.as_ptr().add(start), n),
                    (solved.as_ptr(), columns),
                    (remaining.as_mut_ptr(), columns),
                );
            }
        }
        end = start;
    }
}

trait SolveFloat: Float + Element {
    /// Compute C -= A B with row-major strided matrices.
    ///
    /// # Safety
    /// Matrices must have the given dimensions and valid positive row strides;
    /// C must not overlap either input or alias its own elements.
    unsafe fn gemm_subtract(
        rows: usize,
        inner: usize,
        columns: usize,
        a: (*const Self, usize),
        b: (*const Self, usize),
        c: (*mut Self, usize),
    );
}

macro_rules! impl_solve_float {
    ($float:ty) => {
        impl SolveFloat for $float {
            unsafe fn gemm_subtract(
                rows: usize,
                inner: usize,
                columns: usize,
                a: (*const Self, usize),
                b: (*const Self, usize),
                c: (*mut Self, usize),
            ) {
                // SAFETY: The caller guarantees the matrix dimensions,
                // strides and non-overlapping writable output.
                unsafe {
                    gemm::gemm(
                        rows,
                        columns,
                        inner,
                        c.0,
                        1,
                        c.1 as isize,
                        true,
                        a.0,
                        1,
                        a.1 as isize,
                        b.0,
                        1,
                        b.1 as isize,
                        1.0,
                        -1.0,
                        false,
                        false,
                        false,
                        gemm::Parallelism::None,
                    );
                }
            }
        }
    };
}

impl_solve_float!(f32);
impl_solve_float!(f64);
