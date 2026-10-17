//! Direct Flex CPU factorization. The public function handles shapes, dtype
//! promotion, and routing; gradient-enabled tensors use the tensor algorithm.
use burn_core as burn;
use burn_core::backend::{Backend, Flex, backend_extension, tensor::FloatTensor};
use burn_std::DType;
use num_traits::Float;

#[backend_extension(Flex: cfg(feature = "flex"))]
pub(crate) trait CholeskyCpuOps: Backend {
    fn cholesky_cpu(tensor: FloatTensor<Self>, upper: bool) -> FloatTensor<Self>;
}

impl CholeskyCpuOps for Flex {
    fn cholesky_cpu(tensor: FloatTensor<Self>, upper: bool) -> FloatTensor<Self> {
        let shape = tensor.layout().shape();
        let n = shape[shape.num_dims() - 1];
        let len = shape.num_elements();
        // Normalize views once; operate directly on CPU storage afterwards.
        // storage_mut performs copy-on-write when the caller retains an alias.
        let mut tensor = if tensor.is_contiguous() && tensor.layout().start_offset() == 0 {
            tensor
        } else {
            tensor.to_contiguous()
        };
        match tensor.dtype() {
            DType::F32 => factor_batch(&mut tensor.storage_mut::<f32>()[..len], n, upper),
            DType::F64 => factor_batch(&mut tensor.storage_mut::<f64>()[..len], n, upper),
            _ => unreachable!("Cholesky CPU inputs must be promoted to F32 or F64"),
        }
        tensor
    }
}

fn factor_batch<T: Float + 'static>(values: &mut [T], n: usize, upper: bool) {
    // The public API returns empty inputs before entering this kernel.
    assert!(n > 0);
    let matrix_len = n.checked_mul(n).expect("Cholesky matrix size overflow");
    assert_eq!(values.len() % matrix_len, 0);
    assert!(isize::try_from(matrix_len).is_ok());
    for (batch, a) in values.chunks_exact_mut(matrix_len).enumerate() {
        if upper {
            // Copy only the authoritative upper entries. Finish the copy
            // before clearing the upper triangle, which still holds inputs.
            for i in 0..n {
                for j in 0..i {
                    a[i * n + j] = a[j * n + i];
                }
            }
        }
        // Do not use the unselected input triangle, even if it contains NaNs.
        for i in 0..n {
            a[i * n + i + 1..(i + 1) * n].fill(T::zero());
        }
        if let Err(pivot) = factor(a, n) {
            panic!(
                "linalg::cholesky: input is not positive definite or has non-finite values (pivot {}, batch {})",
                pivot + 1,
                batch
            );
        }
        // GEMM also updates the unused portion of each diagonal block.
        for i in 0..n {
            a[i * n + i + 1..(i + 1) * n].fill(T::zero());
        }
        if upper {
            transpose(a, n);
        }
    }
}

fn transpose<T>(a: &mut [T], n: usize) {
    for i in 0..n {
        for j in 0..i {
            a.swap(i * n + j, j * n + i);
        }
    }
}

// Independent accumulators avoid a long dependency chain for contiguous dot
// products without relying on compiler reassociation of floating-point sums.
#[inline]
fn dot<T: Float>(a: &[T], b: &[T]) -> T {
    debug_assert_eq!(a.len(), b.len());
    let mut sums = [T::zero(); 4];
    let (aa, a_tail) = a.as_chunks::<4>();
    let (bb, b_tail) = b.as_chunks::<4>();
    for (a, b) in aa.iter().zip(bb) {
        for k in 0..4 {
            sums[k] = sums[k] + a[k] * b[k];
        }
    }
    let mut sum = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    for (&a, &b) in a_tail.iter().zip(b_tail) {
        sum = sum + a * b;
    }
    sum
}

fn factor<T: Float + 'static>(a: &mut [T], n: usize) -> Result<(), usize> {
    const BLOCK: usize = 16;
    for start in (0..n).step_by(BLOCK) {
        let end = (start + BLOCK).min(n);
        if start > 0 {
            // A[start.., start..end] -= L[start.., ..start]
            //                            @ L[start..end, ..start]^T.
            let ptr = a.as_mut_ptr();
            // SAFETY: a contains n*n elements and n*n fits in isize. The
            // destination is rows start..n, columns start..end; both sources
            // are in columns 0..start, so neither overlaps the destination.
            // The sources may overlap each other, but are only read. All
            // row/column strides address elements within a, and the call is
            // synchronous, so pointers cannot outlive the exclusive borrow.
            unsafe {
                gemm::gemm(
                    n - start,
                    end - start,
                    start,
                    ptr.add(start * n + start),
                    1,
                    n as isize,
                    true,
                    ptr.add(start * n),
                    1,
                    n as isize,
                    ptr.add(start * n),
                    n as isize,
                    1,
                    T::one(),
                    -T::one(),
                    false,
                    false,
                    false,
                    gemm::Parallelism::None,
                );
            }
        }
        // Factor the diagonal block and solve its panel. The work outside
        // this narrow panel is handled by the matrix-matrix update above.
        for j in start..end {
            let row = &a[j * n + start..j * n + j];
            let pivot = a[j * n + j] - dot(row, row);
            if !pivot.is_finite() || pivot <= T::zero() {
                return Err(j);
            }
            let diagonal = pivot.sqrt();
            a[j * n + j] = diagonal;
            for i in j + 1..n {
                let correction = dot(&a[i * n + start..i * n + j], &a[j * n + start..j * n + j]);
                a[i * n + j] = (a[i * n + j] - correction) / diagonal;
            }
        }
    }
    Ok(())
}
