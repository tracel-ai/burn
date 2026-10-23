use super::*;
use burn_core::tensor::quantization::QuantScheme;
use burn_core::tensor::{DType, Device, TensorData, Tolerance};
use burn_linalg::cholesky;

// Construct A = L L^T on the host, independently of Burn's matmul and
// factorization. Dyadic entries keep the F32 inputs exact and well conditioned.
fn known_factors(n: usize, batch: usize) -> (Vec<f32>, Vec<f32>) {
    let mut factors = vec![0.0; batch * n * n];
    let mut matrices = vec![0.0; batch * n * n];
    for b in 0..batch {
        let base = b * n * n;
        for i in 0..n {
            factors[base + i * n + i] = 2.0 + b as f32 * 0.25 + (i % 5) as f32 * 0.125;
            for j in 0..i {
                factors[base + i * n + j] = ((i * 7 + j * 11 + b * 3) % 9) as f32 / 32.0 - 0.125;
            }
        }
        for i in 0..n {
            for j in 0..n {
                matrices[base + i * n + j] = (0..=i.min(j))
                    .map(|k| {
                        f64::from(factors[base + i * n + k]) * f64::from(factors[base + j * n + k])
                    })
                    .sum::<f64>() as f32;
            }
        }
    }
    (matrices, factors)
}

fn assert_decomposition<const D: usize>(input: TestTensor<D>, lower: Vec<f32>, upper: bool) {
    let dims = input.dims();
    let n = dims[D - 1];
    let dtype = input.dtype();
    let device = input.device();
    let factor = cholesky(input, upper);
    assert_eq!(factor.dims(), dims);
    assert_eq!(factor.dtype(), dtype);
    assert_eq!(factor.device(), device);
    assert_eq!(factor.device().is_autodiff(), device.is_autodiff());

    let mut expected = lower;
    if upper {
        for matrix in expected.chunks_exact_mut(n * n) {
            for i in 0..n {
                for j in 0..i {
                    matrix.swap(i * n + j, j * n + i);
                }
            }
        }
    }
    let data = factor.into_data();
    data.assert_approx_eq::<f32>(
        &TensorData::new(expected, dims),
        Tolerance::rel_abs(1e-4, 1e-5),
    );

    // The known factor is an independent numerical reference. Check exact
    // zeros separately because the approximate comparison allows small errors.
    let values: Vec<_> = data.iter::<f64>().collect();
    for f in values.chunks_exact(n * n) {
        for i in 0..n {
            assert!(f[i * n + i].is_finite() && f[i * n + i] > 0.0);
            for j in 0..n {
                if (upper && i > j) || (!upper && i < j) {
                    assert_eq!(f[i * n + j], 0.0, "unused triangle at ({i}, {j})");
                }
            }
        }
    }
}

fn assert_rejected<const D: usize>(input: TestTensor<D>, upper: bool) {
    assert_panics_with(
        input,
        upper,
        "linalg::cholesky: input is not positive definite or has non-finite values",
    );
}

fn assert_panics_with<const D: usize>(input: TestTensor<D>, upper: bool, expected: &str) {
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cholesky(input, upper).into_data();
    }))
    .expect_err("invalid input must be rejected");
    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .expect("expected a Cholesky diagnostic");
    assert!(message.contains(expected), "unexpected panic: {message}");
}

#[test]
fn test_cholesky_known_lower_and_upper() {
    let device = Default::default();
    // The classic example has L = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]].
    for upper in [false, true] {
        let data = TensorData::from([
            [4.0f32, 12.0, -16.0],
            [12.0, 37.0, -43.0],
            [-16.0, -43.0, 98.0],
        ]);
        let input = TestTensor::<2>::from_data(data.clone(), &device);
        assert_decomposition(
            input.clone(),
            vec![2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0],
            upper,
        );
        // Factorization must respect copy-on-write when an alias survives.
        input.into_data().assert_eq(&data, true);
    }
}

#[test]
fn test_cholesky_singleton() {
    let device = Default::default();
    assert_decomposition(
        TestTensor::<2>::from_data([[9.0]], &device),
        vec![3.0],
        false,
    );
}

#[test]
fn test_cholesky_block_boundaries_and_trailing_updates() {
    let device = Default::default();
    // CPU panels are 16 wide and GPU panels 32 wide. Include exact boundaries,
    // partial panels, and four GPU panels to exercise the workspace updates.
    // The view tests cover sizes 33 and 65.
    for n in [15, 16, 17, 31, 32, 64, 97] {
        let (matrices, lower) = known_factors(n, 1);
        for upper in [false, true] {
            let input =
                TestTensor::<2>::from_data(TensorData::new(matrices.clone(), [n, n]), &device);
            assert_decomposition(input, lower.clone(), upper);
        }
    }
}

#[test]
fn test_cholesky_ignores_unselected_triangle_in_transposed_view() {
    let device = Default::default();
    let n = 65;
    let (matrices, lower) = known_factors(n, 1);
    for upper in [false, true] {
        let mut poisoned = matrices.clone();
        for i in 0..n {
            for j in 0..n {
                if (upper && i > j) || (!upper && i < j) {
                    poisoned[i * n + j] = if (i + j) % 2 == 0 {
                        f32::NAN
                    } else {
                        f32::INFINITY
                    };
                }
            }
        }
        // Transposing swaps the authoritative triangle. Poisoning the other
        // triangle also makes ignoring the view's strides observable.
        let input =
            TestTensor::<2>::from_data(TensorData::new(poisoned, [n, n]), &device).swap_dims(0, 1);
        assert_decomposition(input, lower.clone(), !upper);
    }
}

#[test]
fn test_cholesky_sliced_input_with_offset() {
    let device = Default::default();
    let n = 33;
    let (matrices, lower) = known_factors(n, 1);
    let mut padded = vec![f32::NAN; (n + 2) * (n + 2)];
    for i in 0..n {
        padded[(i + 1) * (n + 2) + 1..(i + 1) * (n + 2) + 1 + n]
            .copy_from_slice(&matrices[i * n..(i + 1) * n]);
    }
    for upper in [false, true] {
        let input =
            TestTensor::<2>::from_data(TensorData::new(padded.clone(), [n + 2, n + 2]), &device)
                .slice_dim(0, 1..n + 1)
                .slice_dim(1, 1..n + 1);
        assert_decomposition(input, lower.clone(), upper);
    }
}

#[test]
fn test_cholesky_permuted_batch_dimensions() {
    let device = Default::default();
    let n = 65;
    let (matrices, lower) = known_factors(n, 6);
    let mut reordered = Vec::with_capacity(lower.len());
    for j in 0..3 {
        for i in 0..2 {
            let base = (i * 3 + j) * n * n;
            reordered.extend_from_slice(&lower[base..base + n * n]);
        }
    }
    for upper in [false, true] {
        let input =
            TestTensor::<4>::from_data(TensorData::new(matrices.clone(), [2, 3, n, n]), &device)
                .swap_dims(0, 1);
        assert_decomposition(input, reordered.clone(), upper);
    }
}

#[test]
fn test_cholesky_half_precision_computes_in_f32() {
    let device = Device::default();
    for dtype in [DType::F16, DType::BF16] {
        if !device.supports_dtype(dtype) {
            continue;
        }
        // L = [[sqrt(3), 0], [sqrt(3), 1]]. Rounding sqrt(3) and the
        // panel update in the input dtype changes the final diagonal.
        let root = if dtype == DType::F16 {
            1.732_421_9
        } else {
            1.734_375
        };
        for upper in [false, true] {
            let input = TestTensor::<2>::from_data([[3.0, 3.0], [3.0, 4.0]], &device).cast(dtype);
            let factor = cholesky(input, upper);
            assert_eq!(factor.dtype(), dtype);
            assert_eq!(factor.device(), device);
            let expected = if upper {
                vec![root, root, 0.0, 1.0]
            } else {
                vec![root, 0.0, root, 1.0]
            };
            assert_eq!(
                factor.into_data().iter::<f32>().collect::<Vec<_>>(),
                expected
            );
        }
    }
}

#[test]
fn test_cholesky_f64_preserves_precision() {
    let device = Device::default();
    if !device.supports_dtype(DType::F64) {
        return;
    }
    let diagonal = 1.0_f64 + 2.0_f64.powi(-30);
    let input = TestTensor::<2>::from_data(
        TensorData::from([[diagonal * diagonal]]),
        (&device, DType::F64),
    );
    let result = cholesky(input, false);
    assert_eq!(result.dtype(), DType::F64);
    assert_eq!(result.into_data().iter::<f64>().next().unwrap(), diagonal);
}

#[test]
fn test_cholesky_f64_multiple_panels() {
    let device = Device::default();
    if !device.supports_dtype(DType::F64) {
        return;
    }
    let n = 33;
    let (matrices, lower) = known_factors(n, 2);
    for upper in [false, true] {
        let input =
            TestTensor::<3>::from_data(TensorData::new(matrices.clone(), [2, n, n]), &device)
                .cast(DType::F64);
        assert_decomposition(input, lower.clone(), upper);
    }
}

#[test]
fn test_cholesky_accepts_small_positive_pivots() {
    let device = Default::default();
    // Exactly representable powers of two: an epsilon cutoff or F32
    // regularization would incorrectly reject or change this factor.
    let pivot = 2.0_f32.powi(-40);
    let root = 2.0_f32.powi(-20);
    let input = TestTensor::<2>::from_data([[pivot, 0.0], [0.0, 1.0]], &device);
    cholesky(input, false)
        .into_data()
        .assert_eq(&TensorData::from([[root, 0.0], [0.0, 1.0]]), true);
}

#[cfg(feature = "flex")]
#[test]
fn test_cholesky_empty_matrices_and_batches_on_flex() {
    // CUDA's allocator rejects zero-byte allocations before Cholesky runs.
    // Check the empty-input contract on a backend that supports empty tensors.
    let device = Device::flex();
    let output = cholesky(TestTensor::<2>::empty([0, 0], &device), false);
    assert_eq!(output.dims(), [0, 0]);
    assert_eq!(output.device(), device);
    assert_eq!(output.dtype(), DType::F32);
    let output = cholesky(TestTensor::<4>::empty([2, 0, 3, 3], &device), true);
    assert_eq!(output.dims(), [2, 0, 3, 3]);
    assert_eq!(output.into_data().iter::<f32>().count(), 0);
}

#[test]
#[should_panic(expected = "linalg::cholesky: input must have at least two dimensions")]
fn test_cholesky_rejects_rank_one() {
    let device = Default::default();
    cholesky(TestTensor::<1>::from_data([1.0, 2.0], &device), false);
}

#[test]
#[should_panic(expected = "linalg::cholesky: input must be square")]
fn test_cholesky_rejects_nonsquare_input() {
    let device = Default::default();
    cholesky(TestTensor::<2>::ones([2, 3], &device), false);
}

#[test]
fn test_cholesky_rejects_quantized_input() {
    let device = Device::default();
    let scheme = QuantScheme::default();
    if !device.supports_dtype(DType::QFloat(scheme)) {
        return;
    }
    // Packed GPU storage requires the last dimension to hold whole words.
    let input = TestTensor::<2>::eye(scheme.num_quants(), &device).quantize_dynamic(&scheme);
    assert!(matches!(input.dtype(), DType::QFloat(_)));
    assert_panics_with(
        input,
        false,
        "linalg::cholesky: input must have a real floating point dtype",
    );
}

#[test]
fn test_cholesky_rejects_singular_and_indefinite_inputs() {
    let device = Default::default();
    for upper in [false, true] {
        // Positive input diagonals must not hide a zero or negative Schur pivot.
        for matrix in [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 2.0], [2.0, 1.0]]] {
            assert_rejected(TestTensor::<2>::from_data(matrix, &device), upper);
        }
    }
}

#[test]
fn test_cholesky_rejects_nonfinite_selected_values() {
    let device = Default::default();
    for upper in [false, true] {
        for value in [f32::NAN, f32::INFINITY] {
            for matrix in [[[value, 0.0], [0.0, 1.0]], [[1.0, value], [value, 1.0]]] {
                assert_rejected(TestTensor::<2>::from_data(matrix, &device), upper);
            }
        }
    }
}

#[test]
fn test_cholesky_rejects_invalid_later_batch_and_panel() {
    let device = Default::default();
    let n = 65;
    for upper in [false, true] {
        // Raw first-pivot status is covered by the backend test. Here, verify
        // that the public API rejects a failure in a later batch and panel.
        let mut matrices = vec![0.0; 2 * n * n];
        for b in 0..2 {
            for i in 0..n {
                matrices[b * n * n + i * n + i] = 1.0;
            }
        }
        matrices[n * n + 64 * n + 64] = 0.0;
        let input = TestTensor::<3>::from_data(TensorData::new(matrices, [2, n, n]), &device);
        assert_panics_with(input, upper, "pivot 65");
    }
}

#[cfg(feature = "autodiff")]
#[test]
fn test_cholesky_autodiff_batched_forward_and_selected_triangle_gradient() {
    let device = Device::default().autodiff();
    for upper in [false, true] {
        let input = TestTensor::<3>::from_data(
            [[[4.0, 2.0], [2.0, 5.0]], [[16.0, 8.0], [8.0, 20.0]]],
            &device,
        )
        .require_grad();
        let factor = cholesky(input.clone(), upper);
        let expected_factor = if upper {
            [[[2.0, 1.0], [0.0, 2.0]], [[4.0, 2.0], [0.0, 4.0]]]
        } else {
            [[[2.0, 0.0], [1.0, 2.0]], [[4.0, 0.0], [2.0, 4.0]]]
        };
        factor.clone().into_data().assert_approx_eq::<f32>(
            &TensorData::from(expected_factor),
            Tolerance::rel_abs(1e-4, 1e-5),
        );
        let grads = factor.sum().backward();
        // For lower: sum(L) = sqrt(a) + b/sqrt(a) + sqrt(c - b^2/a).
        // At (a,b,c) = (4,2,5), its derivatives are (3/16,1/4,1/4).
        // Scaling A by four doubles L and halves these derivatives.
        // Upper uses the transposed authoritative triangle.
        let expected = if upper {
            [
                [[0.1875, 0.25], [0.0, 0.25]],
                [[0.09375, 0.125], [0.0, 0.125]],
            ]
        } else {
            [
                [[0.1875, 0.0], [0.25, 0.25]],
                [[0.09375, 0.0], [0.125, 0.125]],
            ]
        };
        input
            .grad(&grads)
            .expect("input gradient")
            .into_data()
            .assert_approx_eq::<f32>(&TensorData::from(expected), Tolerance::rel_abs(1e-4, 1e-5));
    }
}
