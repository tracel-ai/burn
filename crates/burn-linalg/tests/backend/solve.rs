use super::*;
use burn_core::tensor::{DType, TensorData, Tolerance};
use burn_linalg::solve;

#[test]
fn solve_vector_with_pivot() {
    let device = Default::default();
    let a = TestTensor::<2>::from_data([[0.0, 2.0], [1.0, 3.0]], &device);
    let b = TestTensor::<1>::from_data([4.0, 7.0], &device);
    let x = solve::<2, 1, 1>(a, b);
    let expected = TestTensor::<1>::from_data([1.0, 2.0], &device);
    x.into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), Tolerance::default());
}

#[test]
fn solve_multiple_right_hand_sides() {
    let device = Default::default();
    let a = TestTensor::<2>::from_data([[3.0, 1.0], [1.0, 2.0]], &device);
    let b = TestTensor::<2>::from_data([[9.0, 1.0], [8.0, 0.0]], &device);
    let x = solve::<2, 2, 2>(a.clone(), b.clone());
    let expected = TestTensor::<2>::from_data([[2.0, 0.4], [3.0, -0.2]], &device);
    x.clone()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), Tolerance::default());
    a.matmul(x)
        .into_data()
        .assert_approx_eq::<FloatElem>(&b.into_data(), Tolerance::default());
}

#[test]
fn solve_batched_vectors_and_broadcast_vector() {
    let device = Default::default();
    let a = TestTensor::<3>::from_data(
        [[[3.0, 1.0], [1.0, 2.0]], [[0.0, 2.0], [1.0, 3.0]]],
        &device,
    );
    let b = TestTensor::<2>::from_data([[9.0, 8.0], [4.0, 7.0]], &device);
    let x = solve::<3, 2, 2>(a.clone(), b);
    let expected = TestTensor::<2>::from_data([[2.0, 3.0], [1.0, 2.0]], &device);
    x.into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), Tolerance::default());

    let b = TestTensor::<1>::from_data([9.0, 8.0], &device);
    let x = solve::<3, 1, 2>(a, b);
    let expected = TestTensor::<2>::from_data([[2.0, 3.0], [-5.5, 4.5]], &device);
    x.into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), Tolerance::default());
}

#[test]
fn solve_broadcasts_matrix_rhs_and_singleton_batch() {
    let device = Default::default();
    let a = TestTensor::<3>::from_data([[[3.0, 1.0], [1.0, 2.0]]], &device);
    let b = TestTensor::<3>::from_data([[[9.0], [8.0]], [[1.0], [0.0]]], &device);
    let x = solve::<3, 3, 3>(a, b);
    let expected = TestTensor::<3>::from_data([[[2.0], [3.0]], [[0.4], [-0.2]]], &device);
    x.into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), Tolerance::default());

    let a = TestTensor::<3>::from_data(
        [[[3.0, 1.0], [1.0, 2.0]], [[0.0, 2.0], [1.0, 3.0]]],
        &device,
    );
    let b = TestTensor::<2>::from_data([[9.0, 1.0], [8.0, 0.0]], &device);
    let x = solve::<3, 2, 3>(a.clone(), b.clone());
    let b = b.reshape([1, 2, 2]).expand([2, 2, 2]);
    a.matmul(x)
        .into_data()
        .assert_approx_eq::<FloatElem>(&b.into_data(), Tolerance::default());
}

#[test]
fn solve_accepts_nonzero_small_pivot() {
    let device = Default::default();
    let a = TestTensor::<2>::from_data(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1e-8],
        ],
        &device,
    );
    let b = TestTensor::<1>::from_data([1.0, 1.0, 1.0, 1e-8], &device);
    let x = solve::<2, 1, 1>(a, b);
    let expected = TestTensor::<1>::from_data([1.0, 1.0, 1.0, 1.0], &device);
    x.into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), Tolerance::default());
}

#[test]
#[should_panic(expected = "A is singular")]
fn solve_rejects_singular_matrix() {
    let device = Default::default();
    let a = TestTensor::<2>::from_data([[1.0, 2.0], [2.0, 4.0]], &device);
    let b = TestTensor::<1>::from_data([3.0, 6.0], &device);
    let _ = solve::<2, 1, 1>(a, b);
}

#[test]
#[should_panic(expected = "A is singular")]
fn solve_rejects_batch_with_singular_matrix() {
    let device = Default::default();
    let a = TestTensor::<3>::from_data(
        [[[3.0, 1.0], [1.0, 2.0]], [[1.0, 2.0], [2.0, 4.0]]],
        &device,
    );
    let b = TestTensor::<2>::from_data([[9.0, 8.0], [3.0, 6.0]], &device);
    let _ = solve::<3, 2, 2>(a, b);
}

#[test]
#[should_panic(expected = "A must be square")]
fn solve_rejects_rectangular_matrix() {
    let device = Default::default();
    let a = TestTensor::<2>::from_data([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], &device);
    let b = TestTensor::<1>::from_data([1.0, 2.0, 2.0], &device);
    let _ = solve::<2, 1, 1>(a, b);
}

#[test]
#[should_panic(expected = "B must have shape")]
fn solve_rejects_wrong_rhs_shape() {
    let device = Default::default();
    let a = TestTensor::<2>::from_data([[3.0, 1.0], [1.0, 2.0]], &device);
    let b = TestTensor::<1>::from_data([1.0, 2.0, 3.0], &device);
    let _ = solve::<2, 1, 1>(a, b);
}

#[test]
fn solve_three_by_three_with_two_row_swaps() {
    let device = Default::default();
    let a =
        TestTensor::<2>::from_data([[0.0, 2.0, 1.0], [1.0, 1.0, 0.0], [2.0, 0.0, 1.0]], &device);
    let b = TestTensor::<1>::from_data([7.0, 3.0, 5.0], &device);
    let x = solve::<2, 1, 1>(a, b);
    let expected = TestTensor::<1>::from_data([1.0, 2.0, 3.0], &device);
    x.into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), Tolerance::default());
}

#[test]
#[should_panic(expected = "batch dimensions are not broadcast-compatible")]
fn solve_rejects_incompatible_batches() {
    let device = Default::default();
    let a = TestTensor::<2>::eye(2, &device)
        .reshape([1, 2, 2])
        .expand([2, 2, 2]);
    let b = TestTensor::<2>::ones([3, 2], &device);
    let _ = solve::<3, 2, 2>(a, b);
}

#[cfg(feature = "ndarray")]
#[test]
#[should_panic(expected = "dtypes must match")]
fn solve_rejects_different_dtypes() {
    #[allow(deprecated)]
    let device = burn_core::tensor::Device::ndarray();
    let a = TestTensor::<2>::eye(2, &device);
    let b = TestTensor::<1>::ones([2], &device).cast(DType::F64);
    let _ = solve::<2, 1, 1>(a, b);
}

#[cfg(feature = "autodiff")]
#[test]
fn solve_gradients_match_inverse_rule() {
    let device = burn_core::tensor::Device::default().autodiff();
    let a = TestTensor::<2>::from_data([[3.0, 1.0], [1.0, 2.0]], &device).require_grad();
    let b = TestTensor::<1>::from_data([9.0, 8.0], &device).require_grad();
    let x = solve::<2, 1, 1>(a.clone(), b.clone());
    let grads = x.sum().backward();
    let expected_a = TestTensor::<2>::from_data([[-0.4, -0.6], [-0.8, -1.2]], &device);
    let expected_b = TestTensor::<1>::from_data([0.2, 0.4], &device);
    a.grad(&grads)
        .unwrap()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_a.into_data(), Tolerance::default());
    b.grad(&grads)
        .unwrap()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_b.into_data(), Tolerance::default());
}

#[test]
fn solve_f64_preserves_precision_and_dtype() {
    let device = burn_core::tensor::Device::default();
    if !device.supports_dtype(DType::F64) {
        return;
    }
    let a = TestTensor::<2>::from_data([[3.0, 1.0], [1.0, 2.0]], &device).cast(DType::F64);
    let b = TestTensor::<1>::from_data([9.0, 8.0], &device).cast(DType::F64);
    let x = solve::<2, 1, 1>(a, b);
    assert_eq!(x.dtype(), DType::F64);
    let values = x.into_data().try_to_vec::<f64>().unwrap();
    assert!((values[0] - 2.0).abs() < 1e-12);
    assert!((values[1] - 3.0).abs() < 1e-12);
}

#[cfg(feature = "autodiff")]
#[test]
fn solve_gradients_with_row_pivot() {
    let device = burn_core::tensor::Device::default().autodiff();
    let a = TestTensor::<2>::from_data([[0.0, 2.0], [1.0, 3.0]], &device).require_grad();
    let b = TestTensor::<1>::from_data([4.0, 7.0], &device).require_grad();
    let x = solve::<2, 1, 1>(a.clone(), b.clone());
    let grads = x.sum().backward();
    let expected_a = TestTensor::<2>::from_data([[1.0, 2.0], [-1.0, -2.0]], &device);
    let expected_b = TestTensor::<1>::from_data([-1.0, 1.0], &device);
    a.grad(&grads)
        .unwrap()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_a.into_data(), Tolerance::default());
    b.grad(&grads)
        .unwrap()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_b.into_data(), Tolerance::default());
}

#[test]
fn solve_matrix_rhs_with_more_batch_dimensions_than_a() {
    let device = Default::default();
    let a = TestTensor::<2>::from_data([[3.0, 1.0], [1.0, 2.0]], &device);
    let b = TestTensor::<4>::from_data(
        [
            [[[9.0], [8.0]], [[1.0], [0.0]]],
            [[[3.0], [1.0]], [[4.0], [7.0]]],
        ],
        &device,
    );
    let x = solve::<2, 4, 4>(a.clone(), b.clone());
    assert_eq!(x.dims(), [2, 2, 2, 1]);
    a.reshape([1, 1, 2, 2])
        .matmul(x)
        .into_data()
        .assert_approx_eq::<FloatElem>(&b.into_data(), Tolerance::default());
}

#[test]
fn solve_empty_matrix_and_rhs() {
    let device = Default::default();
    let a = TestTensor::<2>::empty([0, 0], &device);
    let b = TestTensor::<1>::empty([0], &device);
    assert_eq!(solve::<2, 1, 1>(a, b).dims(), [0]);

    let a = TestTensor::<2>::eye(2, &device);
    let b = TestTensor::<2>::empty([2, 0], &device);
    assert_eq!(solve::<2, 2, 2>(a, b).dims(), [2, 0]);
}

#[test]
#[should_panic(expected = "A is singular")]
fn solve_rejects_singular_matrix_with_zero_rhs_columns() {
    let device = Default::default();
    let a = TestTensor::<2>::from_data([[1.0, 2.0], [2.0, 4.0]], &device);
    let b = TestTensor::<2>::empty([2, 0], &device);
    let _ = solve::<2, 2, 2>(a, b);
}

#[test]
fn solve_one_by_one() {
    let device = Default::default();
    let a = TestTensor::<2>::from_data([[4.0]], &device);
    let b = TestTensor::<1>::from_data([8.0], &device);
    let x = solve::<2, 1, 1>(a, b);
    let expected = TestTensor::<1>::from_data([2.0], &device);
    x.into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), Tolerance::default());
}

#[test]
fn solve_output_rank_disambiguates_batched_vectors_and_matrix_rhs() {
    let device = Default::default();
    let a = TestTensor::<3>::from_data(
        [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]],
        &device,
    );
    let b = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);

    let vectors = solve::<3, 2, 2>(a.clone(), b.clone());
    let expected_vectors = TestTensor::<2>::from_data([[1.0, 2.0], [1.5, 2.0]], &device);
    vectors
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_vectors.into_data(), Tolerance::default());

    let matrices = solve::<3, 2, 3>(a, b);
    let expected_matrices = TestTensor::<3>::from_data(
        [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 1.0], [1.5, 2.0]]],
        &device,
    );
    matrices
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_matrices.into_data(), Tolerance::default());
}

#[cfg(feature = "autodiff")]
#[test]
fn solve_gradients_accumulate_when_a_is_broadcast() {
    let device = burn_core::tensor::Device::default().autodiff();
    let a = TestTensor::<2>::from_data([[2.0, 0.0], [0.0, 4.0]], &device).require_grad();
    let b = TestTensor::<3>::from_data([[[2.0], [4.0]], [[4.0], [8.0]]], &device).require_grad();
    let x = solve::<2, 3, 3>(a.clone(), b.clone());
    let grads = x.sum().backward();
    let expected_a = TestTensor::<2>::from_data([[-1.5, -1.5], [-0.75, -0.75]], &device);
    let expected_b = TestTensor::<3>::from_data([[[0.5], [0.25]], [[0.5], [0.25]]], &device);
    a.grad(&grads)
        .unwrap()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_a.into_data(), Tolerance::default());
    b.grad(&grads)
        .unwrap()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_b.into_data(), Tolerance::default());
}

#[test]
fn solve_dense_matrices_with_pivots_across_block_boundaries() {
    let device = burn_core::tensor::Device::default();
    for (n, columns) in [(17, 3), (33, 17), (65, 1), (65, 3), (65, 17), (129, 33)] {
        let values = pivoting_matrix_values(n);
        let solution: Vec<f32> = (0..n * columns)
            .map(|i| ((i / columns * 5 + i % columns * 3) % 23) as f32 / 8.0 - 11.0 / 8.0)
            .collect();
        let mut rhs = vec![0.0f32; n * columns];
        for i in 0..n {
            for j in 0..columns {
                rhs[i * columns + j] = (0..n)
                    .map(|k| values[i * n + k] as f64 * solution[k * columns + j] as f64)
                    .sum::<f64>() as f32;
            }
        }
        let a = TestTensor::<2>::from_data(TensorData::new(values, [n, n]), &device);
        let b = TestTensor::<2>::from_data(TensorData::new(rhs, [n, columns]), &device);
        let expected = TensorData::new(solution, [n, columns]);
        let x = solve::<2, 2, 2>(a.clone(), b.clone());
        x.clone()
            .into_data()
            .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(1e-5, 1e-5));
        a.clone()
            .matmul(x)
            .into_data()
            .assert_approx_eq::<FloatElem>(&b.clone().into_data(), Tolerance::rel_abs(1e-5, 1e-5));

        if device.supports_dtype(DType::F64) {
            // All reference values are dyadic fractions, exactly represented in F32.
            let a = a.cast(DType::F64);
            let b = b.cast(DType::F64);
            let x = solve::<2, 2, 2>(a.clone(), b.clone());
            assert_eq!(x.dtype(), DType::F64);
            x.clone()
                .into_data()
                .assert_approx_eq::<f64>(&expected, Tolerance::rel_abs(1e-12, 1e-12));
            a.matmul(x)
                .into_data()
                .assert_approx_eq::<f64>(&b.into_data(), Tolerance::rel_abs(1e-12, 1e-12));
        }
    }
}

#[test]
fn solve_broadcasts_both_inputs_across_multiple_batch_dimensions() {
    let device = Default::default();
    let a = TestTensor::<4>::from_data(
        [[[[0.0, 2.0], [1.0, 3.0]]], [[[3.0, 1.0], [1.0, 2.0]]]],
        &device,
    );
    let b = TestTensor::<4>::from_data([[[[4.0], [7.0]], [[9.0], [8.0]], [[1.0], [0.0]]]], &device);
    let x = solve::<4, 4, 4>(a.clone(), b.clone());
    assert_eq!(x.dims(), [2, 3, 2, 1]);
    a.expand([2, 3, 2, 2])
        .matmul(x)
        .into_data()
        .assert_approx_eq::<FloatElem>(&b.expand([2, 3, 2, 1]).into_data(), Tolerance::default());
}

#[test]
fn solve_accepts_transposed_and_sliced_inputs() {
    let device = Default::default();
    let a = TestTensor::<2>::from_data(
        [[99.0, 99.0, 99.0], [99.0, 0.0, 1.0], [99.0, 2.0, 3.0]],
        &device,
    )
    .slice([1..3, 1..3])
    .transpose();
    let b = TestTensor::<2>::from_data(
        [[99.0, 99.0, 99.0], [99.0, 4.0, 7.0], [99.0, 1.0, 2.0]],
        &device,
    )
    .slice([1..3, 1..3])
    .transpose();
    let x = solve::<2, 2, 2>(a.clone(), b.clone());
    let expected = TensorData::from([[1.0, 0.5], [2.0, 0.5]]);
    x.clone()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
    a.matmul(x)
        .into_data()
        .assert_approx_eq::<FloatElem>(&b.into_data(), Tolerance::default());
}

#[test]
fn solve_half_precision_preserves_dtype() {
    let device = burn_core::tensor::Device::default();
    for dtype in [DType::F16, DType::BF16] {
        if !device.supports_dtype(dtype) {
            continue;
        }
        let a = TestTensor::<2>::from_data([[0.0, 2.0], [1.0, 3.0]], &device).cast(dtype);
        let b = TestTensor::<2>::from_data([[4.0, 1.0], [7.0, 2.0]], &device).cast(dtype);
        let x = solve::<2, 2, 2>(a, b);
        assert_eq!(x.dtype(), dtype);
        x.cast(DType::F32)
            .into_data()
            .assert_approx_eq::<FloatElem>(
                &TensorData::from([[1.0, 0.5], [2.0, 0.5]]),
                Tolerance::default(),
            );
    }
}

#[test]
fn solve_empty_batches_broadcast_without_factorization() {
    let device = Default::default();
    let a = TestTensor::<4>::empty([0, 1, 2, 2], &device);
    let b = TestTensor::<4>::ones([1, 3, 2, 1], &device);
    assert_eq!(solve::<4, 4, 4>(a, b).dims(), [0, 3, 2, 1]);

    let a = TestTensor::<4>::zeros([1, 3, 2, 2], &device);
    let b = TestTensor::<3>::empty([0, 1, 2], &device);
    assert_eq!(solve::<4, 3, 3>(a, b).dims(), [0, 3, 2]);
}

#[cfg(feature = "autodiff")]
#[test]
fn solve_matrix_gradients_reduce_both_broadcast_inputs() {
    for checkpoint in [false, true] {
        let device = burn_core::tensor::Device::default().autodiff();
        let device = if checkpoint {
            device.gradient_checkpointing()
        } else {
            device
        };
        for (track_a, track_b) in [(true, true), (true, false), (false, true)] {
            let a = TestTensor::<4>::from_data(
                [[[[0.0, 2.0], [1.0, 3.0]]], [[[3.0, 1.0], [1.0, 2.0]]]],
                &device,
            );
            let b = TestTensor::<4>::from_data(
                [[
                    [[4.0, 1.0], [7.0, 2.0]],
                    [[9.0, 3.0], [8.0, 1.0]],
                    [[1.0, -1.0], [0.0, 2.0]],
                ]],
                &device,
            );
            let a = if track_a { a.require_grad() } else { a };
            let b = if track_b { b.require_grad() } else { b };
            let grads = solve::<4, 4, 4>(a.clone(), b.clone()).sum().backward();
            if track_a {
                // For sum(X), dA = -(A^-T 1) X^T, summed over B's batches.
                let expected = TensorData::from([
                    [[[-5.5, 8.5], [5.5, -8.5]]],
                    [[[-0.56, -1.72], [-1.12, -3.44]]],
                ]);
                let gradient = a.grad(&grads).unwrap();
                assert_eq!(gradient.dims(), a.dims());
                gradient
                    .into_data()
                    .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
            }
            if track_b {
                // dB = A^-T 1, summed over the two matrices in A.
                let expected = TensorData::from([[[[-0.8, -0.8], [1.4, 1.4]]; 3]]);
                let gradient = b.grad(&grads).unwrap();
                assert_eq!(gradient.dims(), b.dims());
                gradient
                    .into_data()
                    .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
            }
        }
    }
}

#[cfg(any(feature = "ndarray", feature = "flex"))]
#[test]
fn solve_accepts_subnormal_pivot_without_reciprocal_overflow() {
    // GPU execution may flush subnormals to zero, even when CPU features are enabled.
    #[cfg(feature = "flex")]
    let device = burn_core::tensor::Device::flex();
    #[cfg(all(feature = "ndarray", not(feature = "flex")))]
    #[allow(deprecated)]
    let device = burn_core::tensor::Device::ndarray();
    let tiny = f32::from_bits(1);
    let a = TestTensor::<2>::from_data([[tiny, 0.0], [tiny, 1.0]], &device);
    let b = TestTensor::<1>::from_data([tiny, 2.0], &device);
    solve::<2, 1, 1>(a, b)
        .into_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([1.0, 2.0]), Tolerance::default());
}

#[cfg(all(
    feature = "autodiff",
    any(feature = "ndarray", feature = "flex", feature = "cubecl-backend")
))]
#[test]
fn solve_zero_rhs_columns_has_zero_gradients() {
    let device = burn_core::tensor::Device::default().autodiff();
    let a = TestTensor::<2>::from_data([[0.0, 2.0], [1.0, 3.0]], &device).require_grad();
    let b = TestTensor::<2>::empty([2, 0], &device).require_grad();
    let grads = solve::<2, 2, 2>(a.clone(), b.clone()).sum().backward();
    a.grad(&grads)
        .unwrap()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::from([[0.0, 0.0], [0.0, 0.0]]),
            Tolerance::default(),
        );
    assert_eq!(b.grad(&grads).unwrap().dims(), [2, 0]);
}

#[test]
#[should_panic(expected = "A is singular")]
fn solve_rejects_singular_pivot_in_later_block() {
    let device = Default::default();
    let n = 65;
    let mut values = vec![0.0f32; n * n];
    for i in 0..n - 1 {
        values[i * n + i] = 1.0;
    }
    let a = TestTensor::<2>::from_data(TensorData::new(values, [n, n]), &device);
    let b = TestTensor::<2>::ones([n, 8], &device);
    let _ = solve::<2, 2, 2>(a, b);
}

// A row permutation forces pivots across panel boundaries in a dense,
// nonsymmetric matrix. Its small off-diagonal entries keep it well conditioned.
fn pivoting_matrix_values(n: usize) -> Vec<f32> {
    (0..n)
        .flat_map(|i| {
            let row = (i + n / 2 + 1) % n;
            (0..n).map(move |j| {
                if row == j {
                    4.0
                } else {
                    ((row * 13 + j * 7) % 17) as f32 / 256.0 - 8.0 / 256.0
                }
            })
        })
        .collect()
}

#[test]
fn solve_large_pivoted_matrix() {
    let device = Default::default();
    let n = 257;
    let columns = 5;
    let a = TestTensor::<2>::from_data(TensorData::new(pivoting_matrix_values(n), [n, n]), &device);
    let values: Vec<f32> = (0..n * columns)
        .map(|i| (i % 19) as f32 / 8.0 - 1.0)
        .collect();
    let expected = TestTensor::<2>::from_data(TensorData::new(values, [n, columns]), &device);
    let b = a.clone().matmul(expected.clone());
    let x = solve::<2, 2, 2>(a.clone(), b.clone());
    x.clone()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), Tolerance::rel_abs(2e-5, 2e-5));
    a.matmul(x)
        .into_data()
        .assert_approx_eq::<FloatElem>(&b.into_data(), Tolerance::rel_abs(2e-5, 2e-5));
}

#[test]
fn solve_strided_shared_matrix_with_strided_rhs_batches() {
    let device = Default::default();
    let n = 33;
    let columns = 5;
    let values = pivoting_matrix_values(n);
    let mut padded = vec![99.0; (n + 2) * (n + 2)];
    for i in 0..n {
        for j in 0..n {
            padded[(j + 1) * (n + 2) + i + 1] = values[i * n + j];
        }
    }
    let a = TestTensor::<2>::from_data(TensorData::new(padded, [n + 2, n + 2]), &device)
        .slice([1..n + 1, 1..n + 1])
        .transpose();
    let values: Vec<f32> = (0..6 * n * columns)
        .map(|i| (i % 29) as f32 / 16.0 - 0.75)
        .collect();
    let expected = TestTensor::<4>::from_data(TensorData::new(values, [3, 2, n, columns]), &device);
    let b = a
        .clone()
        .reshape([1, 1, n, n])
        .matmul(expected.clone())
        .swap_dims(0, 1);
    let x = solve::<2, 4, 4>(a, b);
    assert_eq!(x.dims(), [2, 3, n, columns]);
    x.into_data().assert_approx_eq::<FloatElem>(
        &expected.swap_dims(0, 1).into_data(),
        Tolerance::rel_abs(1e-5, 1e-5),
    );
}

#[cfg(feature = "autodiff")]
#[test]
fn solve_pivoted_adjoint_with_prescribed_gradient() {
    let device = burn_core::tensor::Device::default().autodiff();
    let n = 33;
    let columns = 3;
    let a = TestTensor::<2>::from_data(TensorData::new(pivoting_matrix_values(n), [n, n]), &device);
    let solution: Vec<f32> = (0..n * columns)
        .map(|i| (i % 11) as f32 / 8.0 - 0.5)
        .collect();
    let derivative: Vec<f32> = (0..n * columns)
        .map(|i| (i % 13) as f32 / 16.0 - 0.25)
        .collect();
    let expected_x = TestTensor::<2>::from_data(TensorData::new(solution, [n, columns]), &device);
    let expected_b_grad =
        TestTensor::<2>::from_data(TensorData::new(derivative, [n, columns]), &device);
    // Prescribe dB and construct dX = A^T dB independently of solve's backward.
    let upstream = a.clone().transpose().matmul(expected_b_grad.clone());
    let b = a.clone().matmul(expected_x.clone()).require_grad();
    let a = a.require_grad();
    let x = solve::<2, 2, 2>(a.clone(), b.clone());
    let grads = (x * upstream).sum().backward();
    let expected_a_grad = expected_b_grad.clone().matmul(expected_x.transpose()).neg();
    a.grad(&grads)
        .unwrap()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &expected_a_grad.into_data(),
            Tolerance::rel_abs(1e-5, 1e-5),
        );
    b.grad(&grads)
        .unwrap()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &expected_b_grad.into_data(),
            Tolerance::rel_abs(1e-5, 1e-5),
        );
}

#[test]
#[should_panic(expected = "A is singular")]
fn solve_rejects_late_singular_pivot_with_zero_rhs_columns() {
    let device = Default::default();
    let n = 33;
    let mut values = vec![0.0f32; n * n];
    for i in 0..n - 1 {
        values[i * n + i] = 1.0;
    }
    let a = TestTensor::<2>::from_data(TensorData::new(values, [n, n]), &device);
    let b = TestTensor::<2>::empty([n, 0], &device);
    let _ = solve::<2, 2, 2>(a, b);
}

#[cfg(feature = "cubecl-backend")]
#[test]
fn solve_nan_pivots_do_not_corrupt_neighboring_batch() {
    let device = Default::default();
    let n = 33;
    let finite_a = pivoting_matrix_values(n);
    let expected: Vec<f32> = (0..n).map(|i| (i % 7) as f32 / 4.0).collect();
    let mut values = vec![f32::NAN; n * n];
    values.extend_from_slice(&finite_a);
    let mut rhs = vec![1.0; n];
    for row in 0..n {
        rhs.push(
            (0..n)
                .map(|col| finite_a[row * n + col] as f64 * expected[col] as f64)
                .sum::<f64>() as f32,
        );
    }
    let a = TestTensor::<3>::from_data(TensorData::new(values, [2, n, n]), &device);
    let b = TestTensor::<2>::from_data(TensorData::new(rhs, [2, n]), &device);
    let x = solve::<3, 2, 2>(a, b);
    let values = x.into_data().try_to_vec::<f32>().unwrap();
    assert!(values[..n].iter().all(|value| value.is_nan()));
    TensorData::new(values[n..].to_vec(), [n]).assert_approx_eq::<FloatElem>(
        &TensorData::new(expected, [n]),
        Tolerance::rel_abs(1e-5, 1e-5),
    );
}

#[test]
fn solve_f64_pivoted_matrix_with_three_strided_rhs() {
    let device = burn_core::tensor::Device::default();
    if !device.supports_dtype(DType::F64) {
        return;
    }
    let n = 129;
    let columns = 3;
    let values = pivoting_matrix_values(n);
    let solution: Vec<f64> = (0..n * columns)
        .map(|i| (i % 17) as f64 / 8.0 - 0.75 + (i % 7) as f64 * 1e-9)
        .collect();
    // Store A and B transposed so both inputs exercise noncontiguous indexing.
    // Construct B on the host in F64 to retain solution details smaller than F32 epsilon.
    let mut rhs = vec![0.0f64; columns * n];
    for col in 0..columns {
        for row in 0..n {
            rhs[col * n + row] = (0..n)
                .map(|k| values[k * n + row] as f64 * solution[k * columns + col])
                .sum();
        }
    }
    let a = TestTensor::<2>::from_data(TensorData::new(values, [n, n]), &device)
        .cast(DType::F64)
        .transpose();
    let b = TestTensor::<2>::from_data(TensorData::new(rhs, [columns, n]), (&device, DType::F64))
        .transpose();
    let x = solve::<2, 2, 2>(a.clone(), b.clone());
    assert_eq!(x.dtype(), DType::F64);
    x.clone().into_data().assert_approx_eq::<f64>(
        &TensorData::new(solution, [n, columns]),
        Tolerance::rel_abs(1e-10, 1e-10),
    );
    a.matmul(x)
        .into_data()
        .assert_approx_eq::<f64>(&b.into_data(), Tolerance::rel_abs(1e-10, 1e-10));
}

#[test]
fn solve_large_finite_pivots_without_reciprocal_underflow() {
    let device = Default::default();
    // The pivots' reciprocals are subnormal, but all matrix entries,
    // elimination multipliers, right-hand sides, and solutions stay finite.
    let a = TestTensor::<2>::from_data([[1e38, 0.0], [2.5e37, 1e38]], &device);
    let b = TestTensor::<1>::from_data([1e38, 1.25e38], &device);
    solve::<2, 1, 1>(a, b)
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::from([1.0, 1.0]),
            Tolerance::rel_abs(1e-5, 1e-5),
        );
}
