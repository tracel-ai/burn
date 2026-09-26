use super::*;
use burn_core::tensor::{TensorData, Tolerance};
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
    use burn_core::tensor::DType;
    let device = Default::default();
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

#[cfg(any(feature = "ndarray", feature = "flex"))]
#[test]
fn solve_f64_preserves_precision_and_dtype() {
    use burn_core::tensor::DType;
    let device = Default::default();
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
    let device = Default::default();
    for (n, columns) in [(65, 1), (65, 3), (65, 17), (129, 33)] {
        // Permuting a well-conditioned dense matrix forces row swaps both within
        // and across LU panels, while keeping the reference solution accurate.
        let values: Vec<f32> = (0..n)
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
            .collect();
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

        #[cfg(any(feature = "ndarray", feature = "flex"))]
        {
            use burn_core::tensor::DType;
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

#[cfg(feature = "flex")]
#[test]
fn solve_half_precision_preserves_dtype() {
    use burn_core::tensor::DType;
    let device = Default::default();
    for dtype in [DType::F16, DType::BF16] {
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
    let device = Default::default();
    let tiny = f32::from_bits(1);
    let a = TestTensor::<2>::from_data([[tiny, 0.0], [tiny, 1.0]], &device);
    let b = TestTensor::<1>::from_data([tiny, 2.0], &device);
    solve::<2, 1, 1>(a, b)
        .into_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([1.0, 2.0]), Tolerance::default());
}

#[cfg(all(feature = "autodiff", any(feature = "ndarray", feature = "flex")))]
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
