use super::*;
use burn_tensor::TensorData;

#[cfg(feature = "flex-rayon")]
fn with_four_rayon_threads<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("Failed to create Rayon thread pool");

    pool.install(f)
}

#[test]
fn test_int_matmul_d2() {
    let device = Default::default();
    let tensor_1 = TestTensorInt::<2>::from_ints([[1, 7], [2, 3], [1, 5]], &device);
    let tensor_2 = TestTensorInt::<2>::from_ints([[4, 7, 5], [2, 3, 5]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([[18, 28, 40], [14, 23, 25], [14, 22, 30]]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_int_matmul_d3() {
    let device = Default::default();
    let tensor_1 = TestTensorInt::<3>::from_ints([[[1, 7], [2, 3]]], &device);
    let tensor_2 = TestTensorInt::<3>::from_ints([[[4, 7], [2, 3]]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([[[18, 28], [14, 23]]]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_int_matmul_broadcast_1() {
    let device = Default::default();
    let tensor_1 = TestTensorInt::<3>::from_ints([[[1, 7], [2, 3]]], &device);
    let tensor_2 = TestTensorInt::from_ints([[[4, 7], [2, 3]], [[2, 5], [6, 3]]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([[[18, 28], [14, 23]], [[44, 26], [22, 19]]]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_int_matmul_broadcast_4d() {
    let device = Default::default();
    // [2, 1, 2, 2]
    let tensor_1 = TestTensorInt::<4>::from_ints([[[[1, 7], [2, 3]]], [[[2, 5], [6, 3]]]], &device);
    // [1, 2, 2, 2]
    let tensor_2 = TestTensorInt::from_ints([[[[9, 8], [1, 4]], [[2, 7], [3, 5]]]], &device);

    // [2, 1, 2, 2] @ [1, 2, 2, 2] -> [2, 2, 2, 2]
    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([
        [[[16, 36], [21, 28]], [[23, 42], [13, 29]]],
        [[[23, 36], [57, 60]], [[19, 39], [21, 57]]],
    ]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_int_matmul_simple_1() {
    let device = Default::default();
    let tensor_1 = TestTensorInt::<2>::from_ints([[5, 14], [14, 25]], &device);
    let tensor_2 = TestTensorInt::from_ints([[3, 4, 5], [0, 1, 2]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([[15, 34, 53], [42, 81, 120]]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_int_matmul_4_3() {
    if (IntElem::MAX as u32) < 324 {
        return;
    }

    let device = Default::default();
    let tensor_1 =
        TestTensorInt::<2>::from_ints([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], &device);
    let tensor_2 =
        TestTensorInt::from_ints([[0, 1, 2], [4, 5, 6], [8, 9, 10], [12, 13, 14]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([[56, 62, 68], [152, 174, 196], [248, 286, 324]]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_int_matmul_trivial() {
    if (IntElem::MAX as u32) < 506 {
        return;
    }

    let device = Default::default();

    let tensor_1 = TestTensorInt::<1>::arange(0..16, &device).reshape([4, 4]);

    let tensor_3 = tensor_1.clone().matmul(tensor_1);

    tensor_3.into_data().assert_eq(
        &TensorData::from([
            [56, 62, 68, 74],
            [152, 174, 196, 218],
            [248, 286, 324, 362],
            [344, 398, 452, 506],
        ]),
        false,
    );
}

#[test]
fn test_int_matmul_trivial_transposed() {
    if (IntElem::MAX as u32) < 734 {
        return;
    }

    let device = Default::default();

    let tensor_1 = TestTensorInt::<1>::arange(0..16, &device).reshape([4, 4]);

    let tensor_3 = tensor_1.clone().matmul(tensor_1.transpose());

    tensor_3.into_data().assert_eq(
        &TensorData::from([
            [14, 38, 62, 86],
            [38, 126, 214, 302],
            [62, 214, 366, 518],
            [86, 302, 518, 734],
        ]),
        false,
    );
}

#[test]
fn test_int_matmul_4_8() {
    if (IntElem::MAX as u32) < 6092 {
        return;
    }

    let device = Default::default();

    let tensor_1 = TestTensorInt::<1>::arange(0..32, &device).reshape([4, 8]);

    let tensor_3 = tensor_1.clone().matmul(tensor_1.transpose());

    tensor_3.into_data().assert_eq(
        &TensorData::from([
            [140, 364, 588, 812],
            [364, 1100, 1836, 2572],
            [588, 1836, 3084, 4332],
            [812, 2572, 4332, 6092],
        ]),
        false,
    );
}

#[test]
fn test_int_matmul_simple_2() {
    let device = Default::default();
    let tensor_1 = TestTensorInt::<2>::from_ints([[1, 2, 3, 4]], &device);
    let tensor_2 = TestTensorInt::from_ints([[3], [4], [5], [6]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([[50]]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_int_matmul_simple_3() {
    let device = Default::default();
    let tensor_1 =
        TestTensorInt::<2>::from_ints([[3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]], &device);
    let tensor_2 = TestTensorInt::from_ints([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([
        [9, 18, 27, 36],
        [12, 24, 36, 48],
        [15, 30, 45, 60],
        [18, 36, 54, 72],
    ]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
#[should_panic]
fn int_should_panic_when_inner_dimensions_are_not_equal() {
    let device = Default::default();
    let tensor_1 = TestTensorInt::<2>::from_ints([[3, 3], [4, 4], [5, 5], [6, 6]], &device);
    let tensor_2 = TestTensorInt::from_ints([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([
        [9, 18, 27, 36],
        [12, 24, 36, 48],
        [15, 30, 45, 60],
        [18, 36, 54, 72],
    ]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_int_matmul_large_matches_naive() {
    // Large enough to take multithreaded paths on CPU backends; k = 203 is not
    // a multiple of any SIMD width, so vectorized dot products hit their tail.
    let (m, k, n) = (200, 203, 197);
    let lhs: Vec<i32> = (0..m * k).map(|i| (i % 17) as i32 - 8).collect();
    let rhs: Vec<i32> = (0..k * n).map(|i| (i % 13) as i32 - 6).collect();
    let naive = |lhs: &[i32]| {
        let mut out = vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] = (0..k).map(|p| lhs[i * k + p] * rhs[p * n + j]).sum();
            }
        }
        out
    };
    let expected = naive(&lhs);

    let device = Default::default();
    let output =
        TestTensorInt::<2>::from_data(TensorData::new(lhs.clone(), [m, k]), &device).matmul(
            TestTensorInt::from_data(TensorData::new(rhs.clone(), [k, n]), &device),
        );
    output
        .into_data()
        .assert_eq(&TensorData::new(expected.clone(), [m, n]), false);

    // Batched: the second batch negates lhs so each batch has distinct output.
    let lhs_neg: Vec<i32> = lhs.iter().map(|v| -v).collect();
    let expected_batched = [expected, naive(&lhs_neg)].concat();
    let output =
        TestTensorInt::<3>::from_data(TensorData::new([lhs, lhs_neg].concat(), [2, m, k]), &device)
            .matmul(TestTensorInt::from_data(
                TensorData::new(rhs.repeat(2), [2, k, n]),
                &device,
            ));
    output
        .into_data()
        .assert_eq(&TensorData::new(expected_batched, [2, m, n]), false);
}

fn assert_batched_i32_matmul_matches_naive(
    lhs: Vec<i32>,
    rhs: Vec<i32>,
    lhs_batches: usize,
    rhs_batches: usize,
    m: usize,
    k: usize,
    n: usize,
) {
    let batch_size = lhs_batches.max(rhs_batches);

    // Independent scalar reference implementation.
    let mut expected = vec![0i32; batch_size * m * n];

    for b in 0..batch_size {
        let lhs_batch = if lhs_batches == 1 { 0 } else { b };
        let rhs_batch = if rhs_batches == 1 { 0 } else { b };

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0i32;

                for p in 0..k {
                    let a = lhs[lhs_batch * m * k + i * k + p];
                    let b = rhs[rhs_batch * k * n + p * n + j];

                    sum = sum.wrapping_add(a.wrapping_mul(b));
                }

                expected[b * m * n + i * n + j] = sum;
            }
        }
    }

    let device = Default::default();

    let lhs_tensor =
        TestTensorInt::<3>::from_data(TensorData::new(lhs, [lhs_batches, m, k]), &device);

    let rhs_tensor =
        TestTensorInt::<3>::from_data(TensorData::new(rhs, [rhs_batches, k, n]), &device);

    let output = lhs_tensor.matmul(rhs_tensor);

    output
        .into_data()
        .assert_eq(&TensorData::new(expected, [batch_size, m, n]), false);
}

#[test]
fn test_int_matmul_parallel_correctness_cases() {
    // 1. Basic batched matrix multiplication.
    assert_batched_i32_matmul_matches_naive(
        vec![1, 2, 3, 4, 5, 6],
        vec![1, 2, 3, 4, 5, 6],
        1,
        1,
        2,
        3,
        2,
    );

    // 2. Multiple batches.
    assert_batched_i32_matmul_matches_naive(
        (0..4 * 2 * 3).map(|i| (i % 17) as i32 - 8).collect(),
        (0..4 * 3 * 2).map(|i| (i % 13) as i32 - 6).collect(),
        4,
        4,
        2,
        3,
        2,
    );

    // 3. Single batch with many output rows.
    // Exercises row-level parallelism when Rayon is enabled.
    assert_batched_i32_matmul_matches_naive(
        (0..200 * 203).map(|i| (i % 17) as i32 - 8).collect(),
        (0..203 * 197).map(|i| (i % 13) as i32 - 6).collect(),
        1,
        1,
        200,
        203,
        197,
    );

    // 4. Broadcast RHS batch dimension.
    assert_batched_i32_matmul_matches_naive(
        (0..4 * 2 * 3).map(|i| (i % 17) as i32 - 8).collect(),
        (0..3 * 2).map(|i| (i % 13) as i32 - 6).collect(),
        4,
        1,
        2,
        3,
        2,
    );

    // 5. Broadcast LHS batch dimension.
    assert_batched_i32_matmul_matches_naive(
        (0..2 * 3).map(|i| (i % 17) as i32 - 8).collect(),
        (0..4 * 3 * 2).map(|i| (i % 13) as i32 - 6).collect(),
        1,
        4,
        2,
        3,
        2,
    );

    // 6. Uneven row chunks.
    // Five rows cannot be evenly divided into typical parallel chunks.
    assert_batched_i32_matmul_matches_naive(
        (0..3 * 5 * 7).map(|i| (i % 17) as i32 - 8).collect(),
        (0..3 * 7 * 4).map(|i| (i % 13) as i32 - 6).collect(),
        3,
        3,
        5,
        7,
        4,
    );
}

fn test_int_matmul_large_broadcast_lhs_matches_naive_inner() {
    let (m, k, n) = (201, 203, 197);

    // Two distinct matrices on each side, so incorrect batch offsets
    // cannot accidentally produce the expected result.
    let lhs0: Vec<i32> = (0..m * k).map(|i| (i % 17) as i32 - 8).collect();
    let lhs1: Vec<i32> = (0..m * k).map(|i| (i % 19) as i32 - 9).collect();

    let rhs0: Vec<i32> = (0..k * n).map(|i| (i % 13) as i32 - 6).collect();
    let rhs1: Vec<i32> = (0..k * n).map(|i| (i % 11) as i32 - 5).collect();

    // Scalar reference using the same wrapping arithmetic as i32 matmul.
    let naive = |lhs: &[i32], rhs: &[i32]| {
        let mut out = vec![0i32; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0i32;

                for p in 0..k {
                    sum = sum.wrapping_add(lhs[i * k + p].wrapping_mul(rhs[p * n + j]));
                }

                out[i * n + j] = sum;
            }
        }

        out
    };

    let device = Default::default();

    // Case 1: Broadcast LHS.
    //
    // [1, 201, 203] x [2, 203, 197]
    //     -> [2, 201, 197]
    //
    // Both output batches must use the same LHS matrix.
    let expected_lhs_broadcast = [naive(&lhs0, &rhs0), naive(&lhs0, &rhs1)].concat();

    let output = TestTensorInt::<3>::from_data(TensorData::new(lhs0.clone(), [1, m, k]), &device)
        .matmul(TestTensorInt::from_data(
            TensorData::new([rhs0.clone(), rhs1.clone()].concat(), [2, k, n]),
            &device,
        ));

    output
        .into_data()
        .assert_eq(&TensorData::new(expected_lhs_broadcast, [2, m, n]), false);

    // Case 2: Broadcast RHS.
    //
    // [2, 201, 203] x [1, 203, 197]
    //     -> [2, 201, 197]
    //
    // Both output batches must use the same RHS matrix.
    let expected_rhs_broadcast = [naive(&lhs0, &rhs0), naive(&lhs1, &rhs0)].concat();

    let output =
        TestTensorInt::<3>::from_data(TensorData::new([lhs0, lhs1].concat(), [2, m, k]), &device)
            .matmul(TestTensorInt::from_data(
                TensorData::new(rhs0, [1, k, n]),
                &device,
            ));

    output
        .into_data()
        .assert_eq(&TensorData::new(expected_rhs_broadcast, [2, m, n]), false);
}

#[test]
fn test_int_matmul_large_broadcast_lhs_matches_naive() {
    #[cfg(feature = "flex-rayon")]
    return with_four_rayon_threads(|| {
        test_int_matmul_large_broadcast_lhs_matches_naive_inner();
    });

    #[cfg(not(feature = "flex-rayon"))]
    test_int_matmul_large_broadcast_lhs_matches_naive_inner();
}
