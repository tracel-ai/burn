use super::*;
use burn_tensor::TensorData;
use burn_tensor::{ElementConversion, Tolerance};

#[test]
fn test_float_matmul_d2() {
    let device = Default::default();
    let tensor_1 = TestTensor::<2>::from_data([[1.0, 7.0], [2.0, 3.0], [1.0, 5.0]], &device);
    let tensor_2 = TestTensor::from_data([[4.0, 7.0, 5.0], [2.0, 3.0, 5.0]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([[18.0, 28.0, 40.0], [14.0, 23.0, 25.0], [14.0, 22.0, 30.0]]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_float_matmul_d3() {
    let device = Default::default();
    let tensor_1 = TestTensor::<3>::from_data([[[1.0, 7.0], [2.0, 3.0]]], &device);
    let tensor_2 = TestTensor::from_data([[[4.0, 7.0], [2.0, 3.0]]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([[[18.0, 28.0], [14.0, 23.0]]]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_float_matmul_broadcast_1() {
    let device = Default::default();
    let tensor_1 = TestTensor::<3>::from_data([[[1.0, 7.0], [2.0, 3.0]]], &device);
    let tensor_2 = TestTensor::from_data(
        [[[4.0, 7.0], [2.0, 3.0]], [[2.0, 5.0], [6.0, 3.0]]],
        &device,
    );

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([[[18.0, 28.0], [14.0, 23.0]], [[44.0, 26.0], [22.0, 19.0]]]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_float_matmul_broadcast_4d() {
    let device = Default::default();
    // [2, 1, 2, 2]
    let tensor_1 = TestTensor::<4>::from_data(
        [[[[1.0, 7.0], [2.0, 3.0]]], [[[2.0, 5.0], [6.0, 3.0]]]],
        &device,
    );
    // [1, 2, 2, 2]
    let tensor_2 = TestTensor::from_data(
        [[[[9.0, 8.0], [1.0, 4.0]], [[2.0, 7.0], [3.0, 5.0]]]],
        &device,
    );

    // [2, 1, 2, 2] @ [1, 2, 2, 2] -> [2, 2, 2, 2]
    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([
        [[[16.0, 36.0], [21.0, 28.0]], [[23.0, 42.0], [13.0, 29.0]]],
        [[[23.0, 36.0], [57.0, 60.0]], [[19.0, 39.0], [21.0, 57.0]]],
    ]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_float_matmul_simple_1() {
    let device = Default::default();
    let tensor_1 = TestTensor::<2>::from_data([[5.0, 14.0], [14.0, 50.0]], &device);
    let tensor_2 = TestTensor::from_data([[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([[15.0, 34.0, 53.0], [42.0, 106.0, 170.0]]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_float_matmul_4_3() {
    let device = Default::default();
    let tensor_1 = TestTensor::<2>::from_data(
        [[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]],
        &device,
    );
    let tensor_2 = TestTensor::from_data(
        [[0., 1., 2.], [4., 5., 6.], [8., 9., 10.], [12., 13., 14.]],
        &device,
    );

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([[56., 62., 68.], [152., 174., 196.], [248., 286., 324.]]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_float_matmul_batch_vec_mat() {
    let device = Default::default();

    // [..., B, 1, K] = [3, 1, 2]
    let tensor_1 = TestTensor::<3>::from_data([[[1.0, 7.0]], [[2.0, 3.0]], [[1.0, 5.0]]], &device);

    // [..., 1, K, N] = [1, 2, 3]
    let tensor_2 = TestTensor::<3>::from_data([[[4.0, 7.0, 5.0], [2.0, 3.0, 5.0]]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);

    // [..., B, 1, N] = [3, 1, 3]
    let expected = TensorData::from([
        [[18.0, 28.0, 40.0]],
        [[14.0, 23.0, 25.0]],
        [[14.0, 22.0, 30.0]],
    ]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_float_matmul_vecmat() {
    let device = Default::default();

    // [..., B, 1, K] = [3, 1, 2]
    let tensor_1 = TestTensor::<3>::from_data([[[1.0, 7.0]], [[2.0, 3.0]], [[1.0, 5.0]]], &device);

    // [..., B, K, N] = [3, 2, 3]
    let tensor_2 = TestTensor::<3>::from_data(
        [
            [[1.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
            [[8.0, 2.0, 3.0], [0.0, 2.0, 4.0]],
            [[4.0, 7.0, 5.0], [2.0, 3.0, 5.0]],
        ],
        &device,
    );

    let tensor_3 = tensor_1.matmul(tensor_2);

    // [..., B, 1, N] = [3, 1, 3]
    let expected = TensorData::from([
        [[22.0, 39.0, 47.0]],
        [[16.0, 10.0, 18.0]],
        [[14.0, 22.0, 30.0]],
    ]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_float_matmul_trivial() {
    let device = Default::default();

    let tensor_1 = TestTensorInt::<1>::arange(0..16, &device)
        .reshape([4, 4])
        .float();

    let tensor_3 = tensor_1.clone().matmul(tensor_1);

    tensor_3.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([
            [56., 62., 68., 74.],
            [152., 174., 196., 218.],
            [248., 286., 324., 362.],
            [344., 398., 452., 506.],
        ]),
        Tolerance::default(),
    );
}

#[test]
fn test_float_matmul_trivial_transposed() {
    let device = Default::default();

    let tensor_1 = TestTensorInt::<1>::arange(0..16, &device)
        .reshape([4, 4])
        .float();

    let tensor_3 = tensor_1.clone().matmul(tensor_1.transpose());

    tensor_3.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([
            [14., 38., 62., 86.],
            [38., 126., 214., 302.],
            [62., 214., 366., 518.],
            [86., 302., 518., 734.],
        ]),
        Tolerance::default(),
    );
}

/// Regression test for batch bug in fused matmul
#[test]
fn test_float_matmul_vecmat_transposed_fused() {
    let device = Default::default();

    let batch1 = 1;
    let batch2 = 2;
    let batch = batch1 * batch2;
    let seq_length = 3;
    let d_model = 32;

    // Guard int arange limits
    #[allow(clippy::unnecessary_cast)]
    if (IntElem::MAX as i64) < seq_length * d_model * batch {
        return;
    }
    if FloatElem::MAX.elem::<f64>() < 269493.0 {
        return;
    }

    let weight: TestTensor<4> = TestTensorInt::arange(0..d_model * batch, &device)
        .reshape([batch1, batch2, 1, d_model])
        .float();
    let signal: TestTensor<4> = TestTensorInt::arange(0..seq_length * d_model * batch, &device)
        .reshape([batch1, batch2, seq_length, d_model])
        .float();

    device.sync().unwrap();
    let weight = weight.transpose();
    let out = signal.matmul(weight) + 5;
    let expected = TensorData::from([[
        [[10421.0], [26293.0], [42165.0]],
        [[172213.0], [220853.0], [269493.0]],
    ]]);
    expected.assert_approx_eq(&out.into_data(), Tolerance::<f32>::strict());
}

#[test]
fn test_float_matmul_4_8() {
    let device = Default::default();

    let tensor_1 = TestTensorInt::<1>::arange(0..32, &device)
        .reshape([4, 8])
        .float();

    let tensor_3 = tensor_1.clone().matmul(tensor_1.transpose());

    tensor_3.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([
            [140., 364., 588., 812.],
            [364., 1100., 1836., 2572.],
            [588., 1836., 3084., 4332.],
            [812., 2572., 4332., 6092.],
        ]),
        Tolerance::default(),
    );
}

#[test]
fn test_float_matmul_simple_2() {
    let device = Default::default();
    let tensor_1 = TestTensor::<2>::from_data([[1.0, 2.0, 3.0, 4.0]], &device);
    let tensor_2 = TestTensor::from_data([[3.0], [4.0], [5.0], [6.0]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([[50.0]]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_float_matmul_simple_3() {
    let device = Default::default();
    let tensor_1 = TestTensor::<2>::from_data(
        [[3., 3., 3.], [4., 4., 4.], [5., 5., 5.], [6., 6., 6.]],
        &device,
    );
    let tensor_2 = TestTensor::from_data(
        [[1., 2., 3., 4.], [1., 2., 3., 4.], [1., 2., 3., 4.]],
        &device,
    );

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([
        [9., 18., 27., 36.],
        [12., 24., 36., 48.],
        [15., 30., 45., 60.],
        [18., 36., 54., 72.],
    ]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_matmul_transposed_lhs() {
    // [2, 3] -> transpose -> [3, 2], matmul with identity [2, 2] -> [3, 2].
    let device = Default::default();
    let lhs = TestTensor::<2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let rhs = TestTensor::<2>::from_data([[1.0, 0.0], [0.0, 1.0]], &device);

    let output = lhs.transpose().matmul(rhs);
    let expected = TensorData::from([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_matmul_transposed_rhs() {
    // identity [2, 2] matmul [3, 2].transpose() -> [2, 3].
    let device = Default::default();
    let lhs = TestTensor::<2>::from_data([[1.0, 0.0], [0.0, 1.0]], &device);
    let rhs = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device);

    let output = lhs.matmul(rhs.transpose());
    let expected = TensorData::from([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_matmul_both_transposed() {
    // [2, 3].T [3, 2] matmul [3, 2].T [2, 3] -> [3, 3].
    let device = Default::default();
    let lhs = TestTensor::<2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let rhs = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device);

    let output = lhs.transpose().matmul(rhs.transpose());
    // lhs.T = [[1,4],[2,5],[3,6]]; rhs.T = [[1,3,5],[2,4,6]]
    // row0: 1+8,3+16,5+24 = 9,19,29
    // row1: 2+10,6+20,10+30 = 12,26,40
    // row2: 3+12,9+24,15+36 = 15,33,51
    let expected = TensorData::from([[9.0, 19.0, 29.0], [12.0, 26.0, 40.0], [15.0, 33.0, 51.0]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_matmul_batched_transposed_rhs() {
    // QK^T attention pattern: q.matmul(k.swap_dims(1, 2)).
    let device = Default::default();
    let q = TestTensor::<3>::from_data(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
        &device,
    );
    let k = TestTensor::<3>::from_data(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        ],
        &device,
    );

    let output = q.matmul(k.swap_dims(1, 2));
    // batch 0: [[1,2,3],[4,5,6]] @ [[1,0],[0,1],[0,0]] = [[1,2],[4,5]]
    // batch 1: [[7,8,9],[10,11,12]] @ [[1,2],[1,2],[1,2]] = [[24,48],[33,66]]
    let expected = TensorData::from([[[1.0, 2.0], [4.0, 5.0]], [[24.0, 48.0], [33.0, 66.0]]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_matmul_batched_transposed_lhs() {
    // [2, 2, 3].swap_dims(1, 2) -> [2, 3, 2]; rhs [2, 2, 2] -> [2, 3, 2].
    let device = Default::default();
    let a = TestTensor::<3>::from_data(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
        &device,
    );
    let b = TestTensor::<3>::from_data(
        [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]],
        &device,
    );

    let output = a.swap_dims(1, 2).matmul(b);
    // batch 0: a.T = [[1,4],[2,5],[3,6]] @ I = [[1,4],[2,5],[3,6]]
    // batch 1: a.T = [[7,10],[8,11],[9,12]] @ 2I = [[14,20],[16,22],[18,24]]
    let expected = TensorData::from([
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
        [[14.0, 20.0], [16.0, 22.0], [18.0, 24.0]],
    ]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_matmul_batched_both_transposed() {
    // a: [2, 3, 2].swap_dims -> [2, 2, 3]; b: [2, 2, 3].swap_dims -> [2, 3, 2];
    // result: [2, 2, 2].
    let device = Default::default();
    let a = TestTensor::<3>::from_data(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ],
        &device,
    );
    let b = TestTensor::<3>::from_data(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
        &device,
    );

    let output = a.swap_dims(1, 2).matmul(b.swap_dims(1, 2));
    // batch 0: a.T = [[1,3,5],[2,4,6]]; b.T = [[1,4],[2,5],[3,6]]
    //   row0: 1+6+15=22, 4+15+30=49
    //   row1: 2+8+18=28, 8+20+36=64
    // batch 1: a.T = [[7,9,11],[8,10,12]]; b.T = [[7,10],[8,11],[9,12]]
    //   row0: 49+72+99=220, 70+99+132=301
    //   row1: 56+80+108=244, 80+110+144=334
    let expected = TensorData::from([
        [[22.0, 49.0], [28.0, 64.0]],
        [[220.0, 301.0], [244.0, 334.0]],
    ]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_matmul_batched_broadcast_transposed() {
    // lhs [1, 2, 3].swap_dims(1, 2) -> [1, 3, 2] broadcasts against rhs [4, 2, 2].
    let device = Default::default();
    let lhs = TestTensor::<3>::from_data([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], &device);
    let rhs = TestTensor::<3>::from_data(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ],
        &device,
    );

    let output = lhs.swap_dims(1, 2).matmul(rhs);
    // lhs.T = [[1,4],[2,5],[3,6]] (shape [1, 3, 2]) broadcast to [4, 3, 2]
    // batch 0 (I): [[1,4],[2,5],[3,6]]
    // batch 1 (2I): [[2,8],[4,10],[6,12]]
    // batch 2 (ones): each output row is [sum, sum] where sum is the row of lhs.T
    // batch 3 (swap): [[4,1],[5,2],[6,3]]
    let expected = TensorData::from([
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
        [[2.0, 8.0], [4.0, 10.0], [6.0, 12.0]],
        [[5.0, 5.0], [7.0, 7.0], [9.0, 9.0]],
        [[4.0, 1.0], [5.0, 2.0], [6.0, 3.0]],
    ]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
#[should_panic]
fn float_should_panic_when_inner_dimensions_are_not_equal() {
    let device = Default::default();
    let tensor_1 = TestTensor::<2>::from_data([[3., 3.], [4., 4.], [5., 5.], [6., 6.]], &device);
    let tensor_2 = TestTensor::from_data(
        [[1., 2., 3., 4.], [1., 2., 3., 4.], [1., 2., 3., 4.]],
        &device,
    );

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([
        [9., 18., 27., 36.],
        [12., 24., 36., 48.],
        [15., 30., 45., 60.],
        [18., 36., 54., 72.],
    ]);

    tensor_3.into_data().assert_eq(&expected, false);
}
