use super::*;
use burn_tensor::{Distribution, Tolerance, linalg::det, s};

// ---------------------------------------------------------------------
// Small Matrices (single batch)
// ---------------------------------------------------------------------

#[test]
fn test_det_1x1_batched() {
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data([[[5.0]]], &device);
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor);
    let expected = TestTensor::<1>::from_data([5.0], &device);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}

#[test]
fn test_det_2x2_batched() {
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data([[[4.0, 3.0], [6.0, 3.0]]], &device);
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor);
    // det = 4*3 - 3*6 = 12 - 18 = -6
    let expected = TestTensor::<1>::from_data([-6.0], &device);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}

#[test]
fn test_det_3x3_batched() {
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data(
        [[[4.0, 7.0, 3.0], [6.0, 1.0, 3.0], [8.0, 3.0, 7.0]]],
        &device,
    );
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor);
    // det = 4*(1*7 - 3*3) - 7*(6*7 - 3*8) + 3*(6*3 - 1*8)
    // = 4*(7-9) - 7*(42-24) + 3*(18-8) = 4*(-2) - 7*18 + 3*10 = -8 -126 +30 = -104
    let expected = TestTensor::<1>::from_data([-104.0], &device);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}

// ---------------------------------------------------------------------
// Special Matrices (single batch)
// ---------------------------------------------------------------------

#[test]
fn test_det_3x3_identity_matrix() {
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]], &device);
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor);
    let expected = TestTensor::<1>::from_data([1.0], &device);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}

#[test]
fn test_det_300x300_identity_matrix() {
    let device = Default::default();
    let tensor = TestTensor::eye(300, &device).unsqueeze_dim(0);
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor);
    let expected = TestTensor::<1>::from_data([1.0], &device);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}

#[test]
fn test_det_3x3_singular_zero_col() {
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data(
        [[[0.0, 4.0, 2.0], [0.0, 2.0, 1.0], [0.0, 3.0, 9.0]]],
        &device,
    );
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor);
    let expected = TestTensor::<1>::from_data([0.0], &device);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}

#[test]
fn test_det_20x20_singular_zero_col() {
    let device = Default::default();
    let mut tensor = TestTensor::random([1, 20, 20], Distribution::Default, &device);
    tensor = tensor.slice_fill(s![.., .., 16], 0.0);
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor);
    let expected = TestTensor::<1>::from_data([0.0], &device);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}

#[test]
fn test_det_3x3_singular_linearly_dependent() {
    let device = Default::default();
    // Rows: [1,2,3], [2,4,6], [3,6,9] -> linearly dependent
    let tensor = TestTensor::<3>::from_data([[[1., 2., 3.], [2., 4., 6.], [3., 6., 9.]]], &device);
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor);
    let expected = TestTensor::<1>::from_data([0.0], &device);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}

#[test]
fn test_det_20x20_singular_linearly_dependent() {
    let device = Default::default();
    let mut tensor = TestTensor::random([1, 20, 20], Distribution::Default, &device);
    let lin_dep_row1 = tensor.clone().slice(s![.., 5, ..]);
    let lin_dep_row2 = lin_dep_row1.mul_scalar(3);
    tensor = tensor.slice_assign(s![.., 17, ..], lin_dep_row2);
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor);
    let expected = TestTensor::<1>::from_data([0.0], &device);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}

#[test]
fn test_det_3x3_upper_triangular() {
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data([[[1., 2., 3.], [0., 4., 5.], [0., 0., 6.]]], &device);
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor);
    // det = 1 * 4 * 6 = 24
    let expected = TestTensor::<1>::from_data([24.0], &device);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}

#[test]
fn test_det_3x3_lower_triangular() {
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data([[[2., 0., 0.], [3., 4., 0.], [5., 6., 7.]]], &device);
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor);
    // det = 2 * 4 * 7 = 56
    let expected = TestTensor::<1>::from_data([56.0], &device);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}

#[test]
fn test_det_3x3_diagonal_matrix() {
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data([[[3., 0., 0.], [0., 5., 0.], [0., 0., 2.]]], &device);
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor);
    // det = 3 * 5 * 2 = 30
    let expected = TestTensor::<1>::from_data([30.0], &device);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}

#[test]
fn test_det_10x10_all_zeros() {
    let device = Default::default();
    let tensor = TestTensor::<3>::zeros([1, 10, 10], &device);
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor);
    let expected = TestTensor::<1>::from_data([0.0], &device);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}

#[test]
fn test_det_300x300_all_zeros() {
    let device = Default::default();
    let tensor = TestTensor::<3>::zeros([1, 300, 300], &device);
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor);
    let expected = TestTensor::<1>::from_data([0.0], &device);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}

#[test]
fn test_det_10x10_all_ones() {
    let device = Default::default();
    // matrix of all ones has rank 1 -> det = 0
    let tensor = TestTensor::<3>::ones([1, 10, 10], &device);
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor);
    let expected = TestTensor::<1>::from_data([0.0], &device);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}

#[test]
fn test_det_300x300_all_ones() {
    let device = Default::default();
    // matrix of all ones has rank 1 -> det = 0
    let tensor = TestTensor::<3>::ones([1, 300, 300], &device);
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor);
    let expected = TestTensor::<1>::from_data([0.0], &device);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}

#[test]
fn test_det_large_diagonal() {
    let device = Default::default();
    // 8x8 diagonal with known product: 8! = 40320
    let eye = Tensor::<TestBackend, 2>::eye(8, &device);
    let values: Tensor<TestBackend, 1> = Tensor::arange(1..9, &device).float();
    let diag_2d = eye * values.unsqueeze::<2>();
    let tensor = diag_2d.unsqueeze_dim(0);
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor);
    let expected = TestTensor::<1>::from_data([40320.0], &device);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}

#[test]
fn test_det_mixed_singular_non_singular_batch() {
    let device = Default::default();
    let mut singular_tensor =
        Tensor::<TestBackend, 2>::random([10, 10], Distribution::Default, &device);
    singular_tensor = singular_tensor.slice_fill(s![.., 8], 0.0);
    let identity_tensor = Tensor::<TestBackend, 2>::eye(10, &device);
    let input_tensor = Tensor::stack(vec![singular_tensor, identity_tensor], 0);
    let det_tensor = det::<TestBackend, 3, 2, 1>(input_tensor);
    let expected = TestTensor::<1>::from_data([0.0, 1.0], &device);
    let tolerance = Tolerance::default();
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}

// ---------------------------------------------------------------------
// Batched Tensors (3D, multiple matrices)
// ---------------------------------------------------------------------

#[test]
fn test_det_batch_of_1x1() {
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data([[[2.0]], [[3.0]], [[5.0]]], &device);
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor);
    let expected = TestTensor::<1>::from_data([2.0, 3.0, 5.0], &device);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}

#[test]
fn test_det_batch_of_2x2() {
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data(
        [
            [[1.0, 2.0], [3.0, 4.0]], // det = -2
            [[2.0, 0.0], [0.0, 3.0]], // det = 6
            [[5.0, 6.0], [7.0, 8.0]], // det = -2
        ],
        &device,
    );
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor);
    let expected = TestTensor::<1>::from_data([-2.0, 6.0, -2.0], &device);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}

#[test]
fn test_det_batch_of_3x3() {
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data(
        [
            [[4.0, 7.0, 3.0], [6.0, 1.0, 3.0], [8.0, 3.0, 7.0]], // det = -104
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], // det = 6
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], // det = 0 (singular)
        ],
        &device,
    );
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor);
    let expected = TestTensor::<1>::from_data([-104.0, 6.0, 0.0], &device);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}

// ---------------------------------------------------------------------
// 4D Tensors (two batch dimensions)
// ---------------------------------------------------------------------

#[test]
fn test_det_4d_square() {
    let device = Default::default();
    // Shape [2, 2, 3, 3]
    let tensor = TestTensor::<4>::from_data(
        [
            [
                [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                [[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]],
            ],
            [
                [[3., 0., 0.], [0., 3., 0.], [0., 0., 3.]],
                [[4., 0., 0.], [0., 4., 0.], [0., 0., 4.]],
            ],
        ],
        &device,
    );
    let det_tensor = det::<TestBackend, 4, 3, 2>(tensor);
    let expected = TestTensor::<2>::from_data([[1.0, 8.0], [27.0, 64.0]], &device);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}

#[test]
fn test_det_4d_rectangular_batch() {
    let device = Default::default();
    let tensor = TestTensor::<4>::random([3, 4, 5, 5], Distribution::Default, &device);
    let det_tensor = det::<TestBackend, 4, 3, 2>(tensor.clone());
    assert_eq!(det_tensor.dims(), [3, 4]);
}

// ---------------------------------------------------------------------
// Edge Cases and Properties
// ---------------------------------------------------------------------

#[test]
fn test_det_product_property() {
    let device = Default::default();
    // det(A * B) = det(A) * det(B)
    let a = TestTensor::<3>::random([1, 10, 10], Distribution::Default, &device);
    let b = TestTensor::<3>::random([1, 10, 10], Distribution::Default, &device);
    let c = a.clone().matmul(b.clone());
    let det_a = det::<TestBackend, 3, 2, 1>(a);
    let det_b = det::<TestBackend, 3, 2, 1>(b);
    let det_c = det::<TestBackend, 3, 2, 1>(c);
    let det_ab = det_a * det_b;
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_c
        .into_data()
        .assert_approx_eq::<FloatElem>(&det_ab.into_data(), tolerance);
}

#[test]
fn test_det_transpose_invariance() {
    let device = Default::default();
    let tensor = TestTensor::<3>::random([1, 10, 10], Distribution::Default, &device);
    let det_original = det::<TestBackend, 3, 2, 1>(tensor.clone());
    let det_transpose = det::<TestBackend, 3, 2, 1>(tensor.transpose());
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_original
        .into_data()
        .assert_approx_eq::<FloatElem>(&det_transpose.into_data(), tolerance);
}

#[test]
fn test_det_negative_elements() {
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data([[[-1.0, 2.0], [3.0, -4.0]]], &device);
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor);
    // det = (-1)*(-4) - (2*3) = 4 - 6 = -2
    let expected = TestTensor::<1>::from_data([-2.0], &device);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}

#[test]
fn test_det_f16_dtype_roundtrip_1x1() {
    // Does not perform upcasting when the tests are run in normal mode (not f16)
    let device = Default::default();
    let tensor = TestTensor::<3>::random([1, 1, 1], Distribution::Default, &device);
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor.clone());
    assert_eq!(tensor.dtype(), det_tensor.dtype());
}

#[test]
fn test_det_f16_dtype_roundtrip_2x2() {
    // Does not perform upcasting when the tests are run in normal mode (not f16)
    let device = Default::default();
    let tensor = TestTensor::<3>::random([1, 2, 2], Distribution::Default, &device);
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor.clone());
    assert_eq!(tensor.dtype(), det_tensor.dtype());
}

#[test]
fn test_det_f16_dtype_roundtrip_3x3() {
    // Does not perform upcasting when the tests are run in normal mode (not f16)
    let device = Default::default();
    let tensor = TestTensor::<3>::random([1, 3, 3], Distribution::Default, &device);
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor.clone());
    assert_eq!(tensor.dtype(), det_tensor.dtype());
}

#[test]
fn test_det_f16_dtype_roundtrip_10x10() {
    // Does not perform upcasting when the tests are run in normal mode (not f16)
    let device = Default::default();
    let tensor = TestTensor::<3>::random([1, 10, 10], Distribution::Default, &device);
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor.clone());
    assert_eq!(tensor.dtype(), det_tensor.dtype());
}

// ---------------------------------------------------------------------
// Panic Tests (invalid input)
// ---------------------------------------------------------------------

#[test]
#[should_panic(expected = "The input tensor must have at least 3 dimensions")]
fn test_det_panic_rank_less_than_3() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    let _ = det::<TestBackend, 2, 1, 0>(tensor);
}

#[test]
#[should_panic(expected = "The last two dimensions of the input tensor must be equal")]
fn test_det_panic_non_square_last_dims() {
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], &device); // shape [1,2,3]
    let _ = det::<TestBackend, 3, 2, 1>(tensor);
}

#[test]
#[should_panic(expected = "D - 1 = D1 must hold for the generic parameters")]
fn test_det_panic_invalid_d1_generic() {
    let device = Default::default();
    let tensor = TestTensor::<3>::ones([1, 2, 2], &device);
    let _ = det::<TestBackend, 3, 3, 2>(tensor);
}

#[test]
#[should_panic(expected = "The output tensor rank must be less than input tensor rank by 2")]
fn test_det_panic_invalid_d2_generic() {
    let device = Default::default();
    let tensor = TestTensor::<3>::ones([1, 2, 2], &device);
    let _ = det::<TestBackend, 3, 2, 2>(tensor);
}

// ---------------------------------------------------------------------
// Random Tensors (expected determinants from PyTorch)
// ---------------------------------------------------------------------

#[test]
fn test_det_random_2x3x3_pytorch_comparison() {
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data(
        [
            [
                [0.8823, 0.9150, 0.3829],
                [0.9593, 0.3904, 0.6009],
                [0.2566, 0.7936, 0.9408],
            ],
            [
                [0.1332, 0.9346, 0.5936],
                [0.8694, 0.5677, 0.7411],
                [0.4294, 0.8854, 0.5739],
            ],
        ],
        &device,
    );
    let det_tensor = det::<TestBackend, 3, 2, 1>(tensor);
    // Expected determinants computed by PyTorch
    let expected = TestTensor::<1>::from_data([-0.5283, 0.0993], &device);
    let tolerance = Tolerance::default().set_half_precision_absolute(5e-3);
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}

#[test]
fn test_det_random_2x2x4x4_pytorch_comparison() {
    let device = Default::default();
    let tensor = TestTensor::<4>::from_data(
        [
            [
                [
                    [0.7713, 0.0208, 0.6336, 0.7488],
                    [0.4985, 0.2248, 0.1981, 0.7605],
                    [0.1691, 0.0883, 0.6854, 0.9534],
                    [0.0039, 0.5122, 0.8126, 0.6125],
                ],
                [
                    [0.7218, 0.2919, 0.9178, 0.7146],
                    [0.5425, 0.1422, 0.3733, 0.6741],
                    [0.4418, 0.4340, 0.6178, 0.5131],
                    [0.6503, 0.6011, 0.8052, 0.6985],
                ],
            ],
            [
                [
                    [0.2922, 0.0077, 0.5508, 0.5940],
                    [0.4512, 0.3444, 0.3599, 0.6226],
                    [0.8929, 0.8856, 0.1539, 0.8586],
                    [0.7683, 0.8819, 0.9567, 0.1534],
                ],
                [
                    [0.4885, 0.6690, 0.9967, 0.7013],
                    [0.5040, 0.2374, 0.1123, 0.1505],
                    [0.3819, 0.4428, 0.0835, 0.7866],
                    [0.6515, 0.7998, 0.1242, 0.5865],
                ],
            ],
        ],
        &device,
    );
    let det_tensor = det::<TestBackend, 4, 3, 2>(tensor);
    // Expected determinants computed by PyTorch
    let expected = TestTensor::<2>::from_data([[-0.1527, -0.0053], [0.0307, -0.1039]], &device);
    let tolerance = Tolerance::default().set_absolute(1e-4);
    det_tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), tolerance);
}
