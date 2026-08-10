use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{Device, Distribution, Shape};

// Smallest failing case for https://github.com/tracel-ai/burn/issues/5237
#[test]
fn matmul_double_unit_transposed_rhs_should_match_reference() {
    let device = Device::default();
    let device_ref = ReferenceDevice::new();

    let m = 1024;
    let n = 64;
    let k = 8;

    // TODO: force the DoubleUnit autotune choice
    let lhs: Tensor<2> = Tensor::ones(Shape::new([m, k]), &device);
    let rhs: Tensor<2> = Tensor::ones(Shape::new([n, k]), &device);

    let lhs_ref = Tensor::<2>::from_data(lhs.to_data(), &device_ref);
    let rhs_ref = Tensor::<2>::from_data(rhs.to_data(), &device_ref);

    let out = lhs.matmul(rhs.transpose());
    let out_ref = lhs_ref.matmul(rhs_ref.transpose());

    out.into_data()
        .assert_approx_eq::<FloatElem>(&out_ref.into_data(), Tolerance::rel_abs(1e-5, 1e-5));
}

// A `nn::Linear` with `LinearLayout::Col` at batch 1 — the shape reported in
// https://github.com/tracel-ai/burn/issues/5264. The col-major weight makes this a
// `VecMat` problem with a transposed rhs, which selects an entirely different set
// of kernels than the row-major layout does.
#[test]
fn matmul_vec_mat_transposed_rhs_should_match_reference() {
    let device = Device::default();
    let device_ref = ReferenceDevice::new();

    let (m, k, n) = (1, 128, 512);

    let lhs: Tensor<2> = Tensor::random(Shape::new([m, k]), Distribution::Default, &device);
    let rhs: Tensor<2> = Tensor::random(Shape::new([n, k]), Distribution::Default, &device);

    let lhs_ref = Tensor::<2>::from_data(lhs.to_data(), &device_ref);
    let rhs_ref = Tensor::<2>::from_data(rhs.to_data(), &device_ref);

    let out = lhs.matmul(rhs.transpose());
    let out_ref = lhs_ref.matmul(rhs_ref.transpose());

    out.into_data()
        .assert_approx_eq::<FloatElem>(&out_ref.into_data(), Tolerance::rel_abs(1e-4, 1e-5));
}

// A single-row lhs read out of a `[k, 1]` tensor carries strides `[1, 1]`: both dims
// look contiguous, so the layout is only decidable by noticing that the size-1 dim
// can never be indexed past 0. Getting that wrong makes the autotune key describe a
// layout the kernel does not see, and the plan then contains kernels that are
// illegal for the real operand.
#[test]
fn matmul_transposed_operands_single_row_should_match_reference() {
    let device = Device::default();
    let device_ref = ReferenceDevice::new();

    let (m, k, n) = (1, 128, 512);

    // Built on the reference device and transferred, so neither operand is read
    // before the matmul consumes it.
    let lhs_ref: Tensor<2> = Tensor::random(Shape::new([k, m]), Distribution::Default, &device_ref);
    let rhs_ref: Tensor<2> = Tensor::random(Shape::new([n, k]), Distribution::Default, &device_ref);

    let lhs = Tensor::<2>::from_data(lhs_ref.to_data(), &device);
    let rhs = Tensor::<2>::from_data(rhs_ref.to_data(), &device);

    let out = lhs.transpose().matmul(rhs.transpose());
    let out_ref = lhs_ref.transpose().matmul(rhs_ref.transpose());

    out.into_data()
        .assert_approx_eq::<FloatElem>(&out_ref.into_data(), Tolerance::rel_abs(1e-4, 1e-5));
}
