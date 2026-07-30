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
