use super::*;
use burn_tensor::Distribution;
use burn_tensor::{Device, IndexingUpdateOp, Tolerance};

#[test]
fn select_add_should_match_reference_2d_dim0() {
    select_add_matches_reference(0, [256, 6]);
}

#[test]
fn select_add_should_match_reference_2d_dim1() {
    select_add_matches_reference(1, [6, 256]);
}

#[test]
fn select_add_should_match_reference_3d_dim0() {
    select_add_matches_reference(0, [256, 6, 6]);
}

#[test]
fn select_add_should_match_reference_3d_dim1() {
    select_add_matches_reference(1, [6, 256, 6]);
}

#[test]
fn select_add_should_match_reference_3d_dim2() {
    select_add_matches_reference(2, [6, 6, 256]);
}

#[test]
fn select_mul_should_match_reference_2d_dim0() {
    select_mul_matches_reference(0, [256, 6]);
}

#[test]
fn select_mul_should_match_reference_2d_dim1() {
    select_mul_matches_reference(1, [6, 256]);
}

fn select_add_matches_reference<const D: usize>(dim: usize, shape: [usize; D]) {
    let device = Device::default();
    let ref_device = ReferenceDevice::new();

    device.seed(0);

    let tensor = TestTensor::<D>::random(shape, Distribution::Default, &device);
    let value = TestTensor::<D>::random(shape, Distribution::Default, &device);
    let indices = TestTensorInt::<1>::random(
        [shape[dim]],
        Distribution::Uniform(0., shape[dim] as f64),
        &device,
    );
    let tensor_ref = TestTensor::<D>::from_data(tensor.to_data(), &ref_device);
    let value_ref = TestTensor::<D>::from_data(value.to_data(), &ref_device);
    let indices_ref = TestTensorInt::<1>::from_data(indices.to_data(), &ref_device);

    let actual = tensor.select_assign(dim, indices, value, IndexingUpdateOp::Add);
    let expected = tensor_ref.select_assign(dim, indices_ref, value_ref, IndexingUpdateOp::Add);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}

fn select_mul_matches_reference<const D: usize>(dim: usize, shape: [usize; D]) {
    let device = Device::default();
    let ref_device = ReferenceDevice::new();

    device.seed(0);

    let tensor = TestTensor::<D>::random(shape, Distribution::Default, &device);
    let value = TestTensor::<D>::random(shape, Distribution::Default, &device);
    let indices = TestTensorInt::<1>::arange(0..shape[dim] as i64, &device);

    let tensor_ref = TestTensor::<D>::from_data(tensor.to_data(), &ref_device);
    let value_ref = TestTensor::<D>::from_data(value.to_data(), &ref_device);
    let indices_ref = TestTensorInt::<1>::from_data(indices.to_data(), &ref_device);

    let actual = tensor.select_assign(dim, indices, value, IndexingUpdateOp::Mul);
    let expected = tensor_ref.select_assign(dim, indices_ref, value_ref, IndexingUpdateOp::Mul);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}

#[test]
fn select_add_repeated_indices_preserves_bf16_precision() {
    let device = Device::default();
    if !device.supports_dtype(burn_tensor::DType::BF16) {
        return;
    }

    let n = 4096usize;
    let d = 16usize;

    let tensor = TestTensor::<2>::zeros([4, d], &device).cast(burn_tensor::DType::BF16);
    let value = TestTensor::<2>::ones([n, d], &device).cast(burn_tensor::DType::BF16);
    let indices = TestTensorInt::<1>::zeros([n], &device);

    let out = tensor
        .select_assign(0, indices, value, IndexingUpdateOp::Add)
        .cast(burn_tensor::DType::F32);
    let out_vec: Vec<f32> = out.into_data().to_vec().unwrap();
    assert!((out_vec[0] - 4096.0).abs() < 1e-3, "got {}", out_vec[0]);
    assert_eq!(out_vec[d], 0.0);
}

#[test]
fn select_add_noncontiguous_sliced_view_updates_all_rows() {
    let device = Device::default();
    let ref_device = ReferenceDevice::new();

    // Slicing `[4, 256]` to `[0..2, 0..128]` produces a uniquely owned `[2, 128]` view with
    // strides `[256, 1]`: logical `len()` is `256`, while row 1 starts at physical offset `256`.
    let base = TestTensor::<2>::zeros([4, 256], &device);
    let base_ref = TestTensor::<2>::from_data(base.to_data(), &ref_device);
    let sliced = base.slice([0..2, 0..128]);
    let sliced_ref = base_ref.slice([0..2, 0..128]);

    let indices = TestTensorInt::<1>::from_data([1, 0, 1], &device);
    let indices_ref = TestTensorInt::<1>::from_data([1, 0, 1], &ref_device);
    let value = TestTensor::<2>::ones([3, 128], &device);
    let value_ref = TestTensor::<2>::from_data(value.to_data(), &ref_device);

    let actual = sliced.select_assign(0, indices, value, IndexingUpdateOp::Add);
    let expected = sliced_ref.select_assign(0, indices_ref, value_ref, IndexingUpdateOp::Add);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}

#[test]
fn embedding_backward_preserves_flex32_dtype() {
    let device = Device::default();
    if !device.supports_dtype(burn_tensor::DType::Flex32) {
        return;
    }

    let ad_device = device.autodiff();
    let weights =
        TestTensor::<2>::zeros([8, 16], (&ad_device, burn_tensor::DType::Flex32)).require_grad();
    let indices = TestTensorInt::<2>::zeros([2, 4], &ad_device);

    let out = burn_tensor::module::embedding(weights.clone(), indices);
    let grads = out.sum().backward();
    let grad = weights.grad(&grads).unwrap();
    assert_eq!(grad.dtype(), burn_tensor::DType::Flex32);
    let grad_vec: Vec<f32> = grad.cast(burn_tensor::DType::F32).into_data().to_vec().unwrap();
    assert!((grad_vec[0] - 8.0).abs() < 1e-3, "got {}", grad_vec[0]);
}


