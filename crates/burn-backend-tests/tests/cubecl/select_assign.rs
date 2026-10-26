use super::*;
use burn_tensor::Distribution;
use burn_tensor::{DType, Device, IndexingUpdateOp, TensorData, Tolerance};

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
fn select_add_3d_dim1_with_strided_values() {
    let device = Device::default();
    let tensor = TestTensor::<3>::zeros([2, 2, 2], &device);
    let values = TestTensor::<3>::from_data(
        [
            [[1., 2., 3.], [4., 5., 6.]],
            [[7., 8., 9.], [10., 11., 12.]],
        ],
        &device,
    )
    .swap_dims(1, 2);
    let indices = TestTensorInt::<1>::from_data([1, 0, 1], &device);

    tensor
        .select_assign(1, indices, values, IndexingUpdateOp::Add)
        .into_data()
        .assert_eq(
            &TensorData::from([[[2., 5.], [4., 10.]], [[8., 11.], [16., 22.]]]),
            false,
        );
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
    use cubek::cubecl::{features::AtomicUsage, ir::Type};

    let dtype = DType::BF16;
    let device = Device::default();
    if !device.supports_dtype(dtype) {
        return;
    }
    let burn_dispatch::DispatchDevice::Cube(cube_device) = device.as_dispatch() else {
        return;
    };
    let client = cube_device.client();
    let properties = client.properties();
    // The serial fallback intentionally retains the input accumulation dtype.
    if properties.hardware.num_cpu_cores.is_some()
        || !properties
            .atomic_type_usage(Type::atomic(burn_backend::cubecl::dtype_to_elem_type(
                DType::F32,
            )))
            .contains(AtomicUsage::Add)
    {
        return;
    }

    // Repeated unit increments stall at 256 in BF16.
    let tensor = TestTensor::<1>::zeros([2], (&device, dtype));
    let value = TestTensor::<1>::ones([4096], (&device, dtype));
    let indices = TestTensorInt::<1>::zeros([4096], &device);

    let out = tensor.select_assign(0, indices, value, IndexingUpdateOp::Add);
    assert_eq!(out.dtype(), dtype);
    out.into_data()
        .convert::<f32>()
        .assert_eq(&TensorData::from([4096., 0.]), false);
}

#[test]
fn select_add_noncontiguous_sliced_view_updates_all_rows() {
    let device = Device::default();
    // Row 1 starts at physical offset 256, beyond the view's logical length of 256.
    let tensor = TestTensor::<2>::zeros([4, 256], &device).slice([0..2, 0..128]);
    let indices = TestTensorInt::<1>::from_data([1, 0, 1], &device);
    let value = TestTensor::<2>::ones([3, 128], &device);

    tensor
        .select_assign(0, indices, value, IndexingUpdateOp::Add)
        .into_data()
        .assert_eq(&TensorData::from([[1.; 128], [2.; 128]]), false);
}

#[test]
fn select_add_preserves_flex32_dtype() {
    let device = Device::default();
    if !device.supports_dtype(DType::Flex32) {
        return;
    }
    let tensor = TestTensor::<1>::zeros([2], (&device, DType::Flex32));
    let value = TestTensor::<1>::full([3], 2., (&device, DType::Flex32));
    let indices = TestTensorInt::<1>::from_data([1, 0, 1], &device);

    let out = tensor.select_assign(0, indices, value, IndexingUpdateOp::Add);
    assert_eq!(out.dtype(), DType::Flex32);
    out.into_data()
        .convert::<f32>()
        .assert_eq(&TensorData::from([2., 4.]), false);
}
