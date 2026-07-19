use super::*;
use burn_tensor::{Device, IndexingUpdateOp, TensorData, s};

/// Regression for burn-cubecl scatter_nd: values offset used `update_idx * values.stride(0)`.
/// When `values.shape[0] == 1`, stride(0) is the full tensor volume, so update_idx >= 1 OOB-reads
/// values and can raise CUDA_ERROR_ILLEGAL_ADDRESS. BSRoformer ONNX export uses this layout.
#[test]
fn scatter_nd_should_work_with_leading_batch_dim_one() {
    let device = Device::default();
    let ref_device = ReferenceDevice::new();

    let data = TestTensor::<4>::zeros([1, 2, 4, 2], &device);

    let mut indices_vec = Vec::new();
    for i in 0..2usize {
        for j in 0..4usize {
            indices_vec.extend_from_slice(&[0i64, i as i64, j as i64, 0i64]);
        }
    }
    let indices = TestTensorInt::<5>::from_data(
        TensorData::new(indices_vec.clone(), [1, 2, 4, 1, 4]),
        &device,
    );

    let mut values_vec = Vec::new();
    for i in 0..2usize {
        for j in 0..4usize {
            values_vec.push(((i * 4 + j + 1) * 10) as f32);
        }
    }
    let values =
        TestTensor::<4>::from_data(TensorData::new(values_vec.clone(), [1, 2, 4, 1]), &device);

    let data_ref = TestTensor::<4>::from_data(data.to_data(), &ref_device);
    let indices_ref = TestTensorInt::<5>::from_data(
        TensorData::new(indices_vec, [1, 2, 4, 1, 4]),
        &ref_device,
    );
    let values_ref =
        TestTensor::<4>::from_data(TensorData::new(values_vec, [1, 2, 4, 1]), &ref_device);

    let actual = data.scatter_nd(indices, values, IndexingUpdateOp::Assign);
    let expected = data_ref.scatter_nd(indices_ref, values_ref, IndexingUpdateOp::Assign);

    expected
        .into_data()
        .assert_eq(&actual.into_data(), false);
}

#[test]
fn scatter_nd_should_work_with_non_contiguous_values() {
    let device = Device::default();
    let ref_device = ReferenceDevice::new();

    let data = TestTensor::<4>::zeros([1, 2, 4, 2], &device);

    let mut indices_vec = Vec::new();
    for i in 0..2usize {
        for j in 0..4usize {
            indices_vec.extend_from_slice(&[0i64, i as i64, j as i64, 0i64]);
        }
    }
    let indices = TestTensorInt::<5>::from_data(
        TensorData::new(indices_vec.clone(), [1, 2, 4, 1, 4]),
        &device,
    );

    let mut values_vec = Vec::new();
    for i in 0..2usize {
        for j in 0..4usize {
            values_vec.push(((i * 4 + j + 1) * 10) as f32);
        }
    }

    // Embed values in a wider tensor and take a stepped slice so shape stays [1, 2, 4, 1]
    // (valid for ScatterNd) while strides are non-unit / non-contiguous.
    let mut wide_vec = vec![0.0f32; 16];
    for i in 0..2usize {
        for j in 0..4usize {
            wide_vec[i * 8 + j * 2] = ((i * 4 + j + 1) * 10) as f32;
        }
    }
    let values = TestTensor::<4>::from_data(TensorData::new(wide_vec, [1, 2, 8, 1]), &device)
        .slice_dim(2, s![0..8; 2]);

    let data_ref = TestTensor::<4>::from_data(data.to_data(), &ref_device);
    let indices_ref = TestTensorInt::<5>::from_data(
        TensorData::new(indices_vec, [1, 2, 4, 1, 4]),
        &ref_device,
    );
    // NdArray scatter_nd requires contiguous values; same logical content as the stepped view.
    let values_ref =
        TestTensor::<4>::from_data(TensorData::new(values_vec, [1, 2, 4, 1]), &ref_device);

    let actual = data.scatter_nd(indices, values, IndexingUpdateOp::Assign);
    let expected = data_ref.scatter_nd(indices_ref, values_ref, IndexingUpdateOp::Assign);

    expected
        .into_data()
        .assert_eq(&actual.into_data(), false);
}
