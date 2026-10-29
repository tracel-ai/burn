use super::*;
use burn_tensor::Distribution;
use burn_tensor::Shape;
use burn_tensor::Tolerance;

const RANK: usize = 4;
const SHAPE: [usize; RANK] = [2, 4, 8, 16];

#[test]
fn reduction_argmax_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let tensor = TestTensor::<RANK>::random(SHAPE, Distribution::Default, &device);
    let tensor_ref = TestTensor::<RANK>::from_data(tensor.to_data(), &ref_device);
    for dim in 0..RANK {
        tensor
            .clone()
            .argmax(dim)
            .into_data()
            .assert_eq(&tensor_ref.clone().argmax(dim).into_data(), false);
    }
}

#[test]
fn reduction_argtopk_simple() {
    let device = Default::default();

    let tensor = TestTensor::<2>::from_data([[1, 7, 3], [8, 2, 8]], &device);
    let actual = tensor.argtopk(2, -1_i64);
    let expected = TestTensor::<2>::from_data([[1, 2], [0, 2]], &device);

    let output_shape = Shape::new([2, 2]);
    assert_eq!(actual.shape(), output_shape);
    actual.into_data().assert_eq(&expected.into_data(), false);
}

#[test]
fn reduction_argtopk_1d() {
    let device = Default::default();

    let tensor = TestTensor::<1>::from_data([10.0, 50.0, 20.0, 40.0, 30.0], &device);
    let k = 3;
    let actual = tensor.argtopk(k, 0);

    let expected = TestTensor::<1>::from_data([1, 3, 4], &device);

    assert_eq!(actual.shape(), Shape::new([k]));
    actual.into_data().assert_eq(&expected.into_data(), false);
}

#[test]
fn reduction_argtopk_3d_dim0() {
    let device = Default::default();

    // Shape [2, 2, 2]
    let tensor = TestTensor::<3>::from_data(
        [[[10.0, 1.0], [5.0, 20.0]], [[1.0, 10.0], [20.0, 5.0]]],
        &device,
    );
    let k = 1;
    let actual = tensor.argtopk(k, 0);

    let expected = TestTensor::<3>::from_data([[[0, 1], [1, 0]]], &device);

    assert_eq!(actual.shape(), Shape::new([1, 2, 2]));
    actual.into_data().assert_eq(&expected.into_data(), false);
}

#[test]
fn reduction_argtopk_ties() {
    let device = Default::default();

    let tensor = TestTensor::<1>::from_data([5.0, 2.0, 5.0, 5.0], &device);
    let k = 2;
    let actual = tensor.argtopk(k, 0);

    let expected = TestTensor::<1>::from_data([0, 2], &device);

    assert_eq!(actual.shape(), Shape::new([k]));
    actual.into_data().assert_eq(&expected.into_data(), false);
}

#[test]
fn reduction_argtopk_3d_random_complex() {
    let device = Default::default();

    #[rustfmt::skip]
    let tensor = TestTensor::<3>::from_data(
        [
            [
                [0.5, 1.2, 0.8, 3.3],
                [4.4, 2.1, 9.9, 0.1],
                [7.7, 8.8, 6.6, 5.5],
            ],
            [
                [1.1, 0.2, 4.4, 2.2],
                [6.0, 7.0, 5.0, 8.0],
                [3.0, 3.0, 1.0, 2.0],
            ],
        ],
        &device,
    );

    let k = 2;
    let dim = 2;
    let actual = tensor.argtopk(k, dim);

    #[rustfmt::skip]
    let expected = TestTensor::<3>::from_data(
        [
            [
                [3, 1],
                [2, 0],
                [1, 0],
            ],
            [
                [2, 3],
                [3, 1],
                [0, 1],
            ],
        ],
        &device,
    );

    let output_shape = Shape::new([2, 3, 2]);
    assert_eq!(
        actual.shape(),
        output_shape,
        "Output shape should be [2, 3, 2]"
    );
    actual.into_data().assert_eq(&expected.into_data(), false);
}

#[test]
fn reduction_topk_simple() {
    let device = Default::default();

    let tensor = TestTensor::<2>::from_data([[1, 7, 3], [8, 2, 8]], &device);
    let actual = tensor.topk(2, 1);
    let expected = TestTensor::<2>::from_data([[7, 3], [8, 8]], &device);

    let output_shape = Shape::new([2, 2]);
    assert_eq!(actual.shape(), output_shape);
    actual.into_data().assert_eq(&expected.into_data(), false);
}

#[test]
fn reduction_topk_3d_dim0() {
    let device = Default::default();

    // Shape [2, 2, 2]
    let tensor = TestTensor::<3>::from_data(
        [[[10.0, 1.0], [5.0, 20.0]], [[1.0, 10.0], [20.0, 5.0]]],
        &device,
    );
    let k = 1;
    let actual = tensor.topk(k, 0);

    // Max values across dim 0
    let expected = TestTensor::<3>::from_data([[[10.0, 10.0], [20.0, 20.0]]], &device);

    assert_eq!(actual.shape(), Shape::new([1, 2, 2]));
    actual.into_data().assert_eq(&expected.into_data(), false);
}

#[test]
fn reduction_topk_ties() {
    let device = Default::default();

    let tensor = TestTensor::<1>::from_data([5.0, 2.0, 5.0, 5.0], &device);
    let k = 2;
    let actual = tensor.topk(k, 0);

    let expected = TestTensor::<1>::from_data([5.0, 5.0], &device);

    assert_eq!(actual.shape(), Shape::new([k]));
    actual.into_data().assert_eq(&expected.into_data(), false);
}

#[test]
fn reduction_topk_3d_random_complex() {
    let device = Default::default();

    #[rustfmt::skip]
    let tensor = TestTensor::<3>::from_data(
        [
            [
                [0.5, 1.2, 0.8, 3.3],
                [4.4, 2.1, 9.9, 0.1],
                [7.7, 8.8, 6.6, 5.5],
            ],
            [
                [1.1, 0.2, 4.4, 2.2],
                [6.0, 7.0, 5.0, 8.0],
                [3.0, 3.0, 1.0, 2.0],
            ],
        ],
        &device,
    );

    let k = 2;
    let dim = 2;
    let actual = tensor.topk(k, dim);

    #[rustfmt::skip]
    let expected = TestTensor::<3>::from_data(
        [
            [
                [3.3, 1.2],
                [9.9, 4.4],
                [8.8, 7.7],
            ],
            [
                [4.4, 2.2],
                [8.0, 7.0],
                [3.0, 3.0],
            ],
        ],
        &device,
    );

    let output_shape = Shape::new([2, 3, 2]);
    assert_eq!(
        actual.shape(),
        output_shape,
        "Output shape should be [2, 3, 2]"
    );
    actual.into_data().assert_eq(&expected.into_data(), false);
}

#[test]
fn reduction_argmin_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let tensor = TestTensor::<RANK>::random(SHAPE, Distribution::Default, &device);
    let tensor_ref = TestTensor::<RANK>::from_data(tensor.to_data(), &ref_device);
    for dim in 0..RANK {
        tensor
            .clone()
            .argmin(dim)
            .into_data()
            .assert_eq(&tensor_ref.clone().argmin(dim).into_data(), false);
    }
}

#[test]
fn reduction_mean_dim_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let tensor = TestTensor::<RANK>::random(SHAPE, Distribution::Default, &device);
    let tensor_ref = TestTensor::<RANK>::from_data(tensor.to_data(), &ref_device);
    for dim in 0..RANK {
        tensor
            .clone()
            .mean_dim(dim)
            .into_data()
            .assert_approx_eq::<FloatElem>(
                &tensor_ref.clone().mean_dim(dim).into_data(),
                Tolerance::default(),
            );
    }
}

#[test]
fn reduction_mean_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let tensor = TestTensor::<RANK>::random(SHAPE, Distribution::Default, &device);
    let tensor_ref = TestTensor::<RANK>::from_data(tensor.to_data(), &ref_device);
    tensor
        .clone()
        .mean()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &tensor_ref.clone().mean().into_data(),
            Tolerance::default(),
        );
}

#[test]
fn reduction_prod_dim_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let tensor = TestTensor::<RANK>::random(SHAPE, Distribution::Default, &device);
    let tensor_ref = TestTensor::<RANK>::from_data(tensor.to_data(), &ref_device);
    for dim in 0..RANK {
        tensor
            .clone()
            .prod_dim(dim)
            .into_data()
            .assert_approx_eq::<FloatElem>(
                &tensor_ref.clone().prod_dim(dim).into_data(),
                Tolerance::default(),
            );
    }
}

#[test]
fn reduction_prod_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let tensor = TestTensor::<RANK>::random(SHAPE, Distribution::Default, &device);
    let tensor_ref = TestTensor::<RANK>::from_data(tensor.to_data(), &ref_device);
    tensor
        .clone()
        .prod()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &tensor_ref.clone().prod().into_data(),
            Tolerance::default(),
        );
}

#[test]
fn reduction_sum_dim_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let tensor = TestTensor::<RANK>::random(SHAPE, Distribution::Default, &device);
    let tensor_ref = TestTensor::<RANK>::from_data(tensor.to_data(), &ref_device);
    for dim in 0..RANK {
        tensor
            .clone()
            .sum_dim(dim)
            .into_data()
            .assert_approx_eq::<FloatElem>(
                &tensor_ref.clone().sum_dim(dim).into_data(),
                Tolerance::default(),
            );
    }
}

#[test]
fn reduction_sum_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let tensor = TestTensor::<RANK>::random(SHAPE, Distribution::Default, &device);
    let tensor_ref = TestTensor::<RANK>::from_data(tensor.to_data(), &ref_device);
    tensor
        .clone()
        .sum()
        .into_data()
        .assert_approx_eq::<FloatElem>(&tensor_ref.clone().sum().into_data(), Tolerance::default());
}

#[test]
#[ignore = "Impossible to run unless you have tons of VRAM. Also reference backend is broken."]
fn reduction_sum_should_match_reference_backend_64bit() {
    const SHAPE: [usize; RANK] = [33, 512, 512, 512];

    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let tensor = TestTensor::<RANK>::random(SHAPE, Distribution::Default, &device);
    let tensor_ref = TestTensor::<RANK>::from_data(tensor.to_data(), &ref_device);
    let data = tensor.sum().into_data();
    let data_ref = tensor_ref.sum().into_data();
    println!("result: {:?}", data.as_slice::<f32>());
    data.assert_approx_eq::<FloatElem>(&data_ref, Tolerance::default());
}

#[test]
fn reduction_extrema_with_indices_over_a_long_axis_should_match_reference_backend() {
    // Few outputs over a long axis take the cube routine; ties (every value repeats) check that
    // the first index is still the one reported.
    let device = Default::default();
    let ref_device = ReferenceDevice::new();
    let length = 70_000;
    let values: Vec<f32> = (0..3 * length)
        .map(|i| ((i * 7919) % 1000) as f32)
        .collect();
    let tensor =
        TestTensor::<2>::from_data(burn_tensor::TensorData::new(values, [3, length]), &device);
    let tensor_ref = TestTensor::<2>::from_data(tensor.to_data(), &ref_device);
    let (max, max_at) = tensor.clone().max_dim_with_indices(1);
    let (max_ref, max_at_ref) = tensor_ref.clone().max_dim_with_indices(1);
    max.into_data().assert_eq(&max_ref.into_data(), false);
    max_at.into_data().assert_eq(&max_at_ref.into_data(), false);
    let (min, min_at) = tensor.min_dim_with_indices(1);
    let (min_ref, min_at_ref) = tensor_ref.min_dim_with_indices(1);
    min.into_data().assert_eq(&min_ref.into_data(), false);
    min_at.into_data().assert_eq(&min_at_ref.into_data(), false);
}

#[test]
fn reduction_extrema_over_a_long_axis_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();
    let length = 70_000;
    let values: Vec<f32> = (0..3 * length)
        .map(|i| ((i * 7919) % 1000) as f32 - 500.0)
        .collect();
    let tensor =
        TestTensor::<2>::from_data(burn_tensor::TensorData::new(values, [3, length]), &device);
    let tensor_ref = TestTensor::<2>::from_data(tensor.to_data(), &ref_device);
    tensor
        .clone()
        .max_dim(1)
        .into_data()
        .assert_eq(&tensor_ref.clone().max_dim(1).into_data(), false);
    tensor
        .min_dim(1)
        .into_data()
        .assert_eq(&tensor_ref.min_dim(1).into_data(), false);
}

#[test]
fn reduction_sum_and_mean_over_a_long_axis_should_match_reference_backend() {
    // Few outputs over a long axis take the chunked route. Small integers keep every partial sum
    // exact, so any order gives the reference's values.
    let device = Default::default();
    let ref_device = ReferenceDevice::new();
    let length = 70_001;
    let values: Vec<f32> = (0..3 * length).map(|i| ((i * 7919) % 17) as f32).collect();
    let tensor =
        TestTensor::<2>::from_data(burn_tensor::TensorData::new(values, [3, length]), &device);
    let tensor_ref = TestTensor::<2>::from_data(tensor.to_data(), &ref_device);
    tensor
        .clone()
        .sum_dim(1)
        .into_data()
        .assert_eq(&tensor_ref.clone().sum_dim(1).into_data(), false);
    tensor
        .clone()
        .mean_dim(1)
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &tensor_ref.clone().mean_dim(1).into_data(),
            Tolerance::default(),
        );
    // The same axis strided: the transposed tensor reduced over its first axis.
    tensor
        .transpose()
        .sum_dim(0)
        .into_data()
        .assert_eq(&tensor_ref.transpose().sum_dim(0).into_data(), false);
}
