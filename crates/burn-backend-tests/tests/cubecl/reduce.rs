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
    let actual = tensor.argtopk(1, 2);
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
    let actual = tensor.argtopk(0, k);

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
    let actual = tensor.argtopk(0, k);

    let expected = TestTensor::<3>::from_data([[[0, 1], [1, 0]]], &device);

    assert_eq!(actual.shape(), Shape::new([1, 2, 2]));
    actual.into_data().assert_eq(&expected.into_data(), false);
}

#[test]
fn reduction_argtopk_ties() {
    let device = Default::default();

    let tensor = TestTensor::<1>::from_data([5.0, 2.0, 5.0, 5.0], &device);
    let k = 2;
    let actual = tensor.argtopk(0, k);

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
    let actual = tensor.argtopk(dim, k);

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
