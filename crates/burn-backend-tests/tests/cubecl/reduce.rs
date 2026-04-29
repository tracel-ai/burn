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
