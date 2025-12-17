use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{Distribution, Tensor};

const RANK: usize = 4;
const SHAPE: [usize; RANK] = [2, 4, 8, 16];

#[test]
fn reduction_argmax_should_match_reference_backend() {
    let tensor =
        Tensor::<TestBackend, RANK>::random(SHAPE, Distribution::Default, &Default::default());
    let tensor_ref =
        Tensor::<ReferenceBackend, RANK>::from_data(tensor.to_data(), &Default::default());
    for dim in 0..RANK {
        tensor
            .clone()
            .argmax(dim)
            .into_data()
            .assert_eq(&tensor_ref.clone().argmax(dim).into_data(), false);
    }
}

#[test]
fn reduction_argmin_should_match_reference_backend() {
    let tensor =
        Tensor::<TestBackend, RANK>::random(SHAPE, Distribution::Default, &Default::default());
    let tensor_ref =
        Tensor::<ReferenceBackend, RANK>::from_data(tensor.to_data(), &Default::default());
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
    let tensor =
        Tensor::<TestBackend, RANK>::random(SHAPE, Distribution::Default, &Default::default());
    let tensor_ref =
        Tensor::<ReferenceBackend, RANK>::from_data(tensor.to_data(), &Default::default());
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
    let tensor =
        Tensor::<TestBackend, RANK>::random(SHAPE, Distribution::Default, &Default::default());
    let tensor_ref =
        Tensor::<ReferenceBackend, RANK>::from_data(tensor.to_data(), &Default::default());
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
    let tensor =
        Tensor::<TestBackend, RANK>::random(SHAPE, Distribution::Default, &Default::default());
    let tensor_ref =
        Tensor::<ReferenceBackend, RANK>::from_data(tensor.to_data(), &Default::default());
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
    let tensor =
        Tensor::<TestBackend, RANK>::random(SHAPE, Distribution::Default, &Default::default());
    let tensor_ref =
        Tensor::<ReferenceBackend, RANK>::from_data(tensor.to_data(), &Default::default());
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
    let tensor =
        Tensor::<TestBackend, RANK>::random(SHAPE, Distribution::Default, &Default::default());
    let tensor_ref =
        Tensor::<ReferenceBackend, RANK>::from_data(tensor.to_data(), &Default::default());
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
    let tensor =
        Tensor::<TestBackend, RANK>::random(SHAPE, Distribution::Default, &Default::default());
    let tensor_ref =
        Tensor::<ReferenceBackend, RANK>::from_data(tensor.to_data(), &Default::default());
    tensor
        .clone()
        .sum()
        .into_data()
        .assert_approx_eq::<FloatElem>(&tensor_ref.clone().sum().into_data(), Tolerance::default());
}
