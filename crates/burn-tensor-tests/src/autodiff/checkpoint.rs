use crate::*;
use burn_tensor::{Bool, Tensor, TensorData};

#[test]
fn test_autodiff_checkpoint_complicated_computation() {
    let data_0 = TensorData::from([[0.0, 7.0], [7.0, 7.0]]);
    let data_1 = TensorData::from([[0.1, 7.0], [7.0, 7.0]]);
    let data_2 = TensorData::from([[0.2, 7.0], [7.0, 7.0]]);
    let data_3 = TensorData::from([[0.3, 7.0], [7.0, 7.0]]);
    let data_4 = TensorData::from([[0.4, 7.0], [7.0, 7.0]]);

    let device = Default::default();
    let tensor_0 = TestAutodiffTensor::<2>::from_data(data_0, &device).require_grad();
    let tensor_1 = TestAutodiffTensor::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();
    let tensor_3 = TestAutodiffTensor::from_data(data_3, &device).require_grad();
    let tensor_4 = TestAutodiffTensor::from_data(data_4, &device).require_grad();

    let tensor_5 = compute_bound_eager(tensor_0, tensor_1);
    let tensor_6 = compute_bound_lazy(tensor_2, tensor_3.clone());
    let tensor_7 = memory_bound_eager(tensor_3, tensor_4);
    let tensor_8 = compute_bound_lazy(tensor_6, tensor_7.clone());
    let tensor_9 = memory_bound_eager_scalar(tensor_7, 11.);
    let tensor_10 = memory_bound_lazy(tensor_5, tensor_8.clone());
    let tensor_11 = memory_bound_lazy(tensor_8, tensor_9);
    let tensor_12 = compute_bound_lazy(tensor_10, tensor_11);

    assert_checkpoint(tensor_12);
}

#[test]
fn test_autodiff_checkpoint_with_missing_requirement() {
    let data_0 = TensorData::from([[0.0, 7.0], [7.0, 7.0]]);
    let data_1 = TensorData::from([[0.1, 7.0], [7.0, 7.0]]);

    let device = Default::default();
    let tensor_0 = TestAutodiffTensor::<2>::from_data(data_0, &device).require_grad();
    let tensor_1 = TestAutodiffTensor::from_data(data_1, &device); // does not require_grad

    let tensor_2 = memory_bound_eager(tensor_0, tensor_1);
    let tensor_3 = memory_bound_eager_scalar(tensor_2.clone(), 11.);
    let tensor_4 = memory_bound_eager_scalar(tensor_2.clone(), 11.);
    let tensor_5 = compute_bound_lazy(tensor_3, tensor_4);
    let tensor_6 = compute_bound_eager_scalar(tensor_5.clone(), 11.);
    let tensor_7 = memory_bound_eager(tensor_5, tensor_2);
    let tensor_8 = memory_bound_eager(tensor_6, tensor_7);

    assert_checkpoint(tensor_8);
}

#[test]
fn test_autodiff_checkpoint_with_many_duplicates() {
    let data_0 = TensorData::from([[4.0, 7.0], [7.0, 7.0]]);

    let device = Default::default();
    let tensor_0 = TestAutodiffTensor::<2>::from_data(data_0, &device).require_grad();

    let tensor_1 = memory_bound_eager(tensor_0.clone(), tensor_0.clone());
    let tensor_2 = compute_bound_eager(tensor_0.clone(), tensor_0.clone());
    let tensor_3 = memory_bound_lazy(tensor_0.clone(), tensor_0.clone());
    let tensor_4 = compute_bound_lazy(tensor_0.clone(), tensor_0.clone());

    let tensor_5 = memory_bound_eager(tensor_1.clone(), tensor_0.clone());
    let tensor_6 = memory_bound_eager(tensor_0.clone(), tensor_5.clone());
    let tensor_7 = compute_bound_lazy(tensor_3.clone(), tensor_5.clone());
    let tensor_8 = compute_bound_eager(tensor_4.clone(), tensor_2.clone());
    let tensor_9 = memory_bound_lazy(tensor_6, tensor_7);
    let tensor_10 = memory_bound_eager(tensor_0, tensor_9);
    let tensor_11 = memory_bound_eager_scalar(tensor_10, 9.);
    let tensor_12 = compute_bound_lazy(tensor_8, tensor_11);

    assert_checkpoint(tensor_12);
}

#[test]
fn test_autodiff_checkpoint_with_long_chain_of_eager_memory_bound() {
    let data_0 = TensorData::from([[0.0, 7.0], [7.0, 7.0]]);
    let data_1 = TensorData::from([[0.1, 7.0], [7.0, 7.0]]);
    let data_2 = TensorData::from([[0.2, 7.0], [7.0, 7.0]]);
    let data_3 = TensorData::from([[0.3, 7.0], [7.0, 7.0]]);
    let data_4 = TensorData::from([[0.4, 7.0], [7.0, 7.0]]);

    let device = Default::default();
    let tensor_0 = TestAutodiffTensor::<2>::from_data(data_0, &device).require_grad();
    let tensor_1 = TestAutodiffTensor::from_data(data_1, &device);
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();
    let tensor_3 = TestAutodiffTensor::from_data(data_3, &device).require_grad();
    let tensor_4 = TestAutodiffTensor::from_data(data_4, &device).require_grad();

    let tensor_5 = memory_bound_eager(tensor_0, tensor_1.clone());
    let tensor_6 = memory_bound_eager(tensor_5, tensor_2);
    let tensor_7 = memory_bound_eager(tensor_6, tensor_3);
    let tensor_8 = memory_bound_eager(tensor_7, tensor_4);
    let tensor_9 = memory_bound_lazy(tensor_8, tensor_1);

    assert_checkpoint(tensor_9)
}

#[test]
fn test_autodiff_checkpoint_half_sub_graph_not_tracked() {
    let data_0 = TensorData::from([[0.0, 7.0], [7.0, 7.0]]);
    let data_1 = TensorData::from([[0.1, 7.0], [7.0, 7.0]]);
    let data_2 = TensorData::from([[0.2, 7.0], [7.0, 7.0]]);
    let data_3 = TensorData::from([[0.3, 7.0], [7.0, 7.0]]);
    let data_4 = TensorData::from([[0.4, 7.0], [7.0, 7.0]]);
    let data_5 = TensorData::from([[0.5, 7.0], [7.0, 7.0]]);

    let device = Default::default();
    let tensor_0 = TestAutodiffTensor::<2>::from_data(data_0, &device);
    let tensor_1 = TestAutodiffTensor::from_data(data_1, &device);
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device);
    let tensor_3 = TestAutodiffTensor::from_data(data_3, &device).require_grad();
    let tensor_4 = TestAutodiffTensor::from_data(data_4, &device).require_grad();
    let tensor_5 = TestAutodiffTensor::from_data(data_5, &device).require_grad();

    let tensor_6 = memory_bound_lazy(tensor_0, tensor_1);
    let tensor_7 = compute_bound_eager(tensor_6, tensor_2);

    let tensor_8 = memory_bound_eager(tensor_3, tensor_4);
    let tensor_9 = compute_bound_lazy(tensor_8, tensor_5);

    let tensor_10 = compute_bound_lazy(tensor_7, tensor_9);

    assert_checkpoint(tensor_10);
}

#[test]
fn test_autodiff_checkpoint_very_complex() {
    let data_0 = TensorData::from([[0.0, 7.0], [7.0, 7.0]]);
    let data_1 = TensorData::from([[0.1, 7.0], [7.0, 7.0]]);
    let data_2 = TensorData::from([[0.2, 7.0], [7.0, 7.0]]);
    let data_3 = TensorData::from([[0.3, 7.0], [7.0, 7.0]]);
    let data_4 = TensorData::from([[0.4, 7.0], [7.0, 7.0]]);

    let device = Default::default();
    let tensor_0 = TestAutodiffTensor::<2>::from_data(data_0, &device).require_grad();
    let tensor_1 = TestAutodiffTensor::from_data(data_1, &device);
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();
    let tensor_3 = TestAutodiffTensor::from_data(data_3, &device).require_grad();
    let tensor_4 = TestAutodiffTensor::from_data(data_4, &device).require_grad();

    let tensor_5 = memory_bound_eager_scalar(tensor_0, 8.);
    let tensor_6 = memory_bound_lazy(tensor_5.clone(), tensor_1.clone());
    let tensor_7 = compute_bound_lazy(tensor_6.clone(), tensor_6);
    let tensor_8 = memory_bound_lazy(tensor_1.clone(), tensor_5.clone());
    let tensor_9 = memory_bound_eager_scalar(tensor_7.clone(), 7.);
    let tensor_10 = compute_bound_eager(tensor_5, tensor_8);
    let tensor_11 = memory_bound_eager(tensor_2.clone(), tensor_9);
    let tensor_12 = memory_bound_lazy(tensor_2.clone(), tensor_2);
    let tensor_13 = compute_bound_eager(tensor_10.clone(), tensor_11);
    let tensor_14 = compute_bound_eager_scalar(tensor_3, 8.);
    let tensor_15 = compute_bound_lazy(tensor_4, tensor_12);
    let tensor_16 = memory_bound_lazy(tensor_10, tensor_7);
    let tensor_17 = compute_bound_lazy(tensor_13, tensor_1);
    let tensor_18 = memory_bound_eager(tensor_15, tensor_16);
    let tensor_19 = compute_bound_eager(tensor_14, tensor_17);
    let tensor_20 = memory_bound_lazy(tensor_18, tensor_19);
    let tensor_21 = memory_bound_eager_scalar(tensor_20, 8.);

    assert_checkpoint(tensor_21)
}

fn assert_checkpoint<const D: usize>(tensor: TestAutodiffTensor<D>) {
    // Assert is not explicit here, but the test can fail
    // - when a tensor is actually required more than n_required, it won't be found and will panic
    // - when a tensor is actually required less than n_required, the backward states map won't be
    //   empty and will fail the assertion within the backward code, same for retro_forwards
    tensor.backward();
}

// Does not save its state and does not need its parents
fn memory_bound_eager<const D: usize>(
    tensor_a: TestAutodiffTensor<D>,
    tensor_b: TestAutodiffTensor<D>,
) -> TestAutodiffTensor<D> {
    tensor_a.add(tensor_b)
}
fn memory_bound_eager_scalar<const D: usize>(
    tensor_a: TestAutodiffTensor<D>,
    b: f32,
) -> TestAutodiffTensor<D> {
    tensor_a.add_scalar(b)
}

// Saves its own state and does not need its parents
fn compute_bound_eager<const D: usize>(
    tensor_a: TestAutodiffTensor<D>,
    tensor_b: TestAutodiffTensor<D>,
) -> TestAutodiffTensor<D> {
    let mask = Tensor::<TestAutodiffBackend, D, Bool>::empty(tensor_a.shape(), &tensor_a.device());
    tensor_a.mask_where(mask, tensor_b)
}
fn compute_bound_eager_scalar<const D: usize>(
    tensor_a: TestAutodiffTensor<D>,
    b: f32,
) -> TestAutodiffTensor<D> {
    let mask = Tensor::<TestAutodiffBackend, D, Bool>::empty(tensor_a.shape(), &tensor_a.device());
    tensor_a.mask_fill(mask, b)
}

// Does not save its state and needs its parents
fn memory_bound_lazy<const D: usize>(
    tensor_a: TestAutodiffTensor<D>,
    tensor_b: TestAutodiffTensor<D>,
) -> TestAutodiffTensor<D> {
    tensor_a.mul(tensor_b)
}

// Saves its own state and needs its parents
fn compute_bound_lazy<const D: usize>(
    tensor_a: TestAutodiffTensor<D>,
    tensor_b: TestAutodiffTensor<D>,
) -> TestAutodiffTensor<D> {
    tensor_a.matmul(tensor_b)
}
