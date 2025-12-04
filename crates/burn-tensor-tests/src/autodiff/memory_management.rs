use crate::*;
use burn_tensor::{Tensor, TensorData};

#[test]
fn test_mm_independent_trees() {
    let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    let device = Default::default();

    // First tree
    let tensor_0 = TestAutodiffTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_1 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
    let tensor_3 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();

    let tensor_4 = tensor_0 * tensor_1;
    let tensor_5 = tensor_2 * tensor_3;
    let tensor_6 = tensor_4 * tensor_5;

    // Second tree
    let tensor_7 = TestAutodiffTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_8 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
    let tensor_9 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
    let tensor_10 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();

    let tensor_11 = tensor_7.clone() * tensor_8.clone();
    let tensor_12 = tensor_9.clone() * tensor_10.clone();
    let tensor_13 = tensor_11 * tensor_12;

    let _grads = tensor_6.backward();
    let grads = tensor_13.backward();

    assert!(tensor_7.grad(&grads).is_some());
    assert!(tensor_8.grad(&grads).is_some());
    assert!(tensor_9.grad(&grads).is_some());
    assert!(tensor_10.grad(&grads).is_some());
}

#[test]
#[should_panic]
fn test_mm_crossover_trees_root_unavailable() {
    let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    let device = Default::default();

    // First tree
    let tensor_0 = TestAutodiffTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_1 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
    let tensor_3 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();

    let tensor_4 = tensor_0 * tensor_1;
    let tensor_5 = tensor_2 * tensor_3;
    let tensor_6 = tensor_4.clone() * tensor_5;

    // Second tree
    let tensor_7 = TestAutodiffTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_8 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();

    let tensor_9 = tensor_7.clone() * tensor_8.clone();
    let tensor_10 = tensor_4 * tensor_9;

    let _grads = tensor_6.backward();
    let _grads = tensor_10.backward();
}

#[test]
fn test_mm_crossover_trees_with_referred_subtree() {
    let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    let device = Default::default();

    // First tree
    let tensor_0 = TestAutodiffTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_1 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
    let tensor_3 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();

    let tensor_4 = tensor_0 * tensor_1;
    let tensor_5 = tensor_2 * tensor_3;
    let tensor_6 = tensor_4.clone() * tensor_5;

    // Second tree
    let tensor_7 = TestAutodiffTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_8 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();

    let tensor_9 = tensor_7.clone() * tensor_8.clone();
    let _tensor_10 = tensor_4 * tensor_9.clone();

    let _grads = tensor_6.backward();
    let _grads = tensor_9.backward();
}

#[test]
fn test_mm_three_crossover_trees_last_still_usable() {
    let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    let device = Default::default();

    // First tree
    let tensor_0 = TestAutodiffTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_1 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
    let tensor_3 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();

    let tensor_4 = tensor_0 * tensor_1;
    let tensor_5 = tensor_2 * tensor_3;
    let tensor_6 = tensor_4 * tensor_5.clone();

    // Third tree
    let tensor_7 = TestAutodiffTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_8 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
    let tensor_9 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
    let tensor_10 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();

    let tensor_11 = tensor_7 * tensor_8;
    let tensor_12 = tensor_9 * tensor_10;
    let tensor_13 = tensor_11 * tensor_12.clone();

    // Second tree (in between)
    let _tensor_14 = tensor_5 * tensor_12;

    let _grads = tensor_6.backward();
    let _grads = tensor_13.backward();
}

#[test]
#[should_panic]
fn test_mm_three_crossover_trees_middle_one_unavailable() {
    let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    let device = Default::default();

    // First tree
    let tensor_0 = TestAutodiffTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_1 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
    let tensor_3 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();

    let tensor_4 = tensor_0 * tensor_1;
    let tensor_5 = tensor_2 * tensor_3;
    let tensor_6 = tensor_4 * tensor_5.clone();

    // Third tree
    let tensor_7 = TestAutodiffTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_8 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
    let tensor_9 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
    let tensor_10 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();

    let tensor_11 = tensor_7 * tensor_8;
    let tensor_12 = tensor_9 * tensor_10;
    let _tensor_13 = tensor_11 * tensor_12.clone();

    // Second tree (in between)
    let tensor_14 = tensor_5 * tensor_12;

    let _grads = tensor_6.backward();
    let _grads = tensor_14.backward();
}

#[test]
fn test_mm_self_referencing_tree() {
    let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    let device = Default::default();

    // First tree
    let tensor_0 = TestAutodiffTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_1 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data.clone(), &device).require_grad();

    let tensor_3 = tensor_0 * tensor_1;
    let tensor_5 = tensor_2 * tensor_3.clone();
    let tensor_6 = tensor_3 * tensor_5;

    let _grads = tensor_6.backward();
}

#[test]
fn test_mm_with_non_impacting_detach() {
    let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    let device = Default::default();
    let tensor_1 =
        Tensor::<TestAutodiffBackend, 2>::from_data(data.clone(), &device).require_grad();
    let tensor_2 =
        Tensor::<TestAutodiffBackend, 2>::from_data(data.clone(), &device).require_grad();
    let tensor_3 = Tensor::<TestAutodiffBackend, 2>::from_data(data, &device).require_grad();

    let tensor_4 = tensor_1.clone() * tensor_2.clone();
    let tensor_5 = tensor_4.detach() * tensor_3.clone();

    let grads = tensor_5.backward();
    assert!(tensor_3.grad(&grads).is_some());
}

#[test]
fn test_mm_with_missing_require_grad_after_cleanup() {
    let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    let device = Default::default();

    let tensor_1 =
        Tensor::<TestAutodiffBackend, 2>::from_data(data.clone(), &device).require_grad();
    let tensor_2 = Tensor::<TestAutodiffBackend, 2>::from_data(data.clone(), &device);
    let tensor_3 = Tensor::<TestAutodiffBackend, 2>::from_data(data.clone(), &device);

    let tensor_4 = tensor_1.clone() * tensor_2.clone();
    let tensor_5 = tensor_4 * tensor_3.clone();

    // Trivial backward, just to trigger cleanup
    Tensor::<TestAutodiffBackend, 2>::from_data(data, &device)
        .require_grad()
        .backward();

    let grads = tensor_5.backward();
    assert!(tensor_1.grad(&grads).is_some());
    assert!(tensor_2.grad(&grads).is_none());
    assert!(tensor_3.grad(&grads).is_none());
}

#[test]
fn test_mm_with_detach_after_cleanup() {
    let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    let device = Default::default();

    let tensor_1 =
        Tensor::<TestAutodiffBackend, 2>::from_data(data.clone(), &device).require_grad();
    let tensor_2 =
        Tensor::<TestAutodiffBackend, 2>::from_data(data.clone(), &device).require_grad();
    let tensor_3 =
        Tensor::<TestAutodiffBackend, 2>::from_data(data.clone(), &device).require_grad();

    let tensor_4 = tensor_1.clone() * tensor_2.clone();
    let tensor_5 = tensor_4 * tensor_3.clone().detach();

    // Trivial backward, just to trigger cleanup
    Tensor::<TestAutodiffBackend, 2>::from_data(data, &device)
        .require_grad()
        .backward();

    let grads = tensor_5.backward();
    assert!(tensor_1.grad(&grads).is_some());
    assert!(tensor_2.grad(&grads).is_some());
    assert!(tensor_3.grad(&grads).is_none());
}

#[test]
#[should_panic]
fn test_mm_deletables_propagate_well() {
    let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    let device = Default::default();

    let tensor_0 =
        Tensor::<TestAutodiffBackend, 2>::from_data(data.clone(), &device).require_grad();
    let tensor_1 =
        Tensor::<TestAutodiffBackend, 2>::from_data(data.clone(), &device).require_grad();

    let tensor_2 = tensor_0 * tensor_1;
    let tensor_3 = tensor_2.clone().exp();
    let _tensor_4 = tensor_3.clone().log();

    let _grads = tensor_2.backward();

    // We are testing that after backward on tensor_2, not only the leaf tensor_4 is deleted, but
    // the intermediate tensor_3 as well
    let _grads = tensor_3.backward();
}

#[test]
fn test_mm_node_explored_once_can_still_be_tagged_as_useful_when_found_again_deeper() {
    let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    let device = Default::default();

    // The test has 50% chance of starting with leaf tensor_8 instead of tensor_4, which is not informative
    // By repeating it many times it becomes almost impossible that it passes if it shouldn't
    for _ in 0..12 {
        let tensor_0 =
            Tensor::<TestAutodiffBackend, 2>::from_data(data.clone(), &device).require_grad();
        let tensor_1 =
            Tensor::<TestAutodiffBackend, 2>::from_data(data.clone(), &device).require_grad();

        let tensor_2 = tensor_1.clone().exp();
        let tensor_3 = tensor_0.exp();
        let _tensor_4 = tensor_3.clone() * tensor_2.clone();
        let tensor_5 = tensor_2.exp();
        let tensor_6 = tensor_5.exp();
        let tensor_7 = tensor_6.exp();
        let tensor_8 = tensor_7.exp();

        // tensor_2 should be tagged unknown through the leaf tensor_4, then useful through the leaf tensor_8
        // which should happen after because tensor_2 is deeper from tensor_8 point of view and we're in breadth first search
        tensor_3.backward();
        let grads = tensor_8.backward();

        assert!(tensor_1.grad(&grads).is_some());
    }
}
