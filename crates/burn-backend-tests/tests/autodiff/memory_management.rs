use super::*;
use burn_tensor::TensorData;
#[cfg(feature = "std")]
use burn_tensor::Tolerance;

#[test]
fn test_mm_independent_trees() {
    let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    let device = AutodiffDevice::new();

    // First tree
    let tensor_0 = TestTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_1 = TestTensor::from_data(data.clone(), &device).require_grad();
    let tensor_2 = TestTensor::from_data(data.clone(), &device).require_grad();
    let tensor_3 = TestTensor::from_data(data.clone(), &device).require_grad();

    let tensor_4 = tensor_0 * tensor_1;
    let tensor_5 = tensor_2 * tensor_3;
    let tensor_6 = tensor_4 * tensor_5;

    // Second tree
    let tensor_7 = TestTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_8 = TestTensor::from_data(data.clone(), &device).require_grad();
    let tensor_9 = TestTensor::from_data(data.clone(), &device).require_grad();
    let tensor_10 = TestTensor::from_data(data.clone(), &device).require_grad();

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
    let device = AutodiffDevice::new();

    // First tree
    let tensor_0 = TestTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_1 = TestTensor::from_data(data.clone(), &device).require_grad();
    let tensor_2 = TestTensor::from_data(data.clone(), &device).require_grad();
    let tensor_3 = TestTensor::from_data(data.clone(), &device).require_grad();

    let tensor_4 = tensor_0 * tensor_1;
    let tensor_5 = tensor_2 * tensor_3;
    let tensor_6 = tensor_4.clone() * tensor_5;

    // Second tree
    let tensor_7 = TestTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_8 = TestTensor::from_data(data.clone(), &device).require_grad();

    let tensor_9 = tensor_7.clone() * tensor_8.clone();
    let tensor_10 = tensor_4 * tensor_9;

    let _grads = tensor_6.backward();
    let _grads = tensor_10.backward();
}

#[test]
fn test_mm_crossover_trees_with_referred_subtree() {
    let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    let device = AutodiffDevice::new();

    // First tree
    let tensor_0 = TestTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_1 = TestTensor::from_data(data.clone(), &device).require_grad();
    let tensor_2 = TestTensor::from_data(data.clone(), &device).require_grad();
    let tensor_3 = TestTensor::from_data(data.clone(), &device).require_grad();

    let tensor_4 = tensor_0 * tensor_1;
    let tensor_5 = tensor_2 * tensor_3;
    let tensor_6 = tensor_4.clone() * tensor_5;

    // Second tree
    let tensor_7 = TestTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_8 = TestTensor::from_data(data.clone(), &device).require_grad();

    let tensor_9 = tensor_7.clone() * tensor_8.clone();
    let _tensor_10 = tensor_4 * tensor_9.clone();

    let _grads = tensor_6.backward();
    let _grads = tensor_9.backward();
}

#[test]
fn test_mm_three_crossover_trees_last_still_usable() {
    let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    let device = AutodiffDevice::new();

    // First tree
    let tensor_0 = TestTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_1 = TestTensor::from_data(data.clone(), &device).require_grad();
    let tensor_2 = TestTensor::from_data(data.clone(), &device).require_grad();
    let tensor_3 = TestTensor::from_data(data.clone(), &device).require_grad();

    let tensor_4 = tensor_0 * tensor_1;
    let tensor_5 = tensor_2 * tensor_3;
    let tensor_6 = tensor_4 * tensor_5.clone();

    // Third tree
    let tensor_7 = TestTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_8 = TestTensor::from_data(data.clone(), &device).require_grad();
    let tensor_9 = TestTensor::from_data(data.clone(), &device).require_grad();
    let tensor_10 = TestTensor::from_data(data.clone(), &device).require_grad();

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
    let device = AutodiffDevice::new();

    // First tree
    let tensor_0 = TestTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_1 = TestTensor::from_data(data.clone(), &device).require_grad();
    let tensor_2 = TestTensor::from_data(data.clone(), &device).require_grad();
    let tensor_3 = TestTensor::from_data(data.clone(), &device).require_grad();

    let tensor_4 = tensor_0 * tensor_1;
    let tensor_5 = tensor_2 * tensor_3;
    let tensor_6 = tensor_4 * tensor_5.clone();

    // Third tree
    let tensor_7 = TestTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_8 = TestTensor::from_data(data.clone(), &device).require_grad();
    let tensor_9 = TestTensor::from_data(data.clone(), &device).require_grad();
    let tensor_10 = TestTensor::from_data(data.clone(), &device).require_grad();

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
    let device = AutodiffDevice::new();

    // First tree
    let tensor_0 = TestTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_1 = TestTensor::from_data(data.clone(), &device).require_grad();
    let tensor_2 = TestTensor::from_data(data.clone(), &device).require_grad();

    let tensor_3 = tensor_0 * tensor_1;
    let tensor_5 = tensor_2 * tensor_3.clone();
    let tensor_6 = tensor_3 * tensor_5;

    let _grads = tensor_6.backward();
}

#[test]
fn test_mm_with_non_impacting_detach() {
    let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_2 = TestTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_3 = TestTensor::<2>::from_data(data, &device).require_grad();

    let tensor_4 = tensor_1.clone() * tensor_2.clone();
    let tensor_5 = tensor_4.detach() * tensor_3.clone();

    let grads = tensor_5.backward();
    assert!(tensor_3.grad(&grads).is_some());
}

#[test]
fn test_mm_with_missing_require_grad_after_cleanup() {
    let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    let device = AutodiffDevice::new();

    let tensor_1 = TestTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_2 = TestTensor::<2>::from_data(data.clone(), &device);
    let tensor_3 = TestTensor::<2>::from_data(data.clone(), &device);

    let tensor_4 = tensor_1.clone() * tensor_2.clone();
    let tensor_5 = tensor_4 * tensor_3.clone();

    // Trivial backward, just to trigger cleanup
    TestTensor::<2>::from_data(data, &device)
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
    let device = AutodiffDevice::new();

    let tensor_1 = TestTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_2 = TestTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_3 = TestTensor::<2>::from_data(data.clone(), &device).require_grad();

    let tensor_4 = tensor_1.clone() * tensor_2.clone();
    let tensor_5 = tensor_4 * tensor_3.clone().detach();

    // Trivial backward, just to trigger cleanup
    TestTensor::<2>::from_data(data, &device)
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
    let device = AutodiffDevice::new();

    let tensor_0 = TestTensor::<2>::from_data(data.clone(), &device).require_grad();
    let tensor_1 = TestTensor::<2>::from_data(data.clone(), &device).require_grad();

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
    let device = AutodiffDevice::new();

    // The test has 50% chance of starting with leaf tensor_8 instead of tensor_4, which is not informative
    // By repeating it many times it becomes almost impossible that it passes if it shouldn't
    for _ in 0..12 {
        let tensor_0 = TestTensor::<2>::from_data(data.clone(), &device).require_grad();
        let tensor_1 = TestTensor::<2>::from_data(data.clone(), &device).require_grad();

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

#[test]
#[cfg(not(feature = "ndarray"))]
// NdArray conservatively reports false for can_mut(), even for unique buffers.
fn test_mm_reclaims_abandoned_graph_buffers_after_unrelated_backward() {
    let device = AutodiffDevice::new();
    let lhs = TestTensor::<2>::from_data([[1.0, 2.0]], &device).require_grad();
    let rhs = TestTensor::<2>::from_data([[3.0, 4.0]], &device).require_grad();
    // These handles share the buffers without keeping either autodiff graph alive.
    let lhs_buffer = lhs.clone().inner();
    let rhs_buffer = rhs.clone().inner();
    let abandoned = [(lhs.clone() * lhs).tanh(), (rhs.clone() * rhs).tanh()];
    assert!(!lhs_buffer.can_mut());
    assert!(!rhs_buffer.can_mut());

    // Cleanup must reclaim both independent abandoned graphs and preserve this
    // live graph, even though none of them participates in the unrelated backward.
    let live = TestTensor::<1>::from_data([2.0], &device).require_grad();
    let loss = live.clone() * live.clone();
    drop(abandoned);

    // Orphan cleanup is best-effort and may skip graphs contended by other tests.
    for _ in 0..64 {
        TestTensor::<1>::from_data([1.0], &device)
            .require_grad()
            .backward();
        if lhs_buffer.can_mut() && rhs_buffer.can_mut() {
            break;
        }
        #[cfg(feature = "std")]
        std::thread::yield_now();
    }
    assert!(
        lhs_buffer.can_mut(),
        "abandoned graph retained the lhs buffer"
    );
    assert!(
        rhs_buffer.can_mut(),
        "abandoned graph retained the rhs buffer"
    );
    live.grad(&loss.backward())
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([4.0]), false);
}

#[test]
#[cfg(feature = "std")]
fn test_mm_preserves_reused_parameters_during_concurrent_backward() {
    // Regression for #5573. Each worker reuses its parameters across backward
    // passes; the other workers' backward calls concurrently sweep its graphs.
    const WORKERS: usize = 4;
    const ROUNDS: usize = 32;
    const SIZE: usize = 128;
    let device = AutodiffDevice::new();
    let start = std::sync::Barrier::new(WORKERS);

    std::thread::scope(|scope| {
        for _ in 0..WORKERS {
            let device = device.clone();
            let start = &start;
            scope.spawn(move || {
                let a = TestTensor::<2>::full([SIZE, SIZE], 0.5, &device).require_grad();
                let b = TestTensor::<2>::full([SIZE, SIZE], 0.25, &device).require_grad();
                (a.clone().sum() + b.clone().sum()).backward();
                let expected_a = TensorData::new(
                    vec![2.0 * (1.0 - 0.5f32.tanh().powi(2)); SIZE * SIZE],
                    [SIZE, SIZE],
                );
                let expected_b = TensorData::new(
                    vec![1.0 - 0.25f32.tanh().powi(2); SIZE * SIZE],
                    [SIZE, SIZE],
                );
                start.wait();

                for _ in 0..ROUNDS {
                    let retained = a.clone().tanh();
                    let output = TestTensor::cat(
                        vec![a.clone().tanh(), retained.clone(), b.clone().tanh()],
                        0,
                    );
                    let grads = output.sum().backward();
                    // Losing one of a's branches leaves a plausible but partial
                    // gradient. Losing b's only branch removes its gradient entirely.
                    a.grad(&grads)
                        .expect("missing gradient for reused parameter a")
                        .into_data()
                        .assert_approx_eq::<FloatElem>(&expected_a, Tolerance::default());
                    b.grad(&grads)
                        .expect("missing gradient for reused parameter b")
                        .into_data()
                        .assert_approx_eq::<FloatElem>(&expected_b, Tolerance::default());
                }
            });
        }
    });
}

#[test]
#[cfg(feature = "std")]
fn test_mm_preserves_branches_during_backward_sharing_a_leaf() {
    const SIZE: usize = 128;
    let device = AutodiffDevice::new();
    let expected = TensorData::new(
        vec![2.0 * (1.0 - 0.5f32.tanh().powi(2)); SIZE * SIZE],
        [SIZE, SIZE],
    );
    let expected_other = TensorData::new(vec![1.0f32; SIZE * SIZE], [SIZE, SIZE]);

    for _ in 0..32 {
        // A fresh leaf makes both branches share a graph before either backward.
        let tensor = TestTensor::<2>::full([SIZE, SIZE], 0.5, &device).require_grad();
        let other = tensor.clone().sum();
        let start = std::sync::Barrier::new(2);

        let (output, other_grads) = std::thread::scope(|scope| {
            let worker = scope.spawn(|| {
                start.wait();
                other.backward()
            });
            let input = tensor.clone().tanh();
            start.wait();
            let output = TestTensor::cat(vec![input.clone(), input], 0);
            (output, worker.join().unwrap())
        });

        tensor
            .grad(&other_grads)
            .unwrap()
            .into_data()
            .assert_approx_eq::<FloatElem>(&expected_other, Tolerance::default());
        tensor
            .grad(&output.sum().backward())
            .unwrap()
            .into_data()
            .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
    }
}
