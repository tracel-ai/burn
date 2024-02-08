use burn_tensor::{Data, Tensor};
use burn_wgpu::AutoGraphicsApi;

use crate::Autodiff;

pub type TestBackend = burn_wgpu::Wgpu<AutoGraphicsApi, f32, i32>;
pub type TestAutodiffBackend = Autodiff<TestBackend>;
pub type TestAutodiffTensor<const D: usize> = burn_tensor::Tensor<TestAutodiffBackend, D>;

// #[test]
// fn test_div_not_broken() {
//     let device = Default::default();
//     let t1 = Tensor::<AutodiffTestBackend, 2>::from_data([[99., 99.], [99., 99.]], &device);
//     let t2 = Tensor::<AutodiffTestBackend, 2>::from_data([[11., 11.], [11., 11.]], &device);
//     let t3 = t1 / t2;
//     let expected = Tensor::<AutodiffTestBackend, 2>::from_data([[9., 9.], [9., 9.]], &device);
//     t3.to_data().assert_approx_eq(&expected.to_data(), 3);
// }

// #[test]
// fn test_div_not_broken_with_require_grad() {
//     let device = Default::default();
//     let t1 = Tensor::<AutodiffTestBackend, 2>::from_data([[99., 99.], [99., 99.]], &device)
//         .require_grad();
//     let t2 = Tensor::<AutodiffTestBackend, 2>::from_data([[11., 11.], [11., 11.]], &device)
//         .require_grad();
//     let t3 = t1 / t2;
//     let expected = Tensor::<AutodiffTestBackend, 2>::from_data([[9., 9.], [9., 9.]], &device);
//     t3.to_data().assert_approx_eq(&expected.to_data(), 3);
// }

// #[test]
// fn test_div_not_broken_with_backward() {
//     let device = Default::default();
//     let t1 = Tensor::<AutodiffTestBackend, 2>::from_data([[99., 99.], [99., 99.]], &device)
//         .require_grad();
//     let t2 = Tensor::<AutodiffTestBackend, 2>::from_data([[11., 11.], [11., 11.]], &device)
//         .require_grad();
//     let t3 = t1.clone() / t2.clone();
//     let gradients = t3.backward();
//     println!("{:?}", t1.grad(&gradients).unwrap().to_data());
//     println!("{:?}", t2.grad(&gradients).unwrap().to_data());
//     println!("{:?}", t3.grad(&gradients).unwrap().to_data());
//     // let expected = Tensor::<AutodiffTestBackend, 2>::from_data([[9., 9.], [9., 9.]], &device);
//     // t3.to_data().assert_approx_eq(&expected.to_data(), 3);
//     assert!(false);
// }

#[test]
fn should_diff_div() {
    let data_1 = Data::from([1.0, 7.0]);
    let data_2 = Data::from([4.0, 7.0]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().div(tensor_2.clone());
    let grads = tensor_3.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1
        .to_data()
        .assert_approx_eq(&Data::from([0.25, 0.1429]), 3);
    grad_2
        .to_data()
        .assert_approx_eq(&Data::from([-0.0625, -0.1429]), 3);
}

#[test]
fn should_diff_mul() {
    let data_1 = Data::from([1.0, 7.0]);
    let data_2 = Data::from([4.0, 7.0]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::from_data(data_1.clone(), &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2.clone(), &device).require_grad();

    let tensor_3 = tensor_1.clone().mul(tensor_2.clone());
    let grads = tensor_3.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    assert_eq!(grad_1.to_data(), data_2);
    assert_eq!(grad_2.to_data(), data_1);
    assert_eq!(tensor_3.into_data(), Data::from([4.0, 49.0]));
}

#[test]
fn should_diff_mul_tree() {
    // (ab)(cd)
    let data_a = Data::from([1.0, 7.0]);
    let data_b = Data::from([2.0, 7.0]);
    let data_c = Data::from([3.0, 7.0]);
    let data_d = Data::from([4.0, 7.0]);

    let device = Default::default();
    let tensor_a = TestAutodiffTensor::from_data(data_a, &device).require_grad();
    let tensor_b = TestAutodiffTensor::from_data(data_b, &device).require_grad();
    let tensor_c = TestAutodiffTensor::from_data(data_c, &device).require_grad();
    let tensor_d = TestAutodiffTensor::from_data(data_d, &device).require_grad();

    let tensor_e = tensor_a.clone().mul(tensor_b.clone());
    let tensor_f = tensor_c.clone().mul(tensor_d.clone());
    let tensor_g = tensor_e.mul(tensor_f);

    let grads = tensor_g.backward();
    let grad_a = tensor_a.grad(&grads).unwrap().to_data();
    let grad_b = tensor_b.grad(&grads).unwrap().to_data();
    let grad_c = tensor_c.grad(&grads).unwrap().to_data();
    let grad_d = tensor_d.grad(&grads).unwrap().to_data();

    let expected_a = Data::from([24.0, 343.0]);
    let expected_b = Data::from([12.0, 343.0]);
    let expected_c = Data::from([8.0, 343.0]);
    let expected_d = Data::from([6.0, 343.0]);

    assert_eq!(grad_a, expected_a);
    assert_eq!(grad_b, expected_b);
    assert_eq!(grad_c, expected_c);
    assert_eq!(grad_d, expected_d);

    // FAILS because the leaves are not put in the checkpointer
}

#[test]
fn should_diff_mul_div_tree() {
    // For this test please put div checkpointed/stateful and mul lazy
    // (a/b)(c/d)
    let data_a = Data::from([1.0, 7.0]);
    let data_b = Data::from([2.0, 7.0]);
    let data_c = Data::from([3.0, 7.0]);
    let data_d = Data::from([4.0, 7.0]);

    let device = Default::default();
    let tensor_a = TestAutodiffTensor::from_data(data_a, &device).require_grad();
    let tensor_b = TestAutodiffTensor::from_data(data_b, &device).require_grad();
    let tensor_c = TestAutodiffTensor::from_data(data_c, &device).require_grad();
    let tensor_d = TestAutodiffTensor::from_data(data_d, &device).require_grad();

    let tensor_e = tensor_a.clone().div(tensor_b.clone());
    let tensor_f = tensor_c.clone().div(tensor_d.clone());
    let tensor_g = tensor_e.mul(tensor_f);

    let grads = tensor_g.backward();
    let grad_a = tensor_a.grad(&grads).unwrap().to_data();
    let grad_b = tensor_b.grad(&grads).unwrap().to_data();
    let grad_c = tensor_c.grad(&grads).unwrap().to_data();
    let grad_d = tensor_d.grad(&grads).unwrap().to_data();

    let expected_a = Data::from([0.375, 0.1429]);
    let expected_b = Data::from([-0.1875, -0.1429]);
    let expected_c = Data::from([0.125, 0.1429]);
    let expected_d = Data::from([-0.0938, -0.1429]);

    grad_a.assert_approx_eq(&expected_a, 3);
    grad_b.assert_approx_eq(&expected_b, 3);
    grad_c.assert_approx_eq(&expected_c, 3);
    grad_d.assert_approx_eq(&expected_d, 3);
}

#[test]
fn should_diff_mul_div_tree_with_reuse() {
    // For this test please put div checkpointed/stateful and mul lazy
    // (a/b)(b/c)
    let data_a = Data::from([1.0, 7.0]);
    let data_b = Data::from([2.0, 7.0]);
    let data_c = Data::from([3.0, 7.0]);

    let device = Default::default();
    let tensor_a = TestAutodiffTensor::from_data(data_a, &device).require_grad();
    let tensor_b = TestAutodiffTensor::from_data(data_b, &device).require_grad();
    let tensor_c = TestAutodiffTensor::from_data(data_c, &device).require_grad();

    let tensor_e = tensor_a.clone().div(tensor_b.clone());
    let tensor_f = tensor_b.clone().div(tensor_c.clone());
    let tensor_g = tensor_e.mul(tensor_f);

    let grads = tensor_g.backward();
    let grad_a = tensor_a.grad(&grads).unwrap().to_data();
    let grad_b = tensor_b.grad(&grads).unwrap().to_data();
    let grad_c = tensor_c.grad(&grads).unwrap().to_data();

    let expected_a = Data::from([0.3333, 0.1429]);
    let expected_b = Data::from([0., 0.]);
    let expected_c = Data::from([-0.1111, -0.1429]);

    grad_a.assert_approx_eq(&expected_a, 3);
    grad_b.assert_approx_eq(&expected_b, 3);
    grad_c.assert_approx_eq(&expected_c, 3);
}

// TODO
// - n_required: should be the number of children of a node.
//     - The parent cannot know in advance how many
//     - If children are stateful they don't increment it -> not true, stateful means they are able to compute their own backward, but they might do retroforward for others
//          - If children are CHECKPOINTED they don't increment it?? -> but if they are state_lazy?
//     - Then the condition for not incrementing is that they are BOTH STATEFUL AND CHECKPOINTED (they won't ask for it neither for themselves nor their children)
//          - Sadly, the state of the previous cannot be prevented from being copied as we learn too late it's useless
//          - But both stateful and checkpointed is probably not so common
//     - If incremented, by how much?
//          - How many decrementations will there be?
//     - If n_required is always zero, would be nice if not copied
// - If parents are not tracked they should not be put in the state as Some()
//     - Maybe something related to n_required?
// - Cleanup
//     - The Options in ops prep are not nice
//     - The match at the beginning of backward
//     - The Box needed in .recompute()
//     - Both get states and retrieve outputs are called within an ops. confusing
// - Check if all works fine
//     - Make several test trees, with many variations, including sharing a parent node
//     - Check if the checkpointer extend is well done and there's only one
// - Is node_tree redundant?
