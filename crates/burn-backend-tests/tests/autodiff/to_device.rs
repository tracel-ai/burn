use super::*;
use burn_tensor::TensorData;

#[test]
fn to_device_on_the_same_device_keeps_a_trainable_leaf() {
    let device = AutodiffDevice::new();
    let source = TestTensor::<1>::from_floats([2.0, 3.0], &device).require_grad();
    let moved = source.clone().to_device(&device);

    assert!(moved.is_require_grad());

    // The moved leaf is reused across two backward passes, as a parameter is across training
    // steps while the source stays alive.
    for _ in 0..2 {
        let grads = moved.clone().mul_scalar(2.0).sum().backward();
        moved
            .grad(&grads)
            .unwrap()
            .into_data()
            .assert_eq(&TensorData::from([2.0f32, 2.0]), false);
    }
}
