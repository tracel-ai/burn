use burn_tensor::{Device, Distribution, Tensor};

#[test]
fn repeated_backward_after_many_graphs_keeps_gradients() {
    let device = Device::default().autodiff();
    let x = Tensor::<2>::random([8, 4], Distribution::Default, &device);
    let expected: Vec<f32> = {
        let data = x.clone().into_data().try_to_vec::<f32>().unwrap();
        (0..4).map(|j| (0..8).map(|i| data[i * 4 + j]).sum()).collect()
    };

    for _ in 0..64 {
        let w = Tensor::<2>::ones([4, 2], &device).require_grad();
        let y = x.clone().matmul(w.clone()).sum();
        let grads = y.backward();
        let gw = w.grad(&grads).unwrap().into_data().try_to_vec::<f32>().unwrap();
        let mut expected_grad = Vec::with_capacity(8);
        for value in &expected {
            expected_grad.push(*value);
            expected_grad.push(*value);
        }
        assert!(
            gw.iter().zip(&expected_grad).all(|(a, b)| (a - b).abs() < 1e-4),
            "grad mismatch: {gw:?} vs {expected_grad:?}"
        );
    }
}
