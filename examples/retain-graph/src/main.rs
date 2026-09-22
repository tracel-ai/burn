use burn::prelude::*;
use burn::tensor::{Device, Tensor};

fn main() {
    let device = Device::default().autodiff();

    let a = Tensor::<1>::from_data([3.0, 4.0], &device);
    let b = Tensor::<1>::from_data([5.0, 6.0], &device);

    let a = a.require_grad();
    let b = b.require_grad();

    let result = a.clone()
        .reshape([2, 1])
        .matmul(b.clone().reshape([1, 2]));

    let result1 = result.clone().slice(s![0, 0]);
    let result2 = result.clone().slice(s![1, 0]);

    let grads1 = result1.backward_retain();
    let grads2 = result2.backward_retain();

    // let grads1 = result1.backward();
    // let grads2 = result2.backward();

    let grad1 = a.grad(&grads1).unwrap();
    let grad2 = a.grad(&grads2).unwrap();

    println!("Gradient 1: {}", grad1);
    println!("Gradient 2: {}", grad2);
}