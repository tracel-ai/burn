use std::time::{Duration, SystemTime};

use burn::tensor::*;
use burn_tensor::backend::{
    autodiff::{ADCompatibleTensor, ADElement, ADTensor},
    ndarray::NdArrayTensor,
    tch::TchTensor,
    TchDevice,
};

fn my_func<T: ADCompatibleTensor<P, D>, P: ADElement, const D: usize>(
    x: T,
    y: T,
) -> (Data<P, D>, Data<P, D>, Duration) {
    let start = SystemTime::now();

    let x = ADTensor::from_tensor(x);
    let y = ADTensor::from_tensor(y);

    let z = x.matmul(&y);

    let grads = z.backward();

    let x_grad = grads.wrt(&x).expect("x gradient defined");
    let y_grad = grads.wrt(&y).expect("y gradient defined");

    let end = SystemTime::now();
    let duration = end.duration_since(start).unwrap();

    (x_grad.to_data(), y_grad.to_data(), duration)
}

fn run() {
    let x = Data::<f32, 2>::random(Shape::new([2, 3]), Distribution::Standard);
    let y = Data::<f32, 2>::random(Shape::new([3, 1]), Distribution::Standard);

    let (x_grad_ndarray, y_grad_ndarray, duration_ndarray) = my_func(
        NdArrayTensor::from_data(x.clone()),
        NdArrayTensor::from_data(y.clone()),
    );

    let device = TchDevice::Cpu;
    let (x_grad_tch, y_grad_tch, duration_tch) = my_func(
        TchTensor::from_data(x.clone(), device),
        TchTensor::from_data(y.clone(), device),
    );

    println!("x: {}", x);
    println!("y: {}", y);

    println!("--- ndarray ---");
    println!("took: {} ns", duration_ndarray.as_nanos());
    println!("x_grad: {}", x_grad_ndarray);
    println!("y_grad: {}", y_grad_ndarray);

    println!("--- tch ---");
    println!("took: {} ns", duration_tch.as_nanos());
    println!("x_grad: {}", x_grad_tch);
    println!("y_grad: {}", y_grad_tch);
}

fn main() {
    run()
}
