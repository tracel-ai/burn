use burn_tensor::tensor::backend::ndarray::NdArrayBackend;
use burn_tensor::tensor::backend::tch::TchBackend;
use burn_tensor::tensor::{backend::autodiff::ADTensor, Backend, Tensor};
use burn_tensor::tensor::{ops::*, Data, Distribution, Shape};
use std::time::{Duration, SystemTime};

fn loss<B: Backend>(x: &Tensor<2, B>, y: &Tensor<2, B>) -> Tensor<2, B> {
    x.matmul(y)
}

fn my_func<BForward: Backend>(
    x: Tensor<2, BForward>,
    y: Tensor<2, BForward>,
) -> (Data<BForward::E, 2>, Data<BForward::E, 2>, Duration) {
    let start = SystemTime::now();

    let x = ADTensor::from_tensor(x);
    let y = ADTensor::from_tensor(y);

    let loss = x.matmul(&y);

    let grads = loss.backward();

    let x_grad = grads.wrt(&x).expect("x gradient defined");
    let y_grad = grads.wrt(&y).expect("y gradient defined");

    let end = SystemTime::now();
    let duration = end.duration_since(start).unwrap();

    (x_grad.to_data(), y_grad.to_data(), duration)
}

fn run<B: Backend>(x: Data<B::E, 2>, y: Data<B::E, 2>, device: B::Device) {
    let (x_grad, y_grad, duration) = my_func::<B>(B::from_data(x, device), B::from_data(y, device));

    println!("--- {} ---", B::name());
    println!("took: {} ns", duration.as_nanos());
    println!("x_grad: {}", x_grad);
    println!("y_grad: {}", y_grad);
}

fn main() {
    let x = Data::<f32, 2>::random(Shape::new([2, 3]), Distribution::Standard);
    let y = Data::<f32, 2>::random(Shape::new([3, 1]), Distribution::Standard);

    println!("x: {}", x);
    println!("y: {}", y);

    run::<NdArrayBackend<f32>>(
        x.clone(),
        y.clone(),
        burn_tensor::tensor::backend::ndarray::Device::Cpu,
    );
    run::<TchBackend<f32>>(
        x.clone(),
        y.clone(),
        burn_tensor::tensor::backend::tch::Device::Cpu,
    );
}
